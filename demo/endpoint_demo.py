"""Demo: the adaptation endpoint on CORD receipts, before/after TTT.

Two subcommands (run via demo/run_endpoint_demo.sh, which orchestrates both):

    python demo/endpoint_demo.py serve --model Qwen/Qwen2.5-0.5B-Instruct --port 8341
    python demo/endpoint_demo.py demo  --data demo/cord_validation.jsonl --port 8341

`serve` starts the adaptation endpoint over HTTP, reusing the shipped server
machinery unchanged (arcttt.serve.make_handler + http.server.HTTPServer) with
a text-mode service: POST /adapt carries demonstration (input, output) text
pairs plus query inputs; the service runs the shipped text-mode test-time
training path (arcttt.text_ttt.TextPredictor — LoRA injected and trained per
request, exactly the ENTERPRISE_EVAL_SPEC smoke configuration) and returns
the model's completions. `adapt: false` in the payload skips the weight
update (epochs=0; the injected LoRA is zero-initialised, i.e. a true no-op),
giving the honest baseline arm: same prompt, same examples in context, no
test-time training.

`demo` is the client narrative a non-engineer can follow: it POSTs the same
k CORD receipts (default 10) twice — once with adapt:false (BEFORE), once with adapt:true
(AFTER) — queries the same 2 held-out receipts each time, and prints model
output vs gold side by side with field-level micro-F1 and wall-clock timing.

New file only: imports src/arcttt, modifies nothing outside demo/.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import replace
from http.server import HTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arcttt.model import TTTConfig
from arcttt.serve import make_handler
from arcttt.tasks import TaskFormatError
from arcttt.text_task import TextPair, TextTask, from_cord_gt
from arcttt.text_ttt import (
    TextPredictor,
    field_pairs,
    parse_json_object,
    score_text_output,
)

# ---------------------------------------------------------------------------
# server side
# ---------------------------------------------------------------------------


def text_task_from_payload(payload: Any, task_id: str) -> TextTask:
    if not isinstance(payload, dict) or "train" not in payload or "test" not in payload:
        raise TaskFormatError("body must carry train and test lists")

    def pairs(items: Any, split: str, need_output: bool) -> tuple[TextPair, ...]:
        if not isinstance(items, list) or not items:
            raise TaskFormatError(f"{split} must be a non-empty list")
        built = []
        for item in items:
            if not isinstance(item, dict) or "input" not in item:
                raise TaskFormatError(f"{split} pairs need an input string")
            if need_output and "output" not in item:
                raise TaskFormatError(f"{split} pairs need an output string")
            built.append(
                TextPair(input_text=item["input"], output_text=item.get("output"))
            )
        return tuple(built)

    task = TextTask(
        task_id=task_id,
        train=pairs(payload["train"], "train", need_output=True),
        test=pairs(payload["test"], "test", need_output=False),
    )
    task.validate()
    return task


class TextAdaptService:
    """Duck-typed like arcttt.serve.AdaptService, for text tasks.

    Same contract the shipped handler expects: .adapt(payload) -> dict,
    .model_name, .requests_served. Stateless per request: every POST removes
    any prior LoRA, injects a fresh one, and (if adapt is true) trains it on
    the posted examples before answering the queries.
    """

    def __init__(self, model: Any, tokenizer: Any, config: TTTConfig,
                 device: Any, model_name: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.model_name = model_name
        self.requests_served = 0

    def adapt(self, payload: Any) -> dict[str, Any]:
        task = text_task_from_payload(payload, f"request-{self.requests_served}")
        do_adapt = bool(payload.get("adapt", True))
        seed = int(payload.get("seed", 0))
        config = self.config if do_adapt else replace(self.config, epochs=0)
        predictor = TextPredictor(self.model, self.tokenizer, config, self.device)

        started = time.monotonic()
        predictor.adapt_text(task, shuffle_seeds=(seed,))
        adapt_seconds = time.monotonic() - started

        predictions: list[dict[str, Any]] = []
        for index in range(len(task.test)):
            query_started = time.monotonic()
            texts = predictor.predict_text(task, index, samples=1)
            predictions.append(
                {
                    "text": texts[0] if texts else "",
                    "seconds": round(time.monotonic() - query_started, 1),
                }
            )
        self.requests_served += 1
        return {
            "task_id": task.task_id,
            "adapted": do_adapt,
            "adapt_seconds": round(adapt_seconds, 1),
            "seconds": round(time.monotonic() - started, 1),
            "predictions": predictions,
        }


def run_server(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[serve] loading {args.model} on {device.type} ...", flush=True)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    print(f"[serve] model loaded in {time.monotonic() - started:.1f}s", flush=True)

    # Same knobs as scripts/enterprise_smoke.py (the measured CORD smoke).
    config = TTTConfig(
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        epochs=args.epochs,
        max_new_tokens=args.max_new_tokens,
        max_sequence_tokens=args.max_seq,
        gradient_checkpointing=device.type == "cuda",
        shuffle_examples=True,
    )
    service = TextAdaptService(model, tokenizer, config, device, args.model)
    server = HTTPServer(("127.0.0.1", args.port), make_handler(service))
    print(f"[serve] listening on 127.0.0.1:{args.port} | model {args.model}", flush=True)
    server.serve_forever()
    return 0


# ---------------------------------------------------------------------------
# client side (the narrative)
# ---------------------------------------------------------------------------

RULE = "=" * 78
COLUMN = 37  # characters per side-by-side column


def http_json(url: str, payload: dict[str, Any] | None = None,
              timeout: float = 1800.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_for_health(base: str, timeout: float = 600.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        try:
            return http_json(f"{base}/health", timeout=5.0)
        except (urllib.error.URLError, ConnectionError, OSError):
            if time.monotonic() > deadline:
                raise SystemExit("endpoint never came up; is `serve` running?")
            time.sleep(2.0)


def pretty(text: str) -> list[str]:
    """Pretty-print a JSON object with sorted keys; raw text if not JSON."""
    try:
        parsed = parse_json_object(text)
    except TaskFormatError:
        return [line for line in text.splitlines() if line.strip()] or ["(empty)"]
    return json.dumps(parsed, indent=1, sort_keys=True, ensure_ascii=False).splitlines()


def side_by_side(left_title: str, left: list[str],
                 right_title: str, right: list[str]) -> None:
    def clip(line: str) -> str:
        return line if len(line) <= COLUMN else line[: COLUMN - 1] + "…"

    print(f"      {left_title:<{COLUMN}} | {right_title}")
    print(f"      {'-' * COLUMN} | {'-' * COLUMN}")
    for row in range(max(len(left), len(right))):
        l = clip(left[row]) if row < len(left) else ""
        r = clip(right[row]) if row < len(right) else ""
        print(f"      {l:<{COLUMN}} | {r}")


def fields_correct(predicted_text: str, gold_text: str) -> tuple[int, int]:
    gold = field_pairs(parse_json_object(gold_text))
    try:
        predicted = field_pairs(parse_json_object(predicted_text))
    except TaskFormatError:
        return 0, sum(gold.values())
    return sum((predicted & gold).values()), sum(gold.values())


def show_results(task: TextTask, response: dict[str, Any],
                 label: str) -> list[dict[str, Any]]:
    scored = []
    for index, prediction in enumerate(response["predictions"]):
        gold = task.test[index].output_text
        assert gold is not None
        score = score_text_output(prediction["text"], gold)
        correct, total = fields_correct(prediction["text"], gold)
        scored.append({"f1": score.micro_f1, "correct": correct, "total": total,
                       "valid": score.valid_json, "seconds": prediction["seconds"]})
        print()
        print(f"   held-out receipt #{index + 1} — raw OCR text sent to the endpoint:")
        for line in task.test[index].input_text.splitlines():
            print(f"      | {line}")
        print(f"   model answered in {prediction['seconds']}s "
              f"({label}) — recovered {correct}/{total} gold fields; "
              f"field F1 {score.micro_f1:.2f} (F1 also counts extra "
              f"predicted fields against precision)"
              + ("" if score.valid_json else ", output was NOT valid JSON"))
        side_by_side("MODEL OUTPUT", pretty(prediction["text"]),
                     "GOLD (human-labelled)", pretty(gold))
    return scored


def run_demo(args: argparse.Namespace) -> int:
    base = f"http://127.0.0.1:{args.port}"
    overall_started = time.monotonic()

    print(RULE)
    print(" arc-ttt adaptation endpoint — live demo")
    print(" task: receipt OCR text in  →  structured JSON out (CORD-v2, CC BY 4.0)")
    print(RULE)

    health = wait_for_health(base)
    print(f"\n[1] GET {base}/health")
    print(f"    -> {json.dumps(health)}")

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()
            if line.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    k, held = args.k, args.eval_n
    if rows and "gt_parse" in rows[0]:
        task = from_cord_gt(rows[:k], rows[k : k + held],
                            task_id=f"demo-k{k}-seed{args.seed}")
        noun = "real receipts"
    else:
        # Generic tenant format — {"text": ..., "gold": {...}} per line
        # (the same rows the challenge kit consumes), so the demo can run
        # on YOUR documents: --data mine.jsonl.
        from arcttt.text_ttt import json_canonical
        def pair(row: dict) -> TextPair:
            return TextPair(input_text=row["text"],
                            output_text=json_canonical(row["gold"]))
        # gold stays on test pairs for the local scoreboard, exactly like
        # from_cord_gt; the /adapt payload below only ever sends inputs.
        task = TextTask(task_id=f"demo-k{k}-seed{args.seed}",
                        train=tuple(pair(r) for r in rows[:k]),
                        test=tuple(pair(r) for r in rows[k : k + held]))
        task.validate()
        noun = "documents"
    print(f"\n[2] Data: {len(rows)} {noun} on disk. Deterministic split "
          f"(seed {args.seed}):")
    print(f"    {k} {noun} + their labelled JSON = the customer's examples")
    print(f"    {held} held out = the questions we grade the model on")

    payload = {
        "seed": args.seed,
        "train": [{"input": pair.input_text, "output": pair.output_text}
                  for pair in task.train],
        "test": [{"input": pair.input_text} for pair in task.test],
    }

    print(f"\n{RULE}")
    print(" STEP A — BEFORE ADAPTATION")
    print(f"  POST /adapt with adapt:false — the model sees the {args.k} examples in its")
    print("  prompt but its weights are untouched. This is standard few-shot prompting.")
    print(RULE)
    started = time.monotonic()
    before = http_json(f"{base}/adapt", {**payload, "adapt": False})
    print(f"\n   endpoint round-trip {time.monotonic() - started:.1f}s "
          f"(no weight update: {before['adapt_seconds']}s)")
    before_scores = show_results(task, before, "before")

    print(f"\n{RULE}")
    print(" STEP B — ADAPT + RE-ASK")
    print(f"  POST /adapt with adapt:true — same {args.k} examples, but now the endpoint")
    print("  trains a LoRA adapter on them (test-time training) before answering")
    print("  the same 2 held-out receipts with the adapted weights.")
    print(RULE)
    started = time.monotonic()
    after = http_json(f"{base}/adapt", {**payload, "adapt": True})
    print(f"\n   endpoint round-trip {time.monotonic() - started:.1f}s "
          f"(weight update on {len(task.train)} examples: {after['adapt_seconds']}s)")
    after_scores = show_results(task, after, "after")

    print(f"\n{RULE}")
    print(" SCOREBOARD — same receipts, same prompt; only the weights changed")
    print(RULE)
    print(f"   {'receipt':<12} {'fields correct':>16} {'field F1':>10}    "
          f"{'fields correct':>16} {'field F1':>10}")
    print(f"   {'':<12} {'BEFORE':>16} {'':>10}    {'AFTER':>16}")
    for index, (b, a) in enumerate(zip(before_scores, after_scores)):
        before_cell = f"{b['correct']}/{b['total']}"
        after_cell = f"{a['correct']}/{a['total']}"
        print(f"   held-out #{index + 1} "
              f"{before_cell:>16} {b['f1']:>10.2f}    "
              f"{after_cell:>16} {a['f1']:>10.2f}")
    mean_before = sum(s["f1"] for s in before_scores) / len(before_scores)
    mean_after = sum(s["f1"] for s in after_scores) / len(after_scores)
    print(f"   {'mean':<12} {'':>16} {mean_before:>10.2f}    {'':>16} {mean_after:>10.2f}")
    print(f"\n   adaptation cost: {after['adapt_seconds']}s on CPU, "
          f"{len(task.train)} examples, one LoRA adapter")
    print(f"   total demo wall time: {time.monotonic() - overall_started:.1f}s "
          "(single CPU, 0.5B-parameter open model)")
    print(f"\n   n={args.eval_n} receipts is a live demo, not an eval. The measured")
    print("   result on this dataset — 4 paired seed/k arms, 20 held-out receipts")
    print("   each — is net-neutral: mean −1.3 F1, range −6.5 to +12.7")
    print("   (experiments/cord_variance_summary_2026-08-08.json). The 0.661 → 0.788")
    print("   pair is the seed-0 favorable arm of that sweep, not the mean.")
    print(RULE)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="start the text-mode adaptation endpoint")
    serve.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    serve.add_argument("--port", type=int, default=8341)
    serve.add_argument("--rank", type=int, default=16)
    serve.add_argument("--alpha", type=int, default=32)
    serve.add_argument("--epochs", type=int, default=1)
    serve.add_argument("--max-new-tokens", type=int, default=512)
    serve.add_argument("--max-seq", type=int, default=4096)
    serve.add_argument("--device", default=None)

    demo = sub.add_parser("demo", help="run the client narrative against the endpoint")
    demo.add_argument("--data", default=str(Path(__file__).parent / "cord_validation.jsonl"))
    demo.add_argument("--port", type=int, default=8341)
    demo.add_argument("--k", type=int, default=10)
    demo.add_argument("--eval-n", type=int, default=2)
    demo.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    return run_server(args) if args.command == "serve" else run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
