#!/usr/bin/env python3
"""Score an adapted checkpoint on the 30 held-out waybills. Addendum Q.

Serves **document-only** -- the schema lives in the adapter weights, which
is exactly how this project's own 0.5B arm is served. That is the whole
point of the comparison: the adapted arm carries no field list and no
demonstrations in its prompt, and the prompted arms it is measured
against carry one or the other.

Fence-stripping is applied here the same way it is applied to every other
arm. There is no flag to turn it off, so the repair cannot be granted
asymmetrically even by mistake.

The comparator is resolved by the RULE frozen in VERDICT.md before the
number it depends on existed: **the better of the two prompted 3B arms**.
This reads both off disk and picks the higher one rather than having a
number typed in.

    PYTHONPATH=src python3 scripts/score_adapted_arm.py \\
        --model Qwen/Qwen2.5-3B-Instruct --adapter /tmp/arunQ/adapter.pt \\
        --out experiments/waybill_adapted_3b_2026-08-25.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import platform
import statistics
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
RAW = REPO / "experiments" / "blind_rehearsal_2026-08-20_raw"

INSTANCE_USD_PER_HOUR = 0.290
TOLERANCE = 0.01          # frozen in VERDICT.md Addendum Q
ALPHA = 0.05              # frozen: the sign test must agree at p <= 0.05

# Every prompted 3B arm on disk. The comparator is whichever scores
# higher -- the RULE was frozen before the k-shot number existed.
#
# The first version of this tuple hardcoded the float32 k-shot FILENAME,
# and that arm OOM-killed and banked under a bf16 name instead -- so the
# resolver silently skipped the missing file, picked the schema arm at
# 0.8958, and this script PRINTED READING (a) against a comparator the
# frozen rule did not select. The k-shot bf16 arm was on disk at 0.9747,
# in the SAME dtype as the adapted arm -- the exact comparison the
# frozen dtype contingency asks for. The verdict flipped to (b) within
# the hour, by the standing rule that where letter and substance
# disagree, the reading that does not flatter us governs. A resolver
# that can quietly lose a comparator to a filename now refuses instead:
# if fewer than two prompted arms resolve, it exits rather than reads.
PROMPTED_3B = ("waybill_scale_rung_3b_schema_2026-08-25.json",
               "waybill_scale_rung_3b_kshot_2026-08-25.json",
               "waybill_scale_rung_3b_kshot_bf16_2026-08-25.json",
               "waybill_scale_rung_3b_schema_bf16_control_2026-08-25.json")


def _rows(name: str) -> list[dict]:
    return [json.loads(line) for line
            in (RAW / name).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _sign_test(deltas: list[float]) -> dict:
    wins = sum(1 for d in deltas if d > 1e-12)
    losses = sum(1 for d in deltas if d < -1e-12)
    ties = len(deltas) - wins - losses
    n = wins + losses

    def tail(k: int) -> float:
        return (sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n
                if n else 1.0)

    return {"wins_adapted": wins, "losses_adapted": losses, "ties": ties,
            "p_adapted_better": round(tail(wins), 4),
            "p_prompted_better": round(tail(losses), 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--dtype", default="float32",
                        choices=("float32", "bfloat16"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--serve-demos", action="store_true",
                        help="ladder rung E5: serve the ADAPTED model with "
                             "the k=20 demonstrations in the prompt")
    parser.add_argument("--ladder", action="store_true",
                        help="bank scores and predictions only; do NOT "
                             "apply Addendum Q's comparator rule or fire "
                             "its readings -- the engineering ladder has "
                             "its own frozen bars and its own reader")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from arcttt.lora import inject_lora
    from arcttt.model import TTTConfig
    from arcttt.scoring import field_micro_f1, parse_json_object
    from arcttt.text_task import TextTaskFormatError
    from arcttt.text_ttt import TextPredictor, text_task_to_messages

    from fence_rescore import strip_fence          # noqa: E402
    from run_challenge import build_task           # noqa: E402

    torch.set_num_threads(4)
    train, holdout = _rows("train.jsonl"), _rows("holdout.jsonl")
    gold = {r["id"]: r["gold"] for r in _rows("gold_holdout.jsonl")}
    task = build_task(train, holdout)

    # ---- resolve the comparator by the frozen RULE ---------------------
    prompted = []
    for name in PROMPTED_3B:
        path = REPO / "experiments" / name
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        preds = record.get("predictions") or {}
        scores = {}
        for doc_id, text in preds.items():
            try:
                obj = parse_json_object(strip_fence(text)[0])
            except TextTaskFormatError:
                obj = None
            scores[doc_id] = 0.0 if obj is None else field_micro_f1(
                obj, gold[doc_id])
        prompted.append({"artifact": name, "mode": record.get("mode"),
                         "mean_fence_stripped": round(
                             statistics.mean(scores.values()), 4),
                         "per_document": scores})
    if len(prompted) < 2:
        raise SystemExit(
            f"only {len(prompted)} prompted 3B arm(s) resolved; the frozen "
            "rule compares against the better of the schema and k-shot "
            "arms, and reading it against a lone arm is how this script "
            "printed a wrong verdict once. Bank the missing arm first.")
    if not prompted:
        raise SystemExit("no prompted 3B arm is banked; the comparator rule "
                         "cannot be resolved and no reading may fire")
    best = max(prompted, key=lambda a: a["mean_fence_stripped"])

    # ---- the adapted arm ------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch.manual_seed(1)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype))
    inject_lora(model, rank=16, alpha=32)
    state = torch.load(args.adapter, map_location="cpu")
    if not state:
        raise SystemExit("adapter.pt contained no tensors")
    missing = model.load_state_dict(state, strict=False)
    if missing.unexpected_keys:
        raise SystemExit(f"adapter keys not in model: "
                         f"{missing.unexpected_keys[:3]}")
    model.eval()
    config = TTTConfig(lora_rank=16, lora_alpha=32, epochs=1,
                       max_new_tokens=args.max_new_tokens,
                       max_sequence_tokens=8192,
                       gradient_checkpointing=False, chunked_loss_tokens=512)
    predictor = TextPredictor(model, tokenizer, config, torch.device("cpu"))

    per_doc, predictions, invalid, batch_seconds = {}, {}, 0, []
    for i, row in enumerate(holdout):
        ids = predictor._prompt_ids(
            text_task_to_messages(task, i,
                                  include_demos=args.serve_demos))
        if ids is None:
            raise SystemExit(f"prompt {i} exceeded the sequence budget")
        began = time.monotonic()
        with torch.no_grad():
            out = model.generate(input_ids=ids,
                                 attention_mask=torch.ones_like(ids),
                                 max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        batch_seconds.append(time.monotonic() - began)
        text = tokenizer.decode(out[0][ids.shape[1]:],
                                skip_special_tokens=True).strip()
        predictions[row["id"]] = text
        try:
            obj = parse_json_object(strip_fence(text)[0])
        except TextTaskFormatError:
            obj, invalid = None, invalid + 1
        per_doc[row["id"]] = round(
            0.0 if obj is None else field_micro_f1(obj, gold[row["id"]]), 4)
        print(f"  {row['id']} {per_doc[row['id']]:.3f}  "
              f"{batch_seconds[-1]:6.1f}s", flush=True)

    mean = statistics.mean(per_doc.values())
    delta = mean - best["mean_fence_stripped"]
    deltas = [per_doc[d] - best["per_document"][d] for d in per_doc]
    sign = _sign_test(deltas)
    median_s = statistics.median(batch_seconds)

    mean_beats = delta >= TOLERANCE
    mean_loses = delta <= -TOLERANCE
    sign_favours_adapted = sign["p_adapted_better"] <= ALPHA
    sign_favours_prompted = sign["p_prompted_better"] <= ALPHA

    if args.ladder:
        # Ladder rung: Addendum Q's readings must not fire for an arm Q
        # never preregistered. The comparator block above still banks
        # every prompted arm's mean for context; the LADDER reader
        # applies the ladder's own frozen bars.
        verdict = "LADDER RUNG — no Addendum Q reading fires; see the "                   "engineering ladder reader"
        why = ("Served " + ("with the k=20 demonstrations"
               if args.serve_demos else "document-only") +
               " through the adapter. The ladder's frozen bar (the best "
               "prompted arm at the same size, two-statistics rule) is "
               "applied by scripts/ladder_reader.py, not here.")
    elif mean_beats and sign_favours_adapted:
        verdict = "(a) ADAPTATION ADDS AT SCALE"
        why = (f"D = {delta:+.4f} clears the {TOLERANCE} tolerance and the "
               f"sign test agrees ({sign['wins_adapted']}W/"
               f"{sign['losses_adapted']}L/{sign['ties']}T, "
               f"p={sign['p_adapted_better']}). Adaptation is a quality "
               "multiplier at 3B, not a small-model crutch.")
    elif mean_loses and sign_favours_prompted:
        verdict = "(c) IT HURTS"
        why = (f"D = {delta:+.4f} and the sign test agrees. Adapting a 3B on "
               "20 documents damages it.")
    elif abs(delta) <= TOLERANCE or not (
            sign_favours_adapted or sign_favours_prompted):
        verdict = "(b) IT WAS A SMALL-MODEL CRUTCH"
        why = (f"D = {delta:+.4f} against a {TOLERANCE} tolerance, sign test "
               f"{sign['wins_adapted']}W/{sign['losses_adapted']}L/"
               f"{sign['ties']}T. Adaptation buys nothing at 3B that the "
               "prompted model did not already have.")
    else:
        verdict = "(d) THE TWO STATISTICS DISAGREE"
        why = (f"Mean D = {delta:+.4f} and the sign test "
               f"({sign['wins_adapted']}W/{sign['losses_adapted']}L/"
               f"{sign['ties']}T) point opposite ways. NO CLAIM is made in "
               "either direction. The sign test is the one to weight here: "
               "it is paired per document, where a mean over 30 documents "
               "moves on one or two outliers.")

    record = {
        "addendum": "engineering-ladder" if args.ladder else "Q",
        "what": "Does adaptation still add anything once the base model is "
                "a 3B? The cell Addendum O never tested.",
        "preregistered": "VERDICT.md Addendum Q, frozen before this ran, "
                         "including the rule that the comparator is the "
                         "better of the two prompted 3B arms.",
        "model": args.model,
        "dtype": args.dtype,
        "serving": ("adapted PLUS k=20 demonstrations in the prompt -- "
                    "the two mechanisms stacked, which no prior arm "
                    "measured" if args.serve_demos else
                    "document-only: no field list and no demonstrations "
                    "in the prompt. The schema is in the adapter weights, "
                    "exactly as our 0.5B arm is served."),
        "fence_policy": "Fence-stripping applied here as to every other "
                        "arm. No flag disables it.",
        "adapted_mean_micro_f1": round(mean, 4),
        "adapted_invalid_json": invalid,
        "comparator_rule": "the better of the two prompted 3B arms, frozen "
                           "before the k-shot number existed",
        "comparator_chosen": best["artifact"],
        "comparator_mode": best["mode"],
        "comparator_mean": best["mean_fence_stripped"],
        "prompted_arms_considered": [
            {k: v for k, v in a.items() if k != "per_document"}
            for a in prompted],
        "D_adapted_minus_best_prompted": round(delta, 4),
        "tolerance": TOLERANCE,
        "sign_test": sign,
        "alpha": ALPHA,
        "verdict": verdict,
        "why": why,
        "our_adapted_0.5b": 0.8833,
        "seconds_per_document_median": round(median_s, 2),
        "cost_per_1k_documents_usd": round(
            median_s * 1000 / 3600 * INSTANCE_USD_PER_HOUR, 4),
        "seconds_per_document": [round(s, 2) for s in batch_seconds],
        "per_document_micro_f1": per_doc,
        "predictions": predictions,
        "scope": "One corpus, thirty agent-authored documents, one seed, "
                 "greedy, CPU, batch 1, one adaptation run on 20 documents.",
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__},
        "verification_level": "PRIMARY -- raw model text stored per document.",
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")
    print(f"\nadapted 3B        {mean:.4f}  ({invalid} invalid)")
    print(f"best prompted 3B  {best['mean_fence_stripped']:.4f}  "
          f"({best['mode']})")
    print(f"our adapted 0.5B  0.8833")
    print(f"D = {delta:+.4f}   sign {sign['wins_adapted']}W/"
          f"{sign['losses_adapted']}L/{sign['ties']}T")
    print(f"{verdict}\n{why}")
    print(f"\nbanked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
