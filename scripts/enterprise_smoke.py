"""First enterprise-shaped TTT experiment: CORD receipts, per-task adaptation.

Implements the ENTERPRISE_EVAL_SPEC minimal smoke: k demonstration receipts
adapt a small instruct model (LoRA TTT via TextPredictor), which then
extracts structured JSON from held-out receipts; scored with field-level
micro-F1 (primary) and canonicalized exact-match (secondary). Writes a
machine-readable artifact for experiments/.

    python scripts/enterprise_smoke.py --data cord_validation.jsonl \
        --model Qwen/Qwen2.5-0.5B-Instruct --k 10 --eval-n 20 --seed 0 \
        --out experiments/cord_smoke_<date>.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arcttt.model import TTTConfig
from arcttt.text_task import from_cord_gt
from arcttt.text_ttt import TextPredictor, score_text_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CORD JSONL from fetch_cord.py")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--k", type=int, default=10, help="demonstration receipts")
    parser.add_argument("--eval-n", type=int, default=20, help="held-out receipts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-seq", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-adapt", action="store_true",
                        help="baseline arm: k-shot prompting, no TTT")
    parser.add_argument("--dump-texts", action="store_true",
                        help="store input/gold/prediction texts per result")
    parser.add_argument("--only-indices", default=None,
                        help="comma-separated eval indices to predict (skip rest)")
    return parser


def main(argv: list[str] | None = None) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = build_parser().parse_args(argv)
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if len(rows) < args.k + args.eval_n:
        raise SystemExit(f"need >= {args.k + args.eval_n} rows, have {len(rows)}")
    task = from_cord_gt(
        rows[: args.k],
        rows[args.k : args.k + args.eval_n],
        task_id=f"cord-k{args.k}-seed{args.seed}",
    )

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    config = TTTConfig(
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        epochs=0 if args.no_adapt else args.epochs,
        max_new_tokens=args.max_new_tokens,
        max_sequence_tokens=args.max_seq,
        gradient_checkpointing=device.type == "cuda",
        shuffle_examples=True,
    )
    predictor = TextPredictor(model, tokenizer, config, device)

    started = time.monotonic()
    predictor.adapt_text(task, shuffle_seeds=(args.seed,))
    adapt_seconds = time.monotonic() - started

    results = []
    exact = 0
    f1_sum = 0.0
    scored = 0
    keep = (
        {int(part) for part in args.only_indices.split(",")}
        if args.only_indices
        else None
    )
    for index in range(len(task.test)):
        if keep is not None and index not in keep:
            continue
        gold = task.test[index].output_text
        assert gold is not None  # CORD eval outputs are not hidden
        texts = predictor.predict_text(task, index, samples=1)
        if not texts:
            results.append({"index": index, "error": "no completion"})
            continue
        score = score_text_output(texts[0], gold)
        scored += 1
        exact += int(score.exact_match)
        f1_sum += score.micro_f1
        entry = {
            "index": index,
            "valid_json": score.valid_json,
            "exact_match": score.exact_match,
            "micro_f1": round(score.micro_f1, 4),
        }
        if args.dump_texts:
            entry["input_text"] = task.test[index].input_text
            entry["gold"] = gold
            entry["prediction"] = texts[0]
        results.append(entry)
    report = {
        "spec": "ENTERPRISE_EVAL_SPEC.md minimal smoke",
        "dataset": "naver-clova-ix/cord-v2 (CC BY 4.0), text-only post-OCR",
        "model": args.model,
        "arm": "kshot-no-ttt" if args.no_adapt else "adapted",
        "k": args.k,
        "eval_n": args.eval_n,
        "seed": args.seed,
        "config": {"rank": args.rank, "alpha": args.alpha, "epochs": config.epochs},
        "adapt_seconds": round(adapt_seconds, 1),
        "exact_match": exact,
        "scored": scored,
        "mean_micro_f1": round(f1_sum / scored, 4) if scored else 0.0,
        "results": results,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "results"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
