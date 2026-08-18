"""ENTERPRISE_EVAL_SPEC Addendum A scaled run: one rung, all arms, resumable.

Executes the FROZEN preregistration (spec Addendum A, 2026-08-08): per rung,
k ∈ {5, 10, 30} × seeds {1, 2, 3} × arms {adapted, kshot} = 18 arms, 20
held-out receipts each, protocol identical to the dev variance sweep EXCEPT
the single preregistered change — vote/rescore ON in both arms (1 greedy +
4 sampled at T=0.7, pooled on canonical-JSON key, count + likelihood
rescored, top-1 submitted; ``text_ttt.predict_text_voted``).

Hyperparameters are frozen (r=16, alpha=32, epochs=1, eval_n=20,
max_new_tokens=512, max_seq 4096 for k<=10 / 8192 for k=30). Each arm writes
its own machine-readable artifact; arms whose artifact already exists are
skipped, so the run resumes across interrupted sessions (the k=30 lesson).

    python scripts/cord_scale_run.py --rung 0.5b \
        --data demo/cord_validation.jsonl --out-dir experiments
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
from arcttt.text_ttt import TextPredictor, predict_text_voted, score_text_output

# Addendum A frozen ladder (licenses verified via HF API 2026-08-08).
RUNGS = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "4b": "Qwen/Qwen3-4B-Instruct-2507",
}
KS = (5, 10, 30)
SEEDS = (1, 2, 3)
ARMS = ("adapted", "kshot")
EVAL_N = 20
POOL_SAMPLES = 5  # 1 greedy + 4 sampled (T=0.7, the model.py config default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", required=True, choices=sorted(RUNGS))
    parser.add_argument("--data", required=True, help="CORD JSONL from fetch_cord.py")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--date", default=None, help="artifact date stamp (YYYY-MM-DD)")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated k:seed:arm filters, e.g. 10:1:adapted,10:2:adapted",
    )
    return parser


def arm_configs(only: str | None) -> list[tuple[int, int, str]]:
    wanted = None
    if only:
        wanted = set()
        for part in only.split(","):
            k, seed, arm = part.split(":")
            wanted.add((int(k), int(seed), arm))
    combos = [
        (k, seed, arm)
        for k in KS
        for seed in SEEDS
        for arm in ARMS
        if wanted is None or (k, seed, arm) in wanted
    ]
    return combos


def run_arm(
    model: object,
    tokenizer: object,
    device: "object",
    rows: list[dict],
    rung: str,
    k: int,
    seed: int,
    arm: str,
    out_path: Path,
) -> dict:
    import torch

    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)  # per-seed reshuffle, same procedure as the dev sweep
    task = from_cord_gt(
        shuffled[:k],
        shuffled[k : k + EVAL_N],
        task_id=f"cord-scale-{rung}-k{k}-seed{seed}",
    )
    config = TTTConfig(
        lora_rank=16,
        lora_alpha=32,
        epochs=1 if arm == "adapted" else 0,
        max_new_tokens=512,
        max_sequence_tokens=8192 if k == 30 else 4096,
        # memory-for-compute trade, math-identical: the 15GB CPU container
        # SIGKILLs a 4096-token float32 backward without it (08-11)
        gradient_checkpointing=True,
        shuffle_examples=True,
    )
    predictor = TextPredictor(model, tokenizer, config, torch.device(device))
    started = time.monotonic()
    predictor.adapt_text(task, shuffle_seeds=(seed,))
    adapt_seconds = time.monotonic() - started

    results = []
    exact = 0
    f1_sum = 0.0
    scored = 0
    invalid = 0
    no_completion = 0
    for index in range(len(task.test)):
        gold = task.test[index].output_text
        assert gold is not None  # CORD eval outputs are not hidden
        selected = predict_text_voted(predictor, task, index, samples=POOL_SAMPLES)
        if selected is None:
            no_completion += 1
            results.append({"index": index, "error": "no completion"})
            continue
        score = score_text_output(selected, gold)
        scored += 1
        exact += int(score.exact_match)
        invalid += int(not score.valid_json)
        f1_sum += score.micro_f1
        results.append(
            {
                "index": index,
                "valid_json": score.valid_json,
                "exact_match": score.exact_match,
                "micro_f1": round(score.micro_f1, 4),
            }
        )
    report = {
        "spec": "ENTERPRISE_EVAL_SPEC.md Addendum A scaled run (frozen 2026-08-08)",
        "dataset": "naver-clova-ix/cord-v2 (CC BY 4.0), text-only post-OCR",
        "rung": rung,
        "model": RUNGS[rung],
        "arm": arm,
        "k": k,
        "eval_n": EVAL_N,
        "seed": seed,
        "decode": "vote/rescore ON: 1 greedy + 4 sampled (T=0.7), "
        "canonical-JSON pooling, count+likelihood top-1",
        "config": {
            "rank": 16,
            "alpha": 32,
            "epochs": config.epochs,
            "max_new_tokens": 512,
            "max_seq": config.max_sequence_tokens,
        },
        "adapt_seconds": round(adapt_seconds, 1),
        "exact_match": exact,
        "scored": scored,
        "invalid_json": invalid,
        "no_completion": no_completion,
        "mean_micro_f1": round(f1_sum / scored, 4) if scored else 0.0,
        "results": results,
    }
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(out_path)
    return report


def main(argv: list[str] | None = None) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = build_parser().parse_args(argv)
    if not args.date:
        raise SystemExit("--date is required (artifact stamps are explicit, not implicit)")
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    if len(rows) < 30 + EVAL_N:
        raise SystemExit(f"need >= {30 + EVAL_N} rows for k=30 arms, have {len(rows)}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    print(f"rung {args.rung} -> {RUNGS[args.rung]} | device {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(RUNGS[args.rung])
    model = AutoModelForCausalLM.from_pretrained(RUNGS[args.rung], dtype=dtype).to(device)

    done = skipped = 0
    for k, seed, arm in arm_configs(args.only):
        out_path = out_dir / (
            f"cord_scale_{args.rung}_k{k}_seed{seed}_{arm}_{args.date}.json"
        )
        if out_path.exists():
            skipped += 1
            print(f"skip (exists): {out_path.name}", flush=True)
            continue
        started = time.monotonic()
        report = run_arm(
            model, tokenizer, device, rows, args.rung, k, seed, arm, out_path
        )
        done += 1
        print(
            json.dumps(
                {
                    "artifact": out_path.name,
                    "mean_micro_f1": report["mean_micro_f1"],
                    "invalid_json": report["invalid_json"],
                    "wall_seconds": round(time.monotonic() - started, 1),
                }
            ),
            flush=True,
        )
    print(f"rung {args.rung}: {done} arms run, {skipped} resumed/skipped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
