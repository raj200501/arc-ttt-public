#!/usr/bin/env python3
"""Run ONE Addendum G cell: (seed, k) x {prompted baseline, adapted}.

Addendum G sweeps the strength of the prompted baseline by varying the
demonstration count k, and asks whether adaptation's paired advantage is
a decreasing function of that strength. The bar was frozen in
ENTERPRISE_EVAL_SPEC.md Addendum G before any of these arms existed.

Both arms carry the SAME k-shot prompt (spec B.9.1 scoping: adaptation
is measured ON TOP of prompting, never against a bare model), both
decode GREEDY, and both are scored by the same scorer. One cell writes
one artifact holding both arms and their per-document paired deltas.

    python3 scripts/run_addendum_g_cell.py --seed 401 --k 10 \
        --out experiments/novel_schema_g_0.5b_k10_seed401_2026-08-21.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

N_TEST = 20  # frozen in G.3
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--max-seq", type=int, default=8192)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from arcttt.model import TTTConfig
    from arcttt.novel_schema import make_task
    from arcttt.scoring import score_text_output
    from arcttt.text_ttt import TextPredictor

    task, schema = make_task(seed=args.seed, n_train=args.k, n_test=args.n_test)
    gold = [pair.output_text for pair in task.test]

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = TTTConfig(
        lora_rank=16, lora_alpha=32, epochs=1, max_new_tokens=512,
        max_sequence_tokens=args.max_seq, gradient_checkpointing=False,
        chunked_loss_tokens=512,
    )

    def run_arm(adapt: bool) -> tuple[list[dict], float]:
        # Each arm loads its OWN model from the checkpoint. Sharing one
        # model object across the two arms is the single easiest way to
        # fake this result -- an adapter left attached from the adapted
        # arm would silently lift the baseline, or a baseline run after
        # an adapted one would not be a baseline at all. A ~5s reload
        # buys the guarantee, so it is not an optimization worth making.
        torch.manual_seed(1)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32).to(device)
        predictor = TextPredictor(model, tokenizer, config, device)
        seconds = 0.0
        if adapt:
            started = time.monotonic()
            predictor.adapt_text(task)
            seconds = round(time.monotonic() - started, 1)
            print(f"  adapted in {seconds}s", flush=True)
        rows = []
        for index in range(len(task.test)):
            texts = predictor.predict_text(task, index, samples=1,
                                           include_demos=True)
            raw = texts[0] if texts else ""
            score = score_text_output(raw, gold[index])
            rows.append({
                "index": index,
                "valid_json": bool(score.valid_json),
                "micro_f1": float(score.micro_f1) if score.valid_json else 0.0,
            })
            print(f"  doc {index + 1}/{len(task.test)} "
                  f"f1={rows[-1]['micro_f1']:.3f}", flush=True)
        del predictor, model
        return rows, seconds

    print(f"[G] seed={args.seed} k={args.k} arm=kshot", flush=True)
    kshot, _ = run_arm(adapt=False)
    print(f"[G] seed={args.seed} k={args.k} arm=adapted", flush=True)
    adapted, adapt_seconds = run_arm(adapt=True)

    deltas = [a["micro_f1"] - b["micro_f1"] for a, b in zip(adapted, kshot)]
    mean_k = sum(r["micro_f1"] for r in kshot) / len(kshot)
    mean_a = sum(r["micro_f1"] for r in adapted) / len(adapted)
    headroom = 1.0 - mean_k
    record = {
        "addendum": "G",
        "seed": args.seed,
        "k": args.k,
        "rung": "0.5b",
        "model": args.model,
        "device": "cpu",
        "dtype": "torch.float32",
        "decode": "greedy (samples=1), matched on both arms",
        "n_test": len(task.test),
        "schema": schema.describe() if hasattr(schema, "describe") else None,
        "adapt_seconds": adapt_seconds,
        "kshot_mean_micro_f1": round(mean_k, 6),
        "adapted_mean_micro_f1": round(mean_a, 6),
        "paired_mean_delta": round(mean_a - mean_k, 6),
        "captured_headroom_fraction": (
            round((mean_a - mean_k) / headroom, 6) if headroom > 1e-9 else None),
        "kshot_valid_json": sum(1 for r in kshot if r["valid_json"]),
        "adapted_valid_json": sum(1 for r in adapted if r["valid_json"]),
        "per_doc": [
            {"index": i, "kshot": kshot[i]["micro_f1"],
             "adapted": adapted[i]["micro_f1"], "delta": round(deltas[i], 6)}
            for i in range(len(deltas))
        ],
        "host": {"platform": platform.platform(),
                 "python": platform.python_version()},
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[G] seed={args.seed} k={args.k}: kshot {mean_k:.4f} -> "
          f"adapted {mean_a:.4f}  delta {mean_a - mean_k:+.4f}  -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
