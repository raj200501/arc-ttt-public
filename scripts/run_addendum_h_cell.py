#!/usr/bin/env python3
"""Run ONE Addendum H cell: the corpus-DIFFICULTY ablation.

Addendum E answered "is the effect an artifact of one fixed corpus
SHAPE?" by varying geometry. The objection it does not answer -- raised
by an outside technical reader on 2026-08-22, and the strongest one this
project has -- is that the effect may be an artifact of one fixed corpus
DIFFICULTY. Two constants set that difficulty, and neither has ever been
ablated:

  1. `mapping="arbitrary"`: the document label and the JSON key are
     unrelated pseudowords. The generator's own docstring calls this
     "the single most important property". REAL tenant schemas are not
     like this -- "Ship Date:" maps to `ship_date`.
  2. `n_distractors=4`: lines whose labels are outside the schema and
     must be dropped.

The evidence already in the repository is monotone in difficulty and it
runs against us: arbitrary-mapping synthetic -> +46.5; freight waybills
(semantic labels, ordinary decoys) -> +4.14 on a coin-flip sign test;
CORD (real, semantic) -> FAIL at three scales. So this ablation is the
one that decides whether the headline describes adaptation or describes
the generator.

`mapping="mnemonic"` makes the JSON key the SAME token as the document
label. The pool is consumed identically, so the two corpora differ in
NOTHING else: same documents byte-for-byte, same values, same
distractors, same shuffles. Only the key names change.

Both arms carry the same k-shot prompt (spec B.9.1 scoping), both decode
greedy, both load their own model, and each cell stores raw predictions
so `verify_from_primary.py` can re-score it.

    PYTHONPATH=src python3 scripts/run_addendum_h_cell.py \\
        --seed 1 --k 10 --mapping mnemonic \\
        --out experiments/novel_schema_h_0.5b_k10_seed1_mnemonic_2026-08-22.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import platform
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

N_TEST = 20
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def sign_test(deltas: list[float]) -> dict:
    wins = sum(1 for d in deltas if d > 1e-12)
    losses = sum(1 for d in deltas if d < -1e-12)
    ties = len(deltas) - wins - losses
    n = wins + losses
    p = (sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n
         if n else 1.0)
    return {"wins": wins, "losses": losses, "ties": ties, "p_value": p}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--mapping", choices=("arbitrary", "mnemonic"),
                        required=True)
    parser.add_argument("--n-distractors", type=int, default=4)
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

    task, schema = make_task(seed=args.seed, n_train=args.k,
                             n_test=args.n_test,
                             n_distractors=args.n_distractors,
                             mapping=args.mapping)
    gold = [pair.output_text for pair in task.test]

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = TTTConfig(
        lora_rank=16, lora_alpha=32, epochs=1, max_new_tokens=512,
        max_sequence_tokens=args.max_seq, gradient_checkpointing=False,
        chunked_loss_tokens=512,
    )

    def run_arm(adapt: bool) -> tuple[list[dict], float]:
        # Each arm loads its OWN model; a shared object is the easiest way
        # to fake this result.
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
                "prediction": raw,
            })
            print(f"  doc {index + 1}/{len(task.test)} "
                  f"f1={rows[-1]['micro_f1']:.3f}", flush=True)
        del predictor, model
        return rows, seconds

    tag = f"seed={args.seed} k={args.k} mapping={args.mapping} " \
          f"distractors={args.n_distractors}"
    print(f"[H] {tag} arm=kshot", flush=True)
    kshot, _ = run_arm(adapt=False)
    print(f"[H] {tag} arm=adapted", flush=True)
    adapted, adapt_seconds = run_arm(adapt=True)

    deltas = [a["micro_f1"] - b["micro_f1"] for a, b in zip(adapted, kshot)]
    mean_k = sum(r["micro_f1"] for r in kshot) / len(kshot)
    mean_a = sum(r["micro_f1"] for r in adapted) / len(adapted)
    headroom = 1.0 - mean_k
    record = {
        "addendum": "H",
        "what": "corpus-DIFFICULTY ablation: does the effect survive a "
                "label->key mapping that real tenant schemas actually "
                "have, and/or the removal of distractor lines?",
        "preregistered": "ENTERPRISE_EVAL_SPEC.md Addendum H and the "
                         "VERDICT.md row, both committed before this arm "
                         "ran. Readings (a)/(b)/(c)/(u) frozen there.",
        "seed": args.seed,
        "k": args.k,
        "mapping": args.mapping,
        "n_distractors": args.n_distractors,
        "rung": "0.5b",
        "model": args.model,
        "device": "cpu",
        "dtype": "torch.float32",
        "decode": "greedy (samples=1), matched on both arms",
        "serving": "include_demos=True (the gate-1 comparison)",
        "n_test": len(task.test),
        "schema": schema.describe() if hasattr(schema, "describe") else None,
        "adapt_seconds": adapt_seconds,
        "kshot_mean_micro_f1": round(mean_k, 6),
        "adapted_mean_micro_f1": round(mean_a, 6),
        "paired_mean_delta": round(mean_a - mean_k, 6),
        "captured_headroom_fraction": (
            round((mean_a - mean_k) / headroom, 6) if headroom > 1e-9
            else None),
        "sign_test": sign_test(deltas),
        "kshot_valid_json": sum(1 for r in kshot if r["valid_json"]),
        "adapted_valid_json": sum(1 for r in adapted if r["valid_json"]),
        "baseline_saturated": mean_k >= 0.95,
        "results": [
            {"index": i, "kshot": kshot[i]["micro_f1"],
             "adapted": adapted[i]["micro_f1"], "delta": round(deltas[i], 6),
             "kshot_prediction": kshot[i]["prediction"],
             "adapted_prediction": adapted[i]["prediction"]}
            for i in range(len(deltas))
        ],
        "host": {"platform": platform.platform(),
                 "python": platform.python_version()},
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    st = record["sign_test"]
    print(f"[H] {tag}: kshot {mean_k:.4f} -> adapted {mean_a:.4f}  "
          f"delta {mean_a - mean_k:+.4f}  "
          f"{st['wins']}W/{st['losses']}L/{st['ties']}T  -> {out}",
          flush=True)
    if record["baseline_saturated"]:
        print("[H] NOTE: prompted baseline >= 0.95 -- under the frozen "
              "reading (u) this cell is UNINFORMATIVE about the ablation, "
              "not evidence against the effect.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
