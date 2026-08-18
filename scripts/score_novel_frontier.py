"""Score the agent-harness frontier arm on the novel-schema corpus.

Joins predictions (authored by the frontier model from prompts exported
WITHOUT gold in view — export_novel_prompts.py writes gold to a separate
file the predictor never read) against gold, using the identical scorer
as every other arm (arcttt.text_ttt.score_text_output), and writes one
context artifact with the honest-framing block.

    python3 scripts/score_novel_frontier.py \
        --pairs 1:work/novel_s1_preds.jsonl:work/novel_s1_gold.jsonl ... \
        --out experiments/novel_frontier_baseline_<date>.json --date <date>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arcttt.text_ttt import score_text_output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", required=True,
                        help="seed:preds.jsonl:gold.jsonl per tenant")
    parser.add_argument("--out", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)

    tenants = []
    all_f1: list[float] = []
    for spec in args.pairs:
        seed_s, preds_path, gold_path = spec.split(":")
        seed = int(seed_s)
        preds = {r["index"]: r["prediction"]
                 for r in map(json.loads, Path(preds_path).read_text().splitlines())}
        golds = {r["index"]: r["gold"]
                 for r in map(json.loads, Path(gold_path).read_text().splitlines())}
        if set(preds) != set(golds):
            raise SystemExit(f"seed {seed}: index mismatch preds vs gold")
        results = []
        for index in sorted(golds):
            score = score_text_output(preds[index], golds[index])
            results.append({
                "index": index,
                "valid_json": score.valid_json,
                "exact_match": score.exact_match,
                "micro_f1": round(score.micro_f1, 4),
            })
        f1s = [r["micro_f1"] for r in results]
        all_f1.extend(f1s)
        tenants.append({
            "seed": seed,
            "n": len(results),
            "mean_micro_f1": round(sum(f1s) / len(f1s), 4),
            "exact_match": sum(r["exact_match"] for r in results),
            "valid_json": sum(r["valid_json"] for r in results),
            "results": results,
        })

    report = {
        "artifact": "frontier k-shot baseline, novel-schema corpus (context arm, NOT a gate arm)",
        "date": args.date,
        "spec_relation": (
            "Same corpus construction as Addendum B gate arms "
            "(make_task(seed, n_train=k, n_test=60)); scored on the first 20 "
            "eval docs per tenant (indices 0-19), which are construction-"
            "identical to the kernel arms' docs at those indices. Bounded "
            "subset — labeled as such, never pooled with the 60-doc arms."
        ),
        "model": "claude-fable-5 (frontier LLM, k-shot in-context, no adaptation)",
        "protocol": (
            "Prompts exported by scripts/export_novel_prompts.py with gold "
            "split to a file the predictor never read. Run via Claude agent "
            "harness: the model inferred each tenant's schema mapping from "
            "the 10 in-context demos and authored every prediction; a "
            "parsing script reformatted input documents for reading only. "
            "Single sample per receipt. Scored with "
            "arcttt.text_ttt.score_text_output (same scorer as all arms)."
        ),
        "k": args.k,
        "per_tenant": tenants,
        "pooled_mean_micro_f1": round(sum(all_f1) / len(all_f1), 4),
        "pooled_n": len(all_f1),
        "honest_framing_REQUIRED": (
            "Context arm for the 'why not a frontier API' question — NOT part "
            "of any preregistered gate, and states nothing about the gate "
            "delta. The corpus is verbatim-extraction by design (novelty of "
            "schema, not difficulty of reasoning, is the manipulated "
            "variable), so a frontier model is expected to do well; the "
            "product claim next to this number is cost and data-exposure "
            "(payload asymmetry artifacts), never frontier quality parity "
            "on customers' real documents, which this corpus cannot speak to."
        ),
        "comparison_same_corpus": {
            "qwen0.5b_kshot_k10_per_seed": {"1": 0.5708, "2": 0.4854, "3": 0.5021},
            "qwen0.5b_adapted_k10_per_seed": {"1": 0.9688, "2": 0.9542, "3": 0.9854},
            "source": "novel_schema_0.5b_k10_seed{1,2,3}_{arm}_2026-08-12.json "
                      "(full 60-doc arms; this arm is a 20-doc subset)",
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    for t in tenants:
        print(f"seed {t['seed']}: mean_micro_f1={t['mean_micro_f1']} "
              f"exact={t['exact_match']}/{t['n']} valid_json={t['valid_json']}/{t['n']}")
    print(f"pooled: {report['pooled_mean_micro_f1']} over {report['pooled_n']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
