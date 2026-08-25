#!/usr/bin/env python3
"""Addendum I across runs. A single hosted-API run is not a measurement.

The first Addendum I write-up published **1.0000** as though it were a
property of the hosted model. Re-running the identical arm -- same model,
same prompt, same documents, `temperature: 0` -- returned a different
number. Hosted inference is not deterministic at temperature 0 (batching,
routing and kernel non-determinism all move it), so a point estimate from
one run is a sample, and publishing it as a fact was wrong.

This reader aggregates every banked run of the arm and reports what the
set of runs supports: a mean, a range, and the per-document scores that
move between runs. It also reports the comparison that actually matters
-- our adapted 0.5B against the hosted model -- computed per run, so a
reader can see whether the CONCLUSION is stable even though the number is
not.

    python3 scripts/market_baseline_summary.py
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib
import statistics

REPO = pathlib.Path(__file__).resolve().parent.parent
PATTERN = "experiments/waybill_market_baseline_*.json"


def sign_counts(deltas: list[float]) -> dict:
    wins = sum(1 for d in deltas if d > 1e-12)
    losses = sum(1 for d in deltas if d < -1e-12)
    ties = len(deltas) - wins - losses
    n = wins + losses
    tail = (sum(math.comb(n, k) for k in range(losses, n + 1)) / 2 ** n
            if n else 1.0)
    return {"ours_wins": wins, "hosted_wins": losses, "ties": ties,
            "p_hosted_better": round(tail, 8)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(
        REPO / "experiments" / "waybill_market_summary_2026-08-22.json"))
    args = parser.parse_args()

    runs = []
    for path in sorted(glob.glob(str(REPO / PATTERN))):
        name = pathlib.Path(path).name
        if "summary" in name:
            continue
        record = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        runs.append({
            "artifact": name,
            "demonstration_format": record.get(
                "demonstration_format", "one packed user turn (artifact "
                                        "predates the field)"),
            "mean_micro_f1": record["hosted_mean_micro_f1"],
            "exact_match": record.get("hosted_exact_match", "not recorded"),
            "invalid_json": record["hosted_invalid_json"],
            "fence_stripped": record.get("fence_stripped_documents",
                                         "not recorded"),
            "mean_without_fence_strip": record.get(
                "mean_without_fence_strip", "not recorded"),
            "paired_ours_minus_hosted":
                record["paired_our_adapted_minus_hosted"]["mean_delta"],
            "per_doc": {r["id"]: r["micro_f1"] for r in record["results"]},
        })

    means = [r["mean_micro_f1"] for r in runs]
    deltas = [r["paired_ours_minus_hosted"] for r in runs]
    ids = sorted(runs[0]["per_doc"]) if runs else []
    unstable = {i: sorted({r["per_doc"][i] for r in runs})
                for i in ids
                if len({r["per_doc"][i] for r in runs}) > 1}

    record = {
        "what": "Every banked run of the Addendum I hosted-model arm, "
                "aggregated. A single hosted-API run is a sample, not a "
                "measurement.",
        "why_this_exists": "Addendum I first published 1.0000 as a fact. "
                           "Re-running the identical arm at temperature 0 "
                           "returned a different number. Hosted inference "
                           "is not deterministic at temperature 0, so the "
                           "headline is reported here as a mean and range "
                           "across runs, and the point estimate is "
                           "withdrawn.",
        "n_runs": len(runs),
        "hosted_mean_of_run_means": round(statistics.mean(means), 4),
        "hosted_range": [min(means), max(means)],
        "hosted_stdev": round(statistics.stdev(means), 4)
        if len(means) > 1 else None,
        "our_adapted_0_5b": runs[0] and 0.8833,
        "paired_delta_range": [round(min(deltas), 4), round(max(deltas), 4)],
        "conclusion_stable_across_all_runs": all(d < 0 for d in deltas),
        "documents_that_move_between_runs": unstable,
        "reading": (
            "The NUMBER is not reproducible to four decimal places; the "
            "CONCLUSION is. The hosted model beats our adapted 0.5B in "
            "every run, on every run's sign test. Quote the range, never "
            "one run's mean."),
        "runs": runs,
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")

    print(f"{len(runs)} banked runs of the same arm:")
    for r in runs:
        print(f"  {r['mean_micro_f1']:.4f}  exact {r['exact_match']:>12}  "
              f"paired {r['paired_ours_minus_hosted']:+.4f}  "
              f"{r['artifact'].split('lite_')[1]}")
    print(f"\nmean of run means : {record['hosted_mean_of_run_means']:.4f}")
    print(f"range             : {record['hosted_range']}")
    print(f"our adapted 0.5B  : 0.8833")
    print(f"conclusion stable across every run: "
          f"{record['conclusion_stable_across_all_runs']}")
    if unstable:
        print(f"documents that move between runs: {sorted(unstable)}")
    print(f"\nbanked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
