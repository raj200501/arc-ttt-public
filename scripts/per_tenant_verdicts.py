#!/usr/bin/env python3
"""Apply the two-statistics rule PER TENANT, which is the unit we sell.

Every gate in `VERDICT.md` reports a seed-mean and a POOLED sign test. The
blind-rehearsal row reports a single tenant and applies the rule to it
directly -- and returns FAIL, because its sign test disagreed. Those two
treatments are not the same rule, and an outside reader pointed out that
the difference has favoured us in both places: pooling hides a tenant whose
sign test disagrees, and not pooling exposed the one realistic corpus.

This script removes the asymmetry by publishing the per-tenant reading for
every paired gate arm in the repository, under the same frozen rule
(seed-mean delta >= +5.0 micro-F1 AND the sign test agreeing at p < 0.05).

It does not restate any gate verdict. The gates were preregistered on the
pooled statistic and they stay as decided; this is the additional reading
the reader is entitled to, and it is published because one tenant fails it.

    python3 scripts/per_tenant_verdicts.py
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"
BAR = 0.05

GATES = [
    ("gate1_addendum_B_k30", "adaptation vs the same model's 30-shot prompt",
     [(f"novel_schema_0.5b_k30_seed{s}_kshot_2026-08-12.json",
       f"novel_schema_0.5b_k30_seed{s}_adapted_2026-08-12.json", str(s))
      for s in (1, 2, 3)]),
    ("gate4_addendum_F_document_mode",
     "document-mode adapted vs the same 30-shot prompt",
     [(f"novel_schema_0.5b_k30_seed{s}_kshot_2026-08-12.json",
       f"novel_schema_f_0.5b_k30_seed{s}_docadapted_2026-08-19.json", str(s))
      for s in (1, 2, 3)]),
    ("gate5_addendum_E_r2", "six fresh shape-varying tenants",
     [(f"novel_schema_e_0.5b_k30_seed{s}_kshot_2026-08-19.json",
       f"novel_schema_e_0.5b_k30_seed{s}_adapted_2026-08-19.json", str(s))
      for s in (203, 204, 206, 207, 208, 209)]),
]


def scores(name: str) -> dict:
    record = json.loads((EXP / name).read_text(encoding="utf-8"))
    return {r["index"]: r["micro_f1"] for r in record["results"]
            if "micro_f1" in r}


def read_one(base: str, adapt: str, label: str) -> dict:
    a, b = scores(base), scores(adapt)
    ids = sorted(set(a) & set(b))
    deltas = [b[i] - a[i] for i in ids]
    wins = sum(1 for d in deltas if d > 1e-12)
    losses = sum(1 for d in deltas if d < -1e-12)
    ties = len(deltas) - wins - losses
    n = wins + losses
    p = (sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n
         if n else 1.0)
    mean = sum(deltas) / len(deltas)
    clears = mean >= BAR
    agrees = p < 0.05 and wins > losses
    return {
        "tenant": label,
        "n": len(ids),
        "mean_delta": round(mean, 4),
        "sign_test": {"wins": wins, "losses": losses, "ties": ties,
                      "p_value": p},
        "clears_bar": clears,
        "sign_test_agrees": agrees,
        "per_tenant_verdict": "PASS" if (clears and agrees) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(
        EXP / "per_tenant_verdicts_2026-08-22.json"))
    args = parser.parse_args()

    blocks = []
    for name, what, arms in GATES:
        reads = [read_one(b, a, lab) for b, a, lab in arms]
        failing = [r["tenant"] for r in reads
                   if r["per_tenant_verdict"] == "FAIL"]
        blocks.append({
            "gate": name, "comparison": what,
            "tenants_passing_individually":
                f"{len(reads) - len(failing)}/{len(reads)}",
            "tenants_failing_individually": failing,
            "tenants": reads,
        })

    all_reads = [r for b in blocks for r in b["tenants"]]
    failing = [f"{b['gate']}:{r['tenant']}" for b in blocks
               for r in b["tenants"] if r["per_tenant_verdict"] == "FAIL"]

    record = {
        "what": "The frozen two-statistics rule applied PER TENANT rather "
                "than pooled, for every paired gate arm in the repository.",
        "status": "POST-HOC re-reading of banked artifacts under the "
                  "already-frozen rule. It does NOT restate any gate "
                  "verdict -- the gates were preregistered on the pooled "
                  "statistic and stand as decided. Published because the "
                  "pooled and per-tenant treatments were being applied "
                  "asymmetrically, and the asymmetry favoured us.",
        "rule": {"bar_delta": BAR,
                 "requires": "mean delta >= bar AND sign test p < 0.05 "
                             "with wins > losses"},
        "tenants_passing_individually":
            f"{len(all_reads) - len(failing)}/{len(all_reads)}",
        "tenants_failing_individually": failing,
        "gates": blocks,
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")

    for b in blocks:
        print(f"\n{b['gate']}  ({b['tenants_passing_individually']} tenants "
              f"pass individually)")
        for r in b["tenants"]:
            st = r["sign_test"]
            print(f"  tenant {r['tenant']:<4} n={r['n']:>3}  "
                  f"mean {r['mean_delta']:+.4f}  "
                  f"{st['wins']}W/{st['losses']}L/{st['ties']}T  "
                  f"p={st['p_value']:.3g}  -> {r['per_tenant_verdict']}")
    print(f"\nOverall: {record['tenants_passing_individually']} tenants pass "
          f"the frozen rule on their own.")
    if failing:
        print("Failing individually: " + ", ".join(failing))
    print(f"banked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
