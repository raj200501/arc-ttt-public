"""Addendum E (E-r2) verdict: the ONLY authorized reader of the E arms.

Encodes the gate exactly as frozen (ENTERPRISE_EVAL_SPEC.md Addendum E
E.2, amended E-r1 2026-08-19T06:45Z and E-r2 06:50Z — the amendments
changed the geometry and the seed set, and explicitly left "the +5 bar
and both-statistics rule" unchanged):

  PASS      seed-mean paired delta (adapted - kshot) over the frozen
            seeds {203,204,206,207,208,209} >= +5.0 micro-F1, AND the
            receipt-level CI excludes zero on the positive side, AND
            the sign test agrees (p < 0.05, wins > losses).
  FAIL      all six pairs complete and the above does not hold — the
            E.2 branch: "reported as the boundary of the effect
            (novelty may be carried by the fixed shape)".
  UNINFORMATIVE  a validity gate tripped (k-shot mean outside the
            preregistered [0.15, 0.95] window on any gating seed), per
            the B.5/B.6 branch E.4 already exercised once.
  UNDECIDABLE    gating pairs missing — never extrapolated.

The statistics are IMPORTED from novel_schema_summary rather than
reimplemented, so the E verdict cannot silently diverge from the B
verdict's conventions (sign-test tolerance, t95 table, p-value
underflow handling). Cluster-level CI over the six seed deltas is
reported beside the receipt level, as E.2 requires.

    python3 scripts/addendum_e_summary.py --dir experiments --date 2026-08-19
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from novel_schema_summary import (  # noqa: E402  (path set above)
    CEILING,
    FLOOR,
    mean,
    sign_test,
    stdev,
    t95,
)

GATE_DELTA = 0.05
GATE_K = 30
SEEDS = (203, 204, 206, 207, 208, 209)  # E-r2 frozen, tokenizer-screened
PREFIX = "novel_schema_e_"


def load(directory: Path, date: str) -> dict:
    arms: dict = {}
    for path in sorted(directory.glob(f"{PREFIX}*_{date}.json")):
        record = json.loads(path.read_text())
        if not {"k", "seed", "arm"}.issubset(record):
            continue
        if "device" not in record:
            continue  # same environment-provenance rule as every other gate
        arms[(record["k"], record["seed"], record["arm"])] = record
    return arms


def analyse(arms: dict) -> dict:
    seed_rows: list[dict] = []
    seed_deltas: list[float] = []
    pooled: list[float] = []
    attrition: list[dict] = []
    validity_trips: list[dict] = []
    missing: list[int] = []
    heterogeneous: list[dict] = []

    for seed in SEEDS:
        adapted = arms.get((GATE_K, seed, "adapted"))
        kshot = arms.get((GATE_K, seed, "kshot"))
        if adapted is None or kshot is None:
            missing.append(seed)
            continue
        if "error" in adapted or "error" in kshot:
            missing.append(seed)
            continue
        if (adapted.get("device") != kshot.get("device")
                or adapted.get("dtype") != kshot.get("dtype")):
            heterogeneous.append({
                "seed": seed,
                "adapted": [adapted.get("device"), adapted.get("dtype")],
                "kshot": [kshot.get("device"), kshot.get("dtype")],
            })
            continue
        base = kshot["mean_micro_f1"]
        if base is None:
            validity_trips.append({"seed": seed, "gate": "no-mean",
                                   "kshot_f1": None})
            continue
        if base < FLOOR:
            validity_trips.append({"seed": seed, "gate": "floor",
                                   "kshot_f1": base})
        elif base > CEILING:
            validity_trips.append({"seed": seed, "gate": "ceiling",
                                   "kshot_f1": base})

        a = {r["index"]: r["micro_f1"] for r in adapted["results"]
             if "micro_f1" in r}
        b = {r["index"]: r["micro_f1"] for r in kshot["results"]
             if "micro_f1" in r}
        shared = sorted(set(a) & set(b))
        attrition.append({
            "seed": seed,
            "eval_n": adapted.get("eval_n", len(adapted["results"])),
            "scored": len(shared),
            "excluded_no_completion":
                adapted.get("eval_n", len(adapted["results"])) - len(shared),
            "symmetric": ({r["index"] for r in adapted["results"]} - set(a))
                         == ({r["index"] for r in kshot["results"]} - set(b)),
        })
        delta = adapted["mean_micro_f1"] - kshot["mean_micro_f1"]
        seed_deltas.append(delta)
        seed_rows.append({
            "seed": seed,
            "adapted": round(adapted["mean_micro_f1"], 4),
            "kshot": round(kshot["mean_micro_f1"], 4),
            "delta": round(delta, 4),
            "adapt_seconds": adapted.get("adapt_seconds"),
            "resumed": bool(adapted.get("resumed") or kshot.get("resumed")),
        })
        pooled.extend(a[i] - b[i] for i in shared)

    out: dict = {
        "addendum": "E (E-r2)",
        "geometry": "diverse-compact",
        "frozen_seeds": list(SEEDS),
        "bar": GATE_DELTA,
        "k": GATE_K,
        "pairs_complete": len(seed_deltas),
        "missing_seeds": missing,
        "heterogeneous_pairs_refused": heterogeneous,
        "validity_trips": validity_trips,
        "attrition": attrition,
        "per_seed": seed_rows,
    }
    if seed_deltas:
        out["seed_deltas"] = [round(d, 4) for d in seed_deltas]
        out["mean_delta"] = round(mean(seed_deltas), 4)
        if len(seed_deltas) > 1:
            sd = stdev(seed_deltas)
            se = sd / math.sqrt(len(seed_deltas))
            half = t95(len(seed_deltas) - 1) * se
            mu = mean(seed_deltas)
            out["cluster_level"] = {
                "n_seeds": len(seed_deltas),
                "mean": round(mu, 4),
                "ci95": [round(mu - half, 4), round(mu + half, 4)],
                "excludes_zero": (mu - half > 0) or (mu + half < 0),
            }
    if len(pooled) > 1:
        sd = stdev(pooled)
        se = sd / math.sqrt(len(pooled))
        half = t95(len(pooled) - 1) * se
        mu = mean(pooled)
        out["receipt_level"] = {
            "n": len(pooled),
            "mean": round(mu, 4),
            "ci95": [round(mu - half, 4), round(mu + half, 4)],
            "excludes_zero": (mu - half > 0) or (mu + half < 0),
            "sign_test": sign_test(pooled),
        }
    return out


def verdict(row: dict) -> tuple[str, str]:
    if row["validity_trips"]:
        trips = ", ".join(
            f"seed {t['seed']} {t['gate']} (kshot {t['kshot_f1']})"
            for t in row["validity_trips"]
        )
        return "UNINFORMATIVE", (
            f"Validity gate tripped: {trips}. The task was not measurable "
            "at this rung; the delta is not interpreted (B.5, the branch "
            "E.4 already exercised). This is NOT a fail."
        )
    if row["pairs_complete"] < len(SEEDS):
        return "UNDECIDABLE", (
            f"Only {row['pairs_complete']}/{len(SEEDS)} frozen pairs "
            "complete. No verdict is extrapolated from partial pairs."
        )
    rl = row.get("receipt_level", {})
    st = rl.get("sign_test", {})
    passes_mean = row["mean_delta"] >= GATE_DELTA
    ci_agrees = bool(rl.get("excludes_zero")) and rl.get("mean", 0) > 0
    sign_agrees = (st.get("p_value") is not None
                   and st["p_value"] < 0.05
                   and st["wins"] > st["losses"])
    if passes_mean and ci_agrees and sign_agrees:
        p_str = ("p<1e-15" if st["p_value"] < 1e-15
                 else f"p={st['p_value']:.4g}")
        return "PASS", (
            f"Addendum E PASSES at k={GATE_K} on shape-varying geometry: "
            f"seed-mean {row['mean_delta']:+.4f} >= +{GATE_DELTA} over six "
            f"fresh tenants, receipt CI excludes zero, sign test agrees "
            f"({st['wins']}W/{st['losses']}L, {p_str}). Per E.2 this "
            "NARROWS the shared-geometry objection with n doubled: the "
            "effect is not carried by one fixed corpus shape at 6-7 "
            "fields. It does not retire it — these shapes are SMALLER "
            "than the 8-field gate corpus, shapes at or above it are not "
            "measurable at k=30 under the frozen token budget, and every "
            "tenant here still comes from the same generator family."
        )
    reasons = []
    if not passes_mean:
        reasons.append(f"seed-mean {row.get('mean_delta')} < +{GATE_DELTA}")
    if not ci_agrees:
        reasons.append("receipt CI does not exclude zero on the positive side")
    if not sign_agrees:
        reasons.append(f"sign test does not agree ({st})")
    return "FAIL", (
        "Addendum E FAILS: " + "; ".join(reasons) + ". Per the pre-written "
        "E.2 branch this is published as the BOUNDARY of the effect — "
        "novelty may be carried by the fixed shape — and the shape-varying "
        "claim is retired until a larger rung or a different recipe passes."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="experiments")
    parser.add_argument("--date", default="2026-08-19")
    parser.add_argument("--out", default=None,
                        help="write the summary artifact here")
    args = parser.parse_args()

    arms = load(Path(args.dir), args.date)
    row = analyse(arms)
    call, why = verdict(row)
    row["verdict"] = call
    row["rationale"] = why
    row["computed_at"] = datetime.now(timezone.utc).isoformat()

    print(f"Addendum E (E-r2) — geometry diverse-compact, seeds "
          f"{list(SEEDS)}, bar +{GATE_DELTA}")
    for r in row["per_seed"]:
        print(f"  seed {r['seed']}: adapted {r['adapted']:.4f} - kshot "
              f"{r['kshot']:.4f} = {r['delta']:+.4f}"
              + ("  (resumed)" if r["resumed"] else ""))
    if "mean_delta" in row:
        print(f"  seed-mean delta: {row['mean_delta']:+.4f}")
    if "cluster_level" in row:
        cl = row["cluster_level"]
        print(f"  cluster CI95 (n={cl['n_seeds']} seeds): {cl['ci95']}")
    if "receipt_level" in row:
        r = row["receipt_level"]
        st = r["sign_test"]
        print(f"  receipt CI95 (n={r['n']}): {r['ci95']}   sign test "
              f"{st['wins']}W/{st['losses']}L/{st['ties']}T")
    total_excluded = sum(a["excluded_no_completion"] for a in row["attrition"])
    print(f"  attrition: {total_excluded} excluded across "
          f"{len(row['attrition'])} pairs "
          f"(symmetric: {all(a['symmetric'] for a in row['attrition'])})")
    print(f"\nVERDICT: {call}\n{why}")

    if args.out:
        Path(args.out).write_text(json.dumps(row, indent=2))
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
