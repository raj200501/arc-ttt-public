"""B.8 replication readout: the k=10 comparability point across ALL tenants.

Exploratory by design and labeled as such in every output line: this is
the robustness picture of the comparability point, NEVER the gate. Pools
per-record paired deltas across every tenant seed present (originals 1-3
plus the B.8 sweep 4-10), refusing pairs that are environment- or
dtype-heterogeneous, and reports per-tenant deltas alongside the pooled
interval + sign test so a single flipped tenant is visible, not averaged
away.

    python3 scripts/novel_schema_rep_summary.py --dir experiments --date 2026-08-12
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

IDENTITY = {"rung", "k", "seed", "arm"}


def mean(v):
    return sum(v) / len(v)


def stdev(v):
    if len(v) < 2:
        return 0.0
    mu = mean(v)
    return math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1))


def sign_test(deltas, tol=0.01):
    wins = sum(1 for d in deltas if d > tol)
    losses = sum(1 for d in deltas if d < -tol)
    n = wins + losses
    if n == 0:
        return {"wins": 0, "losses": 0, "ties": len(deltas), "p_value": None}
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1))
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(deltas) - n,
        "p_value": round(min(1.0, 2 * tail / 2**n), 6),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    directory = Path(args.dir)

    arms = {}
    for path in sorted(directory.glob(f"novel_schema_*_{args.date}.json")):
        record = json.loads(path.read_text())
        if not IDENTITY.issubset(record) or "device" not in record:
            continue
        if record["k"] != 10 or "error" in record:
            continue
        arms[(record["seed"], record["arm"])] = record

    per_tenant = []
    pooled = []
    refused = []
    seeds = sorted({seed for seed, _ in arms})
    for seed in seeds:
        adapted = arms.get((seed, "adapted"))
        kshot = arms.get((seed, "kshot"))
        if adapted is None or kshot is None:
            continue
        if adapted.get("device") != kshot.get("device") or adapted.get(
            "dtype"
        ) != kshot.get("dtype"):
            refused.append(seed)
            continue
        a = {r["index"]: r["micro_f1"] for r in adapted["results"] if "micro_f1" in r}
        b = {r["index"]: r["micro_f1"] for r in kshot["results"] if "micro_f1" in r}
        deltas = [a[i] - b[i] for i in sorted(set(a) & set(b))]
        pooled.extend(deltas)
        per_tenant.append(
            {
                "tenant_seed": seed,
                "n": len(deltas),
                "adapted_f1": adapted["mean_micro_f1"],
                "kshot_f1": kshot["mean_micro_f1"],
                "delta": round(mean(deltas), 4),
                "sign_test": sign_test(deltas),
            }
        )

    report = {
        "spec": "ENTERPRISE_EVAL_SPEC.md B.8 (frozen 2026-08-15T21:25Z)",
        "ROLE": "EXPLORATORY REPLICATION of the k=10 comparability point. "
        "NOT the gate. May never be quoted as the gate.",
        "tenants": per_tenant,
        "tenants_negative": [t["tenant_seed"] for t in per_tenant if t["delta"] < 0],
        "refused_heterogeneous": refused,
    }
    if len(pooled) > 1:
        sd = stdev(pooled)
        mu = mean(pooled)
        half = 1.96 * sd / math.sqrt(len(pooled))
        report["pooled"] = {
            "n": len(pooled),
            "n_tenants": len(per_tenant),
            "mean": round(mu, 4),
            "ci95": [round(mu - half, 4), round(mu + half, 4)],
            "sign_test": sign_test(pooled),
        }
    out = directory / f"novel_schema_rep_summary_{args.date}.json"
    out.write_text(json.dumps(report, indent=2))
    for t in per_tenant:
        st = t["sign_test"]
        print(
            f"tenant {t['tenant_seed']:>2}: adapted {t['adapted_f1']:.4f} "
            f"vs kshot {t['kshot_f1']:.4f} -> {t['delta']:+.4f} "
            f"({st['wins']}W/{st['losses']}L/{st['ties']}T)"
        )
    if "pooled" in report:
        p = report["pooled"]
        st = p["sign_test"]
        lo, hi = p["ci95"]
        print(
            f"POOLED ({p['n_tenants']} tenants, n={p['n']}): {p['mean']:+.4f} "
            f"CI95=[{lo:+.4f},{hi:+.4f}] "
            f"sign {st['wins']}W/{st['losses']}L p={st['p_value']}"
        )
    print("ROLE: exploratory replication of the comparability point — not the gate")
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
