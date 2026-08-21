"""Addendum B verdict: the ONLY authorized reader of novel-schema artifacts.

Encodes the gate exactly as frozen (ENTERPRISE_EVAL_SPEC.md Addendum B,
2026-08-12T19:40Z) so the verdict is computed, never argued:

  GO        mean paired delta at k=30 over seeds {1,2,3} >= +5.0 micro-F1,
            AND the receipt-level CI excludes zero, AND the sign test agrees
            (p < 0.05 with wins > losses) — the two-statistics rule.
  PIVOT     gate arms complete and the above does not hold.
  UNINFORMATIVE  a validity gate tripped (k-shot mean < 0.15 floor or
            > 0.95 ceiling on any gating seed): the task was not measurable
            at this rung, and the delta is NOT interpreted.
  UNDECIDABLE    gating pairs missing — never extrapolated.

k=10 rows are computed for comparability with Addendum A and are marked so
they cannot be promoted: a k=10 positive with a k=30 non-pass is a FAIL by
preregistration, and this script says so in the output rather than leaving
it to the reader's discipline.

    python3 scripts/novel_schema_summary.py --dir experiments --date 2026-08-12
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

GATE_DELTA = 0.05
GATE_K = 30
FLOOR = 0.15
CEILING = 0.95
SEEDS = (1, 2, 3)
IDENTITY = {"rung", "k", "seed", "arm"}

_Z95 = 1.960

# t95() used to be a lookup table starting at df=39, with a fallback that
# returned the invented constant 2.09 for anything smaller — roughly t at
# df≈19. Addendum E's cluster interval is six seeds (df=5), where the true
# quantile is 2.5706, so the published interval was ~19% too narrow, and
# too narrow in the direction that flatters us. An outside reader
# recomputed it from our own artifacts and found it (2026-08-21; erratum
# P13).
#
# A wider table would have fixed that instance and left the class: any df
# the table missed still silently borrowed a neighbour's quantile. So the
# table is gone. t95() now computes the exact Student-t quantile by
# bisecting its CDF, which is evaluated through the regularized incomplete
# beta function (continued fraction, Lentz). Stdlib only — the verify path
# must keep running on a machine with nothing installed.


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: int) -> float:
    """P(T <= t) for Student's t with df degrees of freedom."""
    x = df / (df + t * t)
    tail = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t95(df: int) -> float:
    """Two-sided 95% Student-t quantile: t(0.975, df). Exact, any df >= 1."""
    if df < 1:
        return float("nan")
    lo, hi = 0.0, 400.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if student_t_cdf(mid, df) < 0.975:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2.0, 4)


def mean(v: list[float]) -> float:
    return sum(v) / len(v)


def stdev(v: list[float]) -> float:
    if len(v) < 2:
        return 0.0
    mu = mean(v)
    return math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1))


def sign_test(deltas: list[float], tol: float = 0.01) -> dict:
    wins = sum(1 for d in deltas if d > tol)
    losses = sum(1 for d in deltas if d < -tol)
    n = wins + losses
    if n == 0:
        return {"wins": 0, "losses": 0, "ties": len(deltas), "p_value": None}
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1))
    p = min(1.0, 2 * tail / 2**n)
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(deltas) - n,
        # Never round a tiny p into a literal 0.0 (B.9.4: "not a probability
        # of zero") — keep the magnitude; JSON carries e.g. 2.19e-47 fine.
        "p_value": round(p, 4) if p >= 1e-4 else p,
    }


def load(directory: Path, date: str) -> dict:
    arms: dict = {}
    for path in sorted(directory.glob(f"novel_schema_*_{date}.json")):
        record = json.loads(path.read_text())
        if not IDENTITY.issubset(record):
            continue
        if "device" not in record:
            continue  # same environment-provenance rule as the CORD gate
        arms[(record["k"], record["seed"], record["arm"])] = record
    return arms


def analyse_k(arms: dict, k: int) -> dict:
    seed_deltas: list[float] = []
    pooled: list[float] = []
    attrition: list[dict] = []
    validity_trips: list[dict] = []
    missing: list[int] = []
    errored: list[int] = []
    heterogeneous: list[dict] = []
    for seed in SEEDS:
        adapted = arms.get((k, seed, "adapted"))
        kshot = arms.get((k, seed, "kshot"))
        if adapted is None or kshot is None:
            missing.append(seed)
            continue
        if "error" in adapted or "error" in kshot:
            errored.append(seed)
            continue
        if adapted.get("device") != kshot.get("device") or adapted.get(
            "dtype"
        ) != kshot.get("dtype"):
            # B.7-r3 made dtype a live degree of freedom (k=30 pairs moved
            # to fp16). A pair split across device or dtype is the same
            # contamination the CORD gate refuses across environments, so
            # it is refused here, not silently averaged. Legacy artifacts
            # without a dtype stamp compare as None == None and pass.
            heterogeneous.append(
                {
                    "seed": seed,
                    "adapted": [adapted.get("device"), adapted.get("dtype")],
                    "kshot": [kshot.get("device"), kshot.get("dtype")],
                }
            )
            continue
        base = kshot["mean_micro_f1"]
        if base < FLOOR:
            validity_trips.append({"seed": seed, "gate": "floor", "kshot_f1": base})
        elif base > CEILING:
            validity_trips.append({"seed": seed, "gate": "ceiling", "kshot_f1": base})
        a = {r["index"]: r["micro_f1"] for r in adapted["results"] if "micro_f1" in r}
        b = {r["index"]: r["micro_f1"] for r in kshot["results"] if "micro_f1" in r}
        shared = sorted(set(a) & set(b))
        # B.9.2: every summary must carry an explicit attrition field —
        # excluded documents (e.g. no_completion) are disclosed per seed.
        attrition.append(
            {
                "seed": seed,
                "eval_n": adapted.get("eval_n", len(adapted["results"])),
                "scored": len(shared),
                "excluded_no_completion": adapted.get(
                    "eval_n", len(adapted["results"])
                )
                - len(shared),
                "symmetric": ({r["index"] for r in adapted["results"]} - set(a))
                == ({r["index"] for r in kshot["results"]} - set(b)),
            }
        )
        seed_deltas.append(adapted["mean_micro_f1"] - kshot["mean_micro_f1"])
        pooled.extend(a[i] - b[i] for i in shared)

    out: dict = {
        "k": k,
        "role": "DECISION" if k == GATE_K else "comparability-only",
        "pairs_complete": len(seed_deltas),
        "missing_seeds": missing,
        "errored_seeds": errored,
        "heterogeneous_pairs_refused": heterogeneous,
        "validity_trips": validity_trips,
        "attrition": attrition,
    }
    if seed_deltas:
        out["seed_deltas"] = [round(d, 4) for d in seed_deltas]
        out["mean_delta"] = round(mean(seed_deltas), 4)
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
            "mde_80pct": round((_Z95 + 0.842) * sd / math.sqrt(len(pooled)), 4),
        }
    return out


def verdict(gate_row: dict) -> tuple[str, str]:
    if gate_row["validity_trips"]:
        trips = ", ".join(
            f"seed {t['seed']} {t['gate']} (kshot {t['kshot_f1']})"
            for t in gate_row["validity_trips"]
        )
        return "UNINFORMATIVE", (
            f"Validity gate tripped: {trips}. The task was not measurable at "
            "this rung; the delta is not interpreted (Addendum B B.5). "
            "Re-run at the next rung; this is NOT a fail."
        )
    if gate_row["pairs_complete"] < len(SEEDS):
        return "UNDECIDABLE", (
            f"Only {gate_row['pairs_complete']}/{len(SEEDS)} gating pairs "
            "complete. No verdict is extrapolated from partial pairs."
        )
    passes_mean = gate_row["mean_delta"] >= GATE_DELTA
    rl = gate_row.get("receipt_level", {})
    st = rl.get("sign_test", {})
    sign_agrees = (
        st.get("p_value") is not None
        and st["p_value"] < 0.05
        and st["wins"] > st["losses"]
    )
    ci_agrees = bool(rl.get("excludes_zero")) and rl.get("mean", 0) > 0
    if passes_mean and ci_agrees and sign_agrees:
        # B.9.4: a rounded-to-zero p must be restated as a bound, never as
        # "p=0.0" (float underflow, not a probability of zero).
        p_str = (
            "p<1e-15"
            if st["p_value"] < 1e-15
            else f"p={st['p_value']:.4g}"
        )
        return "GO", (
            f"Gate PASSES at k={GATE_K}: mean {gate_row['mean_delta']:+.4f} "
            f">= +{GATE_DELTA}, CI excludes zero, sign test agrees "
            f"({st['wins']}W/{st['losses']}L, {p_str}). "
            "Claimable per Addendum B B.6, always next to the CORD negative."
        )
    reasons = []
    if not passes_mean:
        reasons.append(f"mean {gate_row.get('mean_delta')} < +{GATE_DELTA}")
    if not ci_agrees:
        reasons.append("CI does not exclude zero on the positive side")
    if not sign_agrees:
        reasons.append(f"sign test does not agree ({st})")
    return "PIVOT", (
        "Gate FAILS: " + "; ".join(reasons) + ". Combined with Addendum A "
        "(0.5B -7.3, 4B -4.5 on CORD), adaptation as implemented has now "
        "failed both where the model knows the domain and where it cannot. "
        "Addendum B B.6's fail branch applies, in those words."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    directory = Path(args.dir)

    arms = load(directory, args.date)
    gate_row = analyse_k(arms, GATE_K)
    comparability = analyse_k(arms, 10)
    word, detail = verdict(gate_row)

    report = {
        "spec": "ENTERPRISE_EVAL_SPEC.md Addendum B (frozen 2026-08-12T19:40Z)",
        "date": args.date,
        # Recorded at verdict-computation time so downstream doc fillers
        # quote an artifact value, never an invented default.
        "decided": datetime.now(timezone.utc).date().isoformat(),
        "VERDICT": word,
        "verdict_detail": detail,
        "gate_k30": gate_row,
        "comparability_k10": comparability,
        "k10_promotion_bar": (
            "Preregistered: a k=10 positive with a k=30 non-pass is a FAIL "
            "and may not be quoted as the result."
        ),
    }
    out = directory / f"novel_schema_summary_{args.date}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"VERDICT: {word}")
    print(detail)
    print(f"\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
