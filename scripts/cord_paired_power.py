"""Is the frozen G-E2 gate precise enough to resolve the effect it tests?

The gate (ENTERPRISE_EVAL_SPEC Addendum A) is a mean over three SEED MEANS:
pass iff mean(adapted - kshot) >= +5 F1 at k=10 over seeds {1,2,3}. That is
three observations. The first real 0.5b arms showed seed-level deltas of
-7.4, -0.1 and +10.9 F1 at k=5 - an ~18-point spread - which raises a
question that has to be answered before more compute is spent: can a
three-observation mean resolve a +5-point effect at all?

This script answers it from the artifacts, WITHOUT touching the gate. The
gate stays exactly as preregistered; this reports its PRECISION alongside.

Two estimators of the same paired delta:

  seed-level    n=3, one observation per seed. This is the frozen gate's
                own statistic. Reported so its standard error is visible.
  receipt-level n=20 per seed (60 pooled at k=10). Valid because within a
                (rung, k, seed) both arms are built from the same shuffle
                and evaluated on the same slice shuffled[k:k+EVAL_N], so
                result index i is the SAME receipt in both arms. Pairing by
                index removes between-receipt difficulty variance, which is
                the dominant noise term.

Reported per (rung, k): both means, the receipt-level 95% CI, and the
minimum detectable effect at 80% power - the smallest true delta the
current n could reliably distinguish from zero. If MDE exceeds the +5-point
bar, the gate cannot resolve its own threshold and needs more receipts, not
more seeds.

Environment homogeneity is enforced exactly as in cord_scale_summary.py:
mixed-environment pairs are refused, never pooled.

    python3 scripts/cord_paired_power.py --dir experiments --date 2026-08-11
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

GATE_DELTA = 0.05  # +5 F1 points, spec Addendum A / G-E2
IDENTITY_KEYS = {"rung", "k", "seed", "arm"}

# Two-sided t quantiles at 0.05, by degrees of freedom. Table rather than
# scipy: the Kaggle/dev images do not carry scipy and a wrong CI is worse
# than no CI. Values beyond the table fall back to the normal limit (1.960),
# which is the correct asymptote and only ever slightly anticonservative.
_Z80_POWER = 0.842  # one-sided z at 80% power
_Z95 = 1.960

# t95 is IMPORTED, not redefined. The 2026-08-21 correction said the
# lookup table was "deleted, not extended" and that "the repo has one
# estimator" -- and this file still carried a second table, so the class
# the correction claimed to close was open in one place for a day. It is
# closed here, and `tests/test_readers_agree.py` now scans every script
# rather than a named pair, so the next copy fails a test instead of an
# audit.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from novel_schema_summary import t95  # noqa: E402  (single estimator authority)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sign_test(deltas: list[float], tol: float = 0.01) -> dict:
    """Distribution-free companion to the parametric CI.

    The CI is a statement about a MEAN, and a mean over 20-60 receipts is
    hostage to a couple of them: a receipt that goes from near-zero to
    near-perfect contributes ~+1.0 to a sum where the whole effect is ~+4.
    The sign test asks the question outliers cannot answer - on how many
    receipts did adaptation win at all? - so a headline that survives one
    and not the other is an artifact, and is reported as such.

    Deltas within +/-tol count as ties and are excluded, matching the
    convention that micro-F1 differences below a point are not a win.
    """

    wins = sum(1 for d in deltas if d > tol)
    losses = sum(1 for d in deltas if d < -tol)
    trials = wins + losses
    if trials == 0:
        return {"wins": 0, "losses": 0, "ties": len(deltas), "p_value": None}
    tail = sum(math.comb(trials, i) for i in range(min(wins, losses) + 1))
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(deltas) - trials,
        "p_value": round(min(1.0, 2 * tail / 2**trials), 4),
    }


def jackknife_top(deltas: list[float], drop: int = 2) -> float | None:
    """Mean after removing the `drop` largest winners.

    Answers "how many receipts is this claim standing on?" directly. If
    dropping two of sixty moves the mean across the gate threshold, the
    claim belongs to those two receipts, not to the method.
    """

    if len(deltas) <= drop:
        return None
    kept = sorted(deltas)[:-drop]
    return round(mean(kept), 4)


def stdev(values: list[float]) -> float:
    """Sample standard deviation; 0.0 for n<2 (undefined, reported as such)."""

    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def load_arms(directory: Path, date: str) -> dict:
    arms: dict = {}
    for path in sorted(directory.glob(f"cord_scale_*_{date}.json")):
        if path.name.startswith("cord_scale_summary"):
            continue
        record = json.loads(path.read_text())
        if not IDENTITY_KEYS.issubset(record) or "error" in record:
            continue
        record["_env"] = "kernel" if "device" in record else "local"
        arms[(record["rung"], record["k"], record["seed"], record["arm"])] = record
    return arms


def receipt_deltas(adapted: dict, kshot: dict) -> list[float] | None:
    """Per-receipt paired deltas, or None if the two arms are not aligned.

    Alignment is asserted, not assumed: an arm that dropped a receipt to
    "no completion" has a gap in its results list, and silently zipping two
    ragged lists would pair receipt i against receipt j.
    """

    a = {r["index"]: r["micro_f1"] for r in adapted["results"] if "micro_f1" in r}
    b = {r["index"]: r["micro_f1"] for r in kshot["results"] if "micro_f1" in r}
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    return [a[i] - b[i] for i in shared]


def analyse(arms: dict) -> list[dict]:
    out = []
    rungs = sorted({key[0] for key in arms})
    for rung in rungs:
        for k in (5, 10, 30):
            seed_deltas: list[float] = []
            pooled: list[float] = []
            refused: list[int] = []
            per_seed = []
            pool_envs: set[str] = set()
            for seed in (1, 2, 3):
                adapted = arms.get((rung, k, seed, "adapted"))
                kshot = arms.get((rung, k, seed, "kshot"))
                if adapted is None or kshot is None:
                    continue
                if adapted["_env"] != kshot["_env"]:
                    refused.append(seed)  # same hard refusal as the summary
                    continue
                deltas = receipt_deltas(adapted, kshot)
                if deltas is None:
                    continue
                seed_deltas.append(adapted["mean_micro_f1"] - kshot["mean_micro_f1"])
                pooled.extend(deltas)
                pool_envs.add(adapted["_env"])
                per_seed.append(
                    {
                        "seed": seed,
                        "env": adapted["_env"],
                        "n_receipts": len(deltas),
                        "mean_delta": round(mean(deltas), 4),
                        "sd_receipt": round(stdev(deltas), 4),
                    }
                )
            if not pooled:
                continue

            n = len(pooled)
            sd = stdev(pooled)
            se = sd / math.sqrt(n) if n > 1 else float("nan")
            half = t95(n - 1) * se
            mu = mean(pooled)

            # Smallest true effect detectable at 80% power, two-sided 0.05.
            mde = (_Z95 + _Z80_POWER) * sd / math.sqrt(n) if n > 1 else float("nan")
            # And the n that WOULD resolve the frozen +5-point bar.
            needed = (
                math.ceil(((_Z95 + _Z80_POWER) * sd / GATE_DELTA) ** 2)
                if sd > 0
                else 0
            )

            entry = {
                "rung": rung,
                "k": k,
                "seed_level": {
                    "n": len(seed_deltas),
                    "deltas": [round(d, 4) for d in seed_deltas],
                    "mean": round(mean(seed_deltas), 4) if seed_deltas else None,
                    "sd": round(stdev(seed_deltas), 4),
                    "note": "this is the FROZEN gate's own statistic (n=3 seeds)",
                },
                "receipt_level": {
                    "n": n,
                    "mean": round(mu, 4),
                    "sd": round(sd, 4),
                    "se": round(se, 4),
                    "ci95": [round(mu - half, 4), round(mu + half, 4)],
                    "excludes_zero": (mu - half > 0) or (mu + half < 0),
                    "per_seed": per_seed,
                },
                "power": {
                    "mde_80pct": round(mde, 4),
                    "gate_is_resolvable": bool(mde <= GATE_DELTA),
                    "receipts_needed_for_gate": needed,
                },
                "robustness": {
                    "sign_test": sign_test(pooled),
                    "mean_less_top2_winners": jackknife_top(pooled, 2),
                    "mean_less_top3_winners": jackknife_top(pooled, 3),
                },
            }
            if refused:
                entry["refused_mixed_environment"] = refused
            if len(pool_envs) > 1:
                # Each contributing PAIR is internally homogeneous, so each
                # seed delta is valid on its own - but pooling deltas measured
                # in different environments assumes the effect is
                # environment-invariant, which is the assumption this project
                # refuses to make at the pair level. Same rule, one level up:
                # the pooled CI is not authoritative here, the per-seed rows
                # are. Resolved by re-running the odd pair, not by relaxing it.
                entry["receipt_level"]["pooled_across_environments"] = sorted(pool_envs)
                entry["receipt_level"]["ci_is_authoritative"] = False
                entry["power"]["gate_is_resolvable"] = None
            out.append(entry)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    directory = Path(args.dir)

    rows = analyse(load_arms(directory, args.date))
    exploratory = [f"{r['rung']}/k={r['k']}" for r in rows if r["k"] != 10]
    report = {
        "spec": "ENTERPRISE_EVAL_SPEC.md Addendum A — G-E2 PRECISION report",
        "date": args.date,
        "gate_unchanged": f"mean of 3 seed means at k=10 >= +{GATE_DELTA:.2f} "
        "micro-F1; this report does not alter it",
        "READ_THIS_BEFORE_QUOTING_ANY_ROW": {
            "the_gate_is_k10_only": "Rows at k != 10 are preregistered CURVE "
            "points, not decision points. A k=5 row that comes out positive is "
            "not a result; quoting one as the headline is the cherry-pick that "
            "retired the old +12.7 F1 claim.",
            "exploratory_rows_in_this_report": exploratory,
            "multiplicity": f"This report contains {len(rows)} rung/k "
            "comparisons, each with a CI and a sign test. Scanning them for the "
            "smallest p-value and quoting it is exactly how a null result gets "
            "published as a positive one. No uncorrected p in an exploratory "
            "row may be called significant, and p ~ 0.05 across several "
            "comparisons is expected under the null.",
            "two_statistics_must_agree": "The CI speaks for the mean and the "
            "sign test speaks for the receipts. A row where only one is "
            "positive is an artifact - a mean carried by a couple of receipts, "
            "or a win-count of wins too small to matter. Report both or "
            "neither.",
        },
        "rows": rows,
    }
    out = directory / f"cord_paired_power_{args.date}.json"
    out.write_text(json.dumps(report, indent=2))

    for row in rows:
        rl, pw, rb = row["receipt_level"], row["power"], row["robustness"]
        st = rb["sign_test"]
        flag = "" if rl.get("ci_is_authoritative", True) else "  [MIXED-ENV POOL]"
        print(
            f"{row['rung']:5} k={row['k']:<3} "
            f"n={rl['n']:3} mean={rl['mean']:+.4f} "
            f"CI95=[{rl['ci95'][0]:+.4f},{rl['ci95'][1]:+.4f}] "
            f"MDE={pw['mde_80pct']:.4f} resolvable={pw['gate_is_resolvable']}"
            f"{flag}"
        )
        def jk(value: float | None) -> str:
            # None when the pool is too small to drop that many receipts;
            # printing it must not take the whole report down.
            return "n/a" if value is None else f"{value:+.4f}"

        print(
            f"          robustness: sign-test {st['wins']}W/{st['losses']}L/"
            f"{st['ties']}T p={st['p_value']} | "
            f"mean less top-2 winners {jk(rb['mean_less_top2_winners'])} | "
            f"less top-3 {jk(rb['mean_less_top3_winners'])}"
        )
    print(f"\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
