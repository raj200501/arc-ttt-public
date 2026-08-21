#!/usr/bin/env python3
"""Recompute the k=30 novel-schema gate verdict from raw receipts.

Zero dependencies (Python 3.8+ stdlib only). Run from the repo root:

    python3 scripts/verify_verdict.py

It loads the six per-arm artifacts (three seeds x adapted/kshot),
recomputes every statistic from the per-receipt records — per-arm means,
paired deltas, the sign test, receipt-level and cluster-level confidence
intervals, validity windows, attrition — and cross-checks the published
summary. Nothing is trusted from the summary; everything is derived from
the receipts. Trust boundary, stated plainly: this is an ARITHMETIC
audit — the per-receipt scores themselves are the boundary. For
artifacts that store raw predictions (Addendum E onward),
verify_from_primary.py moves the boundary further: it regenerates the
gold labels from the deterministic generator and re-scores every stored
prediction with the real scorer. The full re-run path (kaggle/ entries,
free tier) moves it all the way. The claim rules are frozen in
docs/research/ENTERPRISE_EVAL_SPEC.md (Addendum B; corrections in
B.9), whose hash is anchored via OpenTimestamps
(ENTERPRISE_EVAL_SPEC.md.ots).

Scoping reminder printed with the result (spec B.9.1): both arms carry
the full 30-shot prompt — the delta measures adaptation ADDED ON TOP of
in-context prompting at 0.5B. And per B.6 the positive is always stated
beside the CORD negative: the same recipe FAILED its preregistered CORD
gates at all three scales tested.
"""

import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from novel_schema_summary import t95  # noqa: E402  (single estimator authority)

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
DATE = "2026-08-12"  # spec-freeze date carried in artifact filenames
SEEDS = (1, 2, 3)
BAR = 0.05
FLOOR, CEILING = 0.15, 0.95
DESIGNED_PER_SEED = 60


def load_arm(seed: int, arm: str) -> dict:
    path = EXP / f"novel_schema_0.5b_k30_seed{seed}_{arm}_{DATE}.json"
    record = json.loads(path.read_text())
    for key, want in (("device", "cpu"), ("dtype", "torch.float32"),
                      ("k", 30), ("arm", arm)):
        assert record[key] == want, f"{path.name}: {key}={record[key]!r}"
    assert len(record["results"]) == DESIGNED_PER_SEED, path.name
    return record


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def main() -> int:
    print("== recomputing from per-receipt records (summary not trusted) ==")
    deltas_all = []          # paired per-receipt deltas, pooled
    seed_deltas = []         # per-seed mean deltas (cluster level)
    wins = losses = ties = 0
    scored_total = excluded_total = 0
    validity_ok = True

    for seed in SEEDS:
        adapted = load_arm(seed, "adapted")
        kshot = load_arm(seed, "kshot")
        a = {r["index"]: r["micro_f1"] for r in adapted["results"] if "micro_f1" in r}
        k = {r["index"]: r["micro_f1"] for r in kshot["results"] if "micro_f1" in r}
        # attrition must be symmetric: same excluded docs in both arms
        assert set(a) == set(k), f"seed {seed}: asymmetric attrition"
        excluded = DESIGNED_PER_SEED - len(a)
        scored_total += len(a)
        excluded_total += excluded

        mean_a = sum(a.values()) / len(a)
        mean_k = sum(k.values()) / len(k)
        # recomputed arm means must match the artifacts' own stamps
        assert abs(mean_a - adapted["mean_micro_f1"]) < 5e-4, f"seed {seed} adapted"
        assert abs(mean_k - kshot["mean_micro_f1"]) < 5e-4, f"seed {seed} kshot"
        if not (FLOOR <= mean_k <= CEILING):
            validity_ok = False

        seed_delta = mean_a - mean_k
        seed_deltas.append(seed_delta)
        for i in a:
            d = a[i] - k[i]
            deltas_all.append(d)
            # Tie convention note (spec errata P9): this verifier counts a
            # win/loss at |delta| > 1e-12 (float-noise tolerance), whereas
            # the shipped B.3 rule in cord_paired_power.py uses tol=0.01
            # ("micro-F1 differences below a point are not a win"). On the
            # banked receipts the two conventions yield identical tallies
            # (156W/0L/2T) and identical verdicts; the math here is frozen
            # deliberately — do not change it without a spec errata.
            wins += d > 1e-12
            losses += d < -1e-12
            ties += abs(d) <= 1e-12
        print(f"seed {seed}: adapted {mean_a:.4f}  kshot {mean_k:.4f}  "
              f"delta {seed_delta*100:+.1f} F1  "
              f"(scored {len(a)}/{DESIGNED_PER_SEED}, excluded {excluded})")

    n = len(deltas_all)
    mean = sum(deltas_all) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in deltas_all) / (n - 1))
    # Both intervals below take their quantile from novel_schema_summary,
    # the authorized reader, so there is exactly one estimator in the repo.
    # Three rounds of this number were wrong for three different reasons:
    # the normal quantile here where the reader used t; then a transcribed
    # constant (1.980) standing in for t at df=157; and, in the reader
    # itself, a lookup table that invented 2.09 for every df below 39.
    # Copying a constant is how all three happened, so nothing is copied
    # any more — t95() computes the quantile and both callers ask it.
    half = t95(n - 1) * sd / math.sqrt(n)
    ci_lo, ci_hi = mean - half, mean + half

    cmean = sum(seed_deltas) / 3
    csd = math.sqrt(sum((d - cmean) ** 2 for d in seed_deltas) / 2)
    chalf = t95(2) * csd / math.sqrt(3)

    # exact binomial sign test, ties dropped, one-sided P(X >= wins)
    m = wins + losses
    log_p = None
    if m:
        total = 0.0
        for x in range(wins, m + 1):
            total += math.comb(m, x)
        log_p = math.log(total) - m * math.log(2)

    print(f"\npooled receipts: n={n} scored of {3*DESIGNED_PER_SEED} designed "
          f"({excluded_total} excluded, symmetric both arms)")
    print(f"mean paired delta: {mean*100:+.2f} F1  "
          f"(receipt-level 95% CI [{ci_lo*100:.1f}, {ci_hi*100:.1f}])")
    print(f"cluster level (3 seeds): mean {cmean*100:+.1f}, "
          f"95% CI [{(cmean-chalf)*100:.1f}, {(cmean+chalf)*100:.1f}]")
    print(f"sign test: {wins}W/{losses}L/{ties}T  "
          f"(one-sided p {'< 1e-15' if log_p is not None and log_p < -34.5 else f'= {math.exp(log_p):.2e}'})")
    print(f"validity windows [{FLOOR}, {CEILING}] on kshot arms: "
          f"{'clear' if validity_ok else 'TRIPPED'}")

    # decision per B.3: the gate statistic is the mean over SEEDS,
    # with the interval and the sign test agreeing, validity clear
    verdict = ("GO" if (cmean >= BAR and ci_lo > 0 and wins > losses
                        and validity_ok) else "NOT GO")

    summary = json.loads((EXP / f"novel_schema_summary_{DATE}.json").read_text())
    s_gate = summary["gate_k30"]
    match = (s_gate["pairs_complete"] == 3
             and abs(s_gate["mean_delta"] - cmean) < 5e-4        # seed mean (B.3)
             and abs(s_gate["receipt_level"]["mean"] - mean) < 5e-4  # pooled
             and abs(s_gate["receipt_level"]["ci95"][0] - ci_lo) < 1e-3
             and s_gate["receipt_level"]["n"] == n
             and s_gate["receipt_level"]["sign_test"]["wins"] == wins
             and summary["VERDICT"] == "GO" == verdict)
    print(f"\nrecomputed verdict: {verdict} (decision statistic = seed mean "
          f"{cmean*100:+.1f} vs +{BAR*100:.0f} bar; pooled receipt mean "
          f"{mean*100:+.1f} reported alongside)")
    print(f"published summary cross-check: {'MATCHES' if match else 'MISMATCH'}")
    print("\nscoping (spec B.9.1): both arms carry the full 30-shot prompt — this")
    print("measures adaptation ADDED ON TOP of in-context prompting at 0.5B.")
    print("Per B.6, stated beside the CORD negative: the same recipe FAILED its")
    print("preregistered CORD gates at all three scales (-7.3/-11.5/-4.5 F1).")
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main())
