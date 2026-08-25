#!/usr/bin/env python3
"""Addendum G verdict: the ONLY authorized reader of G cells.

Applies the rule frozen in ENTERPRISE_EVAL_SPEC.md Addendum G before any
G arm existed, and decides nothing else:

  PASS           (a) Spearman(prompted baseline, raw paired delta) <= -0.60
                 AND (b) mean delta over the 4 lowest-baseline cells minus
                 mean delta over the 4 highest-baseline cells >= +10.0 F1
  PARTIAL        exactly one of (a), (b) holds — reported as PARTIAL and
                 NOT rounded up to a pass
  REFUTED        neither holds; the unifying story is not supported by
                 fresh data and nothing downstream may cite the law
  UNINFORMATIVE  fewer than 10 scoreable cells, or every prompted baseline
                 lands inside a 0.10 band (the dial failed to move
                 baseline strength, so the correlation is not measurable)

It also reports the CAPTURED-HEADROOM FRACTION, which is the honest
answer to the obvious objection: ceiling effects alone force a negative
correlation between baseline and raw delta, because a baseline at 0.975
cannot gain more than 0.025. The fraction is REPORTED, NOT GATING — no
threshold was preregistered for it, and inventing one after seeing the
data is exactly the move this whole apparatus exists to prevent.

    python3 scripts/addendum_g_summary.py
"""

from __future__ import annotations

import glob
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# All three frozen in spec G.3/G.4 before any arm ran.
RHO_BAR = -0.60
TERCILE_GAP_BAR = 0.10  # +10.0 F1
MIN_CELLS = 10
MIN_BASELINE_SPREAD = 0.10


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, average ranks for ties. Stdlib only."""

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def load_cells() -> list[dict]:
    cells = []
    for path in sorted(glob.glob(str(REPO / "experiments" /
                                     "novel_schema_gb_*.json"))):
        record = json.loads(pathlib.Path(path).read_text())
        if record.get("addendum") != "G-b":
            continue
        assert record["device"] == "cpu" and record["dtype"] == "torch.float32", (
            f"{path}: unexpected environment stamp")
        record["_file"] = pathlib.Path(path).name
        cells.append(record)
    return cells


def main() -> int:
    cells = load_cells()
    if not cells:
        # Zero cells is the CURRENT state (spec G.10: compute-bound), and
        # VERDICT.md tells readers this script says UNINFORMATIVE. It used
        # to exit(1) with "no artifacts found", so the doc was describing
        # behaviour the code did not have. Fixed here rather than in the
        # sentence: below the cell floor the honest verdict is
        # UNINFORMATIVE, whether the count is 9 or 0.
        print("Addendum G — adaptation-headroom law: 0 cells banked.\n")
        print("VERDICT: UNINFORMATIVE — fewer than "
              f"{MIN_CELLS} scoreable cells. Per G.4 the correlation is not "
              "interpreted,\nand per G.10 nothing in this repository may "
              "cite the law. The design, the frozen\nbars and the measured "
              "reason it has not run are in ENTERPRISE_EVAL_SPEC.md G.1-G.10.")
        return 0

    cells.sort(key=lambda c: (c["seed"], c["j"]))
    print(f"Addendum G — adaptation-headroom law, {len(cells)} cells "
          f"(bar frozen before any arm ran)\n")
    print(f"{'seed':>5}{'j':>4}{'prompted':>11}{'adapted':>10}{'delta':>10}"
          f"{'headroom captured':>20}{'valid JSON p/a':>17}")
    for c in cells:
        frac = c.get("captured_headroom_fraction")
        frac_s = f"{frac:+.3f}" if frac is not None else "  n/a"
        print(f"{c['seed']:>5}{c['j']:>4}{c['kshot_mean_micro_f1']:>11.4f}"
              f"{c['adapted_mean_micro_f1']:>10.4f}"
              f"{c['paired_mean_delta']:>+10.4f}{frac_s:>20}"
              f"{str(c['kshot_valid_json']) + '/' + str(c['adapted_valid_json']):>17}")

    baselines = [c["kshot_mean_micro_f1"] for c in cells]
    deltas = [c["paired_mean_delta"] for c in cells]
    spread = max(baselines) - min(baselines)

    print(f"\nbaseline range: {min(baselines):.4f} .. {max(baselines):.4f} "
          f"(spread {spread:.4f})")

    if len(cells) < MIN_CELLS or spread < MIN_BASELINE_SPREAD:
        why = ("too few scoreable cells" if len(cells) < MIN_CELLS
               else "the j dial did not move baseline strength")
        print(f"\nVERDICT: UNINFORMATIVE — {why}. Per G.4 the correlation is "
              "not interpreted.")
        return 0

    rho = spearman(baselines, deltas)
    ordered = sorted(zip(baselines, deltas))
    low = [d for _, d in ordered[:4]]
    high = [d for _, d in ordered[-4:]]
    gap = sum(low) / len(low) - sum(high) / len(high)

    a_holds = rho <= RHO_BAR
    b_holds = gap >= TERCILE_GAP_BAR

    print(f"\n(a) Spearman(baseline, delta) = {rho:+.4f}   "
          f"bar <= {RHO_BAR:+.2f}   -> {'HOLDS' if a_holds else 'does not hold'}")
    print(f"(b) low-4 mean delta {sum(low) / len(low):+.4f} minus high-4 "
          f"{sum(high) / len(high):+.4f} = {gap:+.4f}   "
          f"bar >= +{TERCILE_GAP_BAR:.2f}   "
          f"-> {'HOLDS' if b_holds else 'does not hold'}")

    fracs = [(c["kshot_mean_micro_f1"], c["captured_headroom_fraction"])
             for c in cells if c.get("captured_headroom_fraction") is not None]
    if len(fracs) >= MIN_CELLS:
        frho = spearman([b for b, _ in fracs], [f for _, f in fracs])
        print(f"\nMECHANISM (reported, NOT gating — no threshold was "
              f"preregistered):\n  Spearman(baseline, captured-headroom "
              f"fraction) = {frho:+.4f}")
        if frho > -0.3:
            print("  The fraction does NOT fall as the baseline rises, so "
                  "adaptation captures a\n  similar share of whatever room "
                  "exists. Per G.2 the law is then substantially a\n  CEILING "
                  "EFFECT: still a true and actionable buying rule, but NOT "
                  "evidence\n  that adaptation is better at hard documents.")
        else:
            print("  The fraction falls as the baseline rises, i.e. "
                  "adaptation captures MORE of\n  the available room where "
                  "prompting is weak. That is more than a ceiling\n  effect.")

    verdict = ("PASS" if (a_holds and b_holds)
               else "PARTIAL" if (a_holds or b_holds) else "REFUTED")
    print(f"\nVERDICT: {verdict}")
    if verdict == "PARTIAL":
        print("Exactly one preregistered reading holds. Per G.4 this is "
              "reported as PARTIAL and\nis not rounded up to a pass.")
    if verdict == "REFUTED":
        print("Neither reading holds. Per G.4 the unifying story is NOT "
              "supported by fresh data;\nthe post-hoc table in G.1 is an "
              "artifact of the corpora it came from, and nothing\ndownstream "
              "may cite the law.")
    if verdict == "PASS":
        print("Per G.5 this licenses: the measured value of adaptation at "
              "0.5B is predictable in\nadvance from a cheap prompted-baseline "
              "measurement on the tenant's own documents.\nIt does NOT "
              "license any claim above 0.5B, any claim about real customer\n"
              "documents, or 'better at hard documents' unless the mechanism "
              "line above says so.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
