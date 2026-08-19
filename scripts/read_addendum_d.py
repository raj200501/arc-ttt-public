#!/usr/bin/env python3
"""Compute the Addendum D reads from artifacts, per the frozen D.2 rules.

Zero dependencies. Run from the repo root once the doconly artifacts are
banked in experiments/:

    python3 scripts/read_addendum_d.py

Reads (frozen 2026-08-18T18:05Z, before any run):
- Read 1 (retention): seed-mean of [doconly − adapted-with-prompt] on
  scored-index intersections; PASS iff ≥ −5.0 F1.
- Read 2 (unified claim): seed-mean of [doconly − kshot-with-prompt];
  PASS iff ≥ +5.0 F1 with interval and sign test agreeing (B.3).
- D.5 comparability (non-gating): doconly − doczero, reported only.
Refuses to print a verdict until all three doconly artifacts exist.
"""

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
SEEDS = (1, 2, 3)
B_DATE, D_DATE = "2026-08-12", "2026-08-18"


def scores(path: pathlib.Path) -> dict:
    record = json.loads(path.read_text())
    assert record["device"] == "cpu" and record["dtype"] == "torch.float32", path.name
    return {r["index"]: r["micro_f1"] for r in record["results"] if "micro_f1" in r}


def main() -> int:
    missing = [s for s in SEEDS if not
               (EXP / f"novel_schema_d_0.5b_k30_seed{s}_doconly_{D_DATE}.json").exists()]
    if missing:
        print(f"REFUSED: doconly artifacts missing for seeds {missing} — "
              "no partial reads (B.3 discipline).")
        return 2

    r1, r2, comp = [], [], []
    r1_deltas_all = []
    r2_deltas_all = []
    for s in SEEDS:
        d = scores(EXP / f"novel_schema_d_0.5b_k30_seed{s}_doconly_{D_DATE}.json")
        a = scores(EXP / f"novel_schema_0.5b_k30_seed{s}_adapted_{B_DATE}.json")
        k = scores(EXP / f"novel_schema_0.5b_k30_seed{s}_kshot_{B_DATE}.json")
        i1 = sorted(set(d) & set(a))
        i2 = sorted(set(d) & set(k))
        m1 = sum(d[i] - a[i] for i in i1) / len(i1)
        m2 = sum(d[i] - k[i] for i in i2) / len(i2)
        r1.append(m1)
        r2.append(m2)
        r1_deltas_all += [d[i] - a[i] for i in i1]
        r2_deltas_all += [d[i] - k[i] for i in i2]
        line = (f"seed {s}: doconly {sum(d.values())/len(d):.4f}  "
                f"retention {m1*100:+.1f} (n={len(i1)})  "
                f"vs-kshot {m2*100:+.1f} (n={len(i2)})")
        dz_path = EXP / f"novel_schema_d_0.5b_k30_seed{s}_doczero_{D_DATE}.json"
        if dz_path.exists():
            z = scores(dz_path)
            iz = sorted(set(d) & set(z))
            cz = sum(d[i] - z[i] for i in iz) / len(iz)
            comp.append(cz)
            line += f"  adapter-contribution {cz*100:+.1f} (n={len(iz)}, comparability)"
        print(line)

    m_r1 = sum(r1) / 3
    m_r2 = sum(r2) / 3
    n = len(r2_deltas_all)
    mean2 = sum(r2_deltas_all) / n
    sd2 = math.sqrt(sum((x - mean2) ** 2 for x in r2_deltas_all) / (n - 1))
    lo2 = mean2 - 1.96 * sd2 / math.sqrt(n)
    wins = sum(x > 1e-12 for x in r2_deltas_all)
    losses = sum(x < -1e-12 for x in r2_deltas_all)
    ties = n - wins - losses

    # Read 1 sign test (frozen D.2: seed-mean bar WITH the sign test not
    # contradicting) — pooled retention deltas
    n1 = len(r1_deltas_all)
    wins1 = sum(x > 1e-12 for x in r1_deltas_all)
    losses1 = sum(x < -1e-12 for x in r1_deltas_all)
    ties1 = n1 - wins1 - losses1

    read1 = "PASS" if (m_r1 >= -0.05 and wins1 >= losses1) else "FAIL"
    read2 = "PASS" if (m_r2 >= 0.05 and lo2 > 0 and wins > losses) else "FAIL"
    print(f"\nRead 1 (retention, bar >= -5.0): seed-mean {m_r1*100:+.2f} F1, "
          f"sign {wins1}W/{losses1}L/{ties1}T -> {read1}")
    print(f"Read 2 (unified claim, bar >= +5.0): seed-mean {m_r2*100:+.2f} F1, "
          f"receipt CI low {lo2*100:+.1f}, sign {wins}W/{losses}L/{ties}T -> {read2}")
    if comp:
        print(f"D.5 comparability (non-gating): adapter contribution "
              f"seed-mean {sum(comp)/len(comp)*100:+.1f} F1 over doczero")
    print("\nOutcome branch per D.3:",
          "both reads PASS -> retire the B.9.1 caveat, publish artifact #2"
          if read1 == read2 == "PASS" else
          "a read FAILED -> publish the pre-written failure sentence verbatim; no spin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
