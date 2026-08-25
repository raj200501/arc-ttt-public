#!/usr/bin/env python3
"""Read Addendum H under its frozen rule. Written BEFORE the arms landed.

This script was committed while the H cells were still running, for the
same reason the bar was: a reader written after seeing the numbers is a
reader that can be shaped by them. It applies the rule in spec Addendum
H.3 and returns one of the four preregistered readings. It decides
nothing else.

    (a) mnemonic delta >= +5.0 AND sign test agrees -> the effect is NOT
        an artifact of the arbitrary label->key mapping
    (b) positive but under the bar -> SUBSTANTIALLY an artifact; the
        headline is restated everywhere with the mnemonic number beside
        it at the same size
    (c) at or below zero -> the headline IS an artifact of the generator,
        and the pages say so in those words
    (u) mnemonic prompted baseline >= 0.95 -> UNINFORMATIVE. A task with
        no headroom cannot show a delta. This guard binds BEFORE (b) and
        (c) so a ceiling effect cannot be banked as a refutation, just as
        it stops one being banked as a pass.

Missing cells are reported as missing. A partial sweep returns
INCOMPLETE rather than a verdict on the cells that happen to exist --
reading three seeds when three more are still running is how a sweep
gets stopped at a flattering moment.

    python3 scripts/addendum_h_summary.py
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

BAR = 0.05
SATURATION = 0.95
SEEDS = (1, 2, 3)


def load_cells(pattern: str) -> dict[int, dict]:
    cells = {}
    for path in sorted(glob.glob(str(REPO / "experiments" / pattern))):
        record = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        cells[record["seed"]] = record
    return cells


def arm_summary(cells: dict[int, dict], label: str) -> dict:
    seeds = sorted(cells)
    missing = [s for s in SEEDS if s not in cells]
    deltas = [cells[s]["paired_mean_delta"] for s in seeds]
    baselines = [cells[s]["kshot_mean_micro_f1"] for s in seeds]
    adapted = [cells[s]["adapted_mean_micro_f1"] for s in seeds]
    wins = sum(c["sign_test"]["wins"] for c in cells.values())
    losses = sum(c["sign_test"]["losses"] for c in cells.values())
    ties = sum(c["sign_test"]["ties"] for c in cells.values())
    n = wins + losses
    p = (sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n
         if n else 1.0)
    mean = sum(deltas) / len(deltas) if deltas else float("nan")
    mean_base = sum(baselines) / len(baselines) if baselines else float("nan")
    out = {
        "arm": label,
        "seeds_present": seeds,
        "seeds_missing": missing,
        "seed_deltas": [round(d, 4) for d in deltas],
        "seed_mean_delta": round(mean, 4) if deltas else None,
        "seed_mean_baseline": round(mean_base, 4) if baselines else None,
        "seed_mean_adapted": round(sum(adapted) / len(adapted), 4)
        if adapted else None,
        "pooled_sign_test": {"wins": wins, "losses": losses, "ties": ties,
                             "p_value": p},
        "baseline_saturated": bool(baselines) and mean_base >= SATURATION,
    }
    return out


def reading(mnemonic: dict, control: dict) -> tuple[str, str]:
    if mnemonic["seeds_missing"] or control["seeds_missing"]:
        return "INCOMPLETE", (
            "Cells are still missing: mnemonic "
            f"{mnemonic['seeds_missing']}, control "
            f"{control['seeds_missing']}. No reading is returned on a "
            "partial sweep -- reading the cells that happen to exist is "
            "how a sweep gets stopped at a flattering moment.")
    if mnemonic["baseline_saturated"]:
        return "(u) UNINFORMATIVE", (
            "The mnemonic prompted baseline is "
            f"{mnemonic['seed_mean_baseline']:.4f} >= {SATURATION}. A task "
            "with no headroom cannot show a delta, so this sweep says "
            "nothing about the ablation in either direction. Per the "
            "frozen guard this is NOT reading (c) and must not be "
            "reported as one.")
    delta = mnemonic["seed_mean_delta"]
    st = mnemonic["pooled_sign_test"]
    agrees = st["p_value"] < 0.05 and st["wins"] > st["losses"]
    if delta >= BAR and agrees:
        return "(a) SURVIVES", (
            f"The mnemonic-mapping delta is {delta:+.4f} with the sign "
            f"test agreeing ({st['wins']}W/{st['losses']}L/{st['ties']}T, "
            f"p={st['p_value']:.3g}). The effect is NOT an artifact of "
            "the arbitrary label->key mapping. This is the strongest "
            "cheap control proposed against this result.")
    if delta > 0:
        return "(b) SUBSTANTIALLY AN ARTIFACT", (
            f"The mnemonic-mapping delta is {delta:+.4f} -- positive but "
            f"under the +{BAR:.2f} bar"
            + ("" if agrees else ", and the sign test does not agree "
               f"({st['wins']}W/{st['losses']}L/{st['ties']}T, "
               f"p={st['p_value']:.3g})")
            + f". Against the arbitrary-mapping control's "
            f"{control['seed_mean_delta']:+.4f} on the same documents, the "
            "effect is substantially a property of the mapping. Per the "
            "frozen reading, the headline is restated everywhere with "
            "this number beside it at the same size.")
    return "(c) ARTIFACT OF THE GENERATOR", (
        f"The mnemonic-mapping delta is {delta:+.4f}, at or below zero, "
        f"against the arbitrary-mapping control's "
        f"{control['seed_mean_delta']:+.4f} on byte-identical documents. "
        "Per the reading frozen before these arms ran: the headline is an "
        "artifact of the generator, and this project's pages say so in "
        "those words.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(
        REPO / "experiments" / "novel_schema_h_summary_2026-08-22.json"))
    args = parser.parse_args()

    control = arm_summary(
        load_cells("novel_schema_h_0.5b_k10_seed*_arbitrary_*.json"),
        "arbitrary mapping (control, re-run on this host)")
    mnemonic = arm_summary(
        load_cells("novel_schema_h_0.5b_k10_seed*_mnemonic_*.json"),
        "mnemonic mapping (the ablation)")
    nodecoy = arm_summary(
        load_cells("novel_schema_h_0.5b_k10_seed*_nodecoy_*.json"),
        "arbitrary mapping, no distractor lines (H-B)")

    verdict, why = reading(mnemonic, control)
    record = {
        "addendum": "H",
        "rule": {"bar_delta": BAR,
                 "requires": "seed-mean delta >= bar AND the sign test "
                             "agreeing",
                 "saturation_guard": SATURATION,
                 "frozen_in": "ENTERPRISE_EVAL_SPEC.md Addendum H.3 and the "
                              "VERDICT.md row, both committed before any "
                              "arm ran"},
        "reader_written": "before the arms landed, deliberately",
        "control": control,
        "mnemonic": mnemonic,
        "no_distractors": nodecoy,
        "verdict": verdict,
        "why": why,
    }
    if control["seed_mean_delta"] is not None and \
            mnemonic["seed_mean_delta"] is not None:
        record["mapping_cost"] = round(
            control["seed_mean_delta"] - mnemonic["seed_mean_delta"], 4)
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")

    for block in (control, mnemonic, nodecoy):
        print(f"\n{block['arm']}")
        if not block["seeds_present"]:
            print("  no cells banked yet")
            continue
        st = block["pooled_sign_test"]
        print(f"  seeds {block['seeds_present']}"
              + (f"  MISSING {block['seeds_missing']}"
                 if block["seeds_missing"] else ""))
        print(f"  baseline {block['seed_mean_baseline']:.4f} -> adapted "
              f"{block['seed_mean_adapted']:.4f}   delta "
              f"{block['seed_mean_delta']:+.4f}   "
              f"{st['wins']}W/{st['losses']}L/{st['ties']}T "
              f"p={st['p_value']:.3g}")
        if block["baseline_saturated"]:
            print(f"  NOTE: baseline >= {SATURATION} -> saturated")
    if "mapping_cost" in record:
        print(f"\nWhat the arbitrary mapping is worth: "
              f"{record['mapping_cost']:+.4f} F1 of the measured delta")
    print(f"\nVERDICT: {verdict}\n{why}")
    print(f"\nbanked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
