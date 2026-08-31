#!/usr/bin/env python3
"""Addendum R: is the fence caused by prompt length, or by the format?

    PYTHONPATH=src python3 scripts/addendum_r_table.py

The fence impact table established that one prompt regime fences every
output and another fences none. It could not say WHY, and two boring
explanations were live:

  (a) LENGTH. The schema prompt is short (196 tokens); the k-shot prompt
      is long. Maybe a short prompt just leaves the model in chat mode.
  (b) DEMONSTRATIONS AS SUCH. Maybe seeing any worked example at all is
      what stops the model reaching for a fence.

Both are testable with cells that were preregistered before any of them
ran, and both make predictions that the other does not:

  * If (a), a LONGER schema prompt should fence less.
  * If (b), a schema prompt with one demonstration should fence less.

The discriminator is `schema_kshot` -- the schema block PLUS worked
examples -- read against pure k-shot at matched and mismatched lengths.

This script reads the banked arms and classifies every stored output with
the SHIPPED tool (`tools/fencecheck.py`), rather than with a private
copy of the same logic. If the tool is wrong about what a fence is, this
table is wrong in the same direction, which is the correct coupling: the
number we publish and the instrument we hand out cannot disagree.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO / "experiments"

# (family, artifact, what the prompt contains). The demonstration count
# is NOT written here -- it is read from each artifact and printed from
# there. The first version of this table hardcoded "k-shot k=10" for an
# arm the artifact records as 20, which is the same defect as every other
# number this repository restated by hand instead of reading: a label
# asserting something the file next to it contradicts.
ARMS = (
    ("schema", "waybill_scale_rung_0.5b_schema_2026-08-25.json",
     "the tenant's field list, nothing else"),
    ("schema", "waybill_fence_dose_k1schema_2026-08-25.json",
     "the field list AND worked examples"),
    ("k-shot", "waybill_fence_dose_k1_2026-08-25.json",
     "demonstrations as chat turns, no field list"),
    ("k-shot", "waybill_fence_dose_k3_2026-08-25.json",
     "demonstrations as chat turns, no field list"),
    ("k-shot", "waybill_scale_rung_0.5b_kshot_2026-08-25.json",
     "demonstrations as chat turns, no field list"),
)


def _label(family: str, demos: int) -> str:
    if family == "schema":
        return "schema only" if demos == 0 else f"schema + {demos} demo"
    return f"k-shot k={demos}"


def _tool():
    spec = importlib.util.spec_from_file_location(
        "fencecheck", REPO / "tools" / "fencecheck.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(
        EXPERIMENTS / "addendum_r_fence_dose.json"))
    args = parser.parse_args()

    fc = _tool()
    rows, missing = [], []
    for family, name, prompt in ARMS:
        path = EXPERIMENTS / name
        if not path.exists():
            missing.append(name)
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        label = _label(family, record["n_demonstrations"])
        texts = list(record["predictions"].values())
        fenced = sum(1 for t in texts if fc.strip_fence(t)[1])
        rows.append({
            "arm": label,
            "prompt_contains": prompt,
            "artifact": name,
            "model": record["model"],
            "n_demonstrations": record["n_demonstrations"],
            "mean_prompt_tokens": record["mean_prompt_tokens"],
            "n": len(texts),
            "fenced": fenced,
            "fence_rate": round(fenced / len(texts), 4) if texts else None,
            "parses_as_written": sum(1 for t in texts if fc._parses(t)),
            "parses_after_stripping": sum(
                1 for t in texts if fc._parses(fc.strip_fence(t)[0])),
            "mean_micro_f1_as_scored": record["mean_micro_f1"],
        })

    record = {
        "what": "Addendum R. Every arm's stored outputs classified by "
                "tools/fencecheck.py, to separate prompt LENGTH from prompt "
                "FORMAT as the cause of fencing.",
        "model": "Qwen/Qwen2.5-0.5B-Instruct throughout — one model, so the "
                 "only thing varying across rows is the prompt.",
        "classified_by": "tools/fencecheck.py strip_fence(), the shipped "
                         "tool, not a private reimplementation.",
        "rows": rows,
        "missing_arms": missing,
    }

    if missing:
        record["the_finding"] = (
            "WITHHELD: " + ", ".join(missing) + " have not been banked. "
            "The reading is stated once every preregistered cell exists, "
            "not from the subset that finished first.")
    else:
        by = {r["arm"]: r for r in rows}
        schema = by["schema only"]
        kshot_arms = sorted(
            (r for r in rows if r["arm"].startswith("k-shot")),
            key=lambda r: r["n_demonstrations"])
        schema_demo = next(r for r in rows
                           if r["arm"].startswith("schema +"))
        f1 = kshot_arms[0]["fence_rate"]
        f3 = kshot_arms[1]["fence_rate"] if len(kshot_arms) > 1 else None
        fs1 = schema_demo["fence_rate"]
        # The FROZEN readings from the Addendum R preregistration row are
        # applied by their own arithmetic, not re-derived from the shape
        # of the data. The first version of this script announced "it is
        # the FORMAT" -- a conclusion not on the preregistered menu, whose
        # nearest licensed reading (d) did not fire by its own condition.
        # A preregistration binds its author first.
        if f1 <= 0.10 and fs1 > 0.50:
            reading = ("(d) THE SCHEMA LINE ITSELF PROVOKES IT: one "
                       "demonstration suffices in the k-shot regime but "
                       "not beside the field list.")
        elif f1 <= 0.10:
            reading = ("(a) IT IS THE DEMONSTRATIONS, AND ONE IS ENOUGH.")
        elif f3 is not None and f3 <= 0.10:
            reading = ("(b) IT IS THE DEMONSTRATIONS, AND IT IS A DOSE: "
                       "the published sentence stands with its "
                       "dose-response curve attached, reported as "
                       "measured.")
        elif all(r["fence_rate"] > 0.50 for r in
                 [kshot_arms[0]] + ([kshot_arms[1]] if f3 is not None
                                    else []) + [schema_demo]):
            reading = ("(c) IT IS NOT THE DEMONSTRATIONS: the published "
                       "sentence is withdrawn and rewritten to say only "
                       "that the regimes differ.")
        else:
            reading = ("NO FROZEN READING FIRES: the rates fall between "
                       "the preregistered conditions and this table "
                       "claims nothing beyond the rates themselves.")
        record["frozen_reading_applied"] = {
            "reading": reading,
            "rates": {"f_k1": f1, "f_k3": f3, "f_schema_plus_demo": fs1},
        }
        record["observation_outside_the_preregistered_menu"] = (
            f"schema + demo fences at {fs1} while being LONGER than the "
            f"k=1 prompt that fences at {f1} -- the suppression measured "
            "in the k-shot family does not transfer into the schema "
            "regime, and length runs the wrong way for a length "
            "explanation. HYPOTHESIS-GENERATING, NOT CLAIM-BEARING: the "
            "reading that would license a format sentence did not fire.")
        record["the_finding"] = (
            reading + " Beside it, one observation outside the frozen "
            "menu, held at hypothesis status -- see "
            "observation_outside_the_preregistered_menu.")
        record["what_this_does_not_show"] = (
            "One model, one corpus, one seed, greedy decoding. It says "
            "nothing about other model families. It does not license any "
            "sentence about WHY the regimes differ beyond the frozen "
            "reading above — the format hypothesis stays a hypothesis "
            "until a follow-up with its own frozen readings runs. The "
            "k-shot arms are also worse at the task at low k, so nothing "
            "here is a recommendation to drop the schema.")

    pathlib.Path(args.out).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(f"{'arm':18s} {'demos':>5s} {'tokens':>7s} {'fenced':>8s} "
          f"{'as-written':>11s} {'stripped':>9s}")
    for row in rows:
        print(f"{row['arm']:18s} {row['n_demonstrations']:5d} "
              f"{row['mean_prompt_tokens']:7d} "
              f"{row['fenced']:4d}/{row['n']:<3d} "
              f"{row['parses_as_written']:11d} "
              f"{row['parses_after_stripping']:9d}")
    print(f"\n{record['the_finding']}\nbanked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
