#!/usr/bin/env python3
"""How far can a stranger actually re-derive each result? Machine-checked.

The strongest thing this repository offers is that a reader does not have
to trust it. But "verifiable" is not one property, it is three, and until
now the difference lived in prose that a reader had to piece together
from `VERDICT.md`'s footnotes:

  PRIMARY      raw predictions are stored, so gold can be regenerated
               from the deterministic generator and every prediction
               re-scored. The trust boundary is the generator seed plus
               the scorer code, both in this tree.
  ARITHMETIC   per-item scores are stored but the predictions that
               produced them are not. A reader can re-add the numbers.
               They cannot check that model output -> score was done
               right, only that score -> mean was.
  AGGREGATE    only summary statistics survive. Nothing is re-derivable;
               the number must be taken on trust.
  EXTERNAL     the figure is a third-party quote (a list price, a
               leaderboard row). Not ours to verify and labelled as such.

Every reader who has audited this project asked some version of "which
numbers can I actually recompute?" and the honest answer has always been
"it depends on the row". This makes that answer machine-checked and
publishes the WEAK rows as loudly as the strong ones -- including the
+46.5 headline, which is the least verifiable number on the page.

An artifact is downgraded, never upgraded, by ambiguity: if this cannot
find stored predictions it says ARITHMETIC even if the artifact claims
more, because the failure mode being guarded against is a page implying
more checkability than it has.

    PYTHONPATH=src python3 scripts/verification_coverage.py
"""

from __future__ import annotations

import argparse
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO / "experiments"

# Gate exhaust, not evidence. These two files are what verifiers WRITE,
# so counting them as banked artifacts would let a gate inflate the
# coverage total by running -- the same laundering defect the
# reconciliation gate was already caught committing against its own
# output. Named here and banked in the record so anything that needs to
# reproduce this total reads the rule instead of re-deriving it; a
# downstream copy of this tuple is a copy that goes stale on its own.
EXHAUST = ("verification_coverage.json", "outbound_reconciliation.json")

# Artifacts whose figures are third-party quotes rather than our runs.
EXTERNAL_MARKERS = ("external list price", "external quote",
                    "not a measurement", "leaderboard")


def _classify(record: object) -> tuple[str, str]:
    """PRIMARY / ARITHMETIC / AGGREGATE / EXTERNAL, plus the reason."""
    blob = json.dumps(record)
    if isinstance(record, dict):
        text = " ".join(str(v) for v in record.values() if isinstance(v, str))
        external = any(marker in text.lower() for marker in EXTERNAL_MARKERS)
        if external and '"prediction"' not in blob and '"predictions"' not in blob:
            return "EXTERNAL", ("figures are third-party quotes, labelled "
                                "as such and not ours to verify")
    if '"prediction"' in blob or '"predictions"' in blob:
        return "PRIMARY", ("stores raw predictions: gold can be regenerated "
                           "and every prediction re-scored")
    if '"micro_f1"' in blob or "per_document" in blob or "per_receipt" in blob:
        return "ARITHMETIC", ("stores per-item scores but not the predictions "
                              "that produced them: a reader can re-add the "
                              "numbers, not re-derive them from model output")
    return "AGGREGATE", ("only summary statistics survive; nothing here is "
                         "re-derivable and the figure must be taken on trust")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(
        EXPERIMENTS / "verification_coverage.json"))
    args = parser.parse_args()

    rows = []
    for path in sorted(EXPERIMENTS.glob("*.json")):
        if path.name in EXHAUST:
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            rows.append({"artifact": path.name, "level": "UNREADABLE",
                         "why": "not valid JSON"})
            continue
        level, why = _classify(record)
        rows.append({"artifact": path.name, "level": level, "why": why,
                     "bytes": path.stat().st_size})

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["level"]] = counts.get(row["level"], 0) + 1

    record = {
        "what": "How far each banked artifact can be re-derived by a "
                "stranger, classified mechanically.",
        "levels": {
            "PRIMARY": "raw predictions stored; gold regenerable; every "
                       "prediction re-scorable. Trust boundary is the "
                       "generator seed plus the scorer code.",
            "ARITHMETIC": "per-item scores stored, predictions not. The "
                          "aggregation is checkable; the step from model "
                          "output to score is not.",
            "AGGREGATE": "summary statistics only. Not re-derivable.",
            "EXTERNAL": "third-party quote, labelled as such, not ours to "
                        "verify.",
        },
        "counts": counts,
        "total_artifacts": len(rows),
        "excluded_as_gate_exhaust": list(EXHAUST),
        # Banked as a COUNT as well as a fraction, because the count is
        # the form outbound copy actually quotes -- "33 of 181 banked
        # artifacts are primary-verifiable". Reconstructing it downstream
        # as fraction x total is a rounding bug waiting for its first
        # off-by-one, and it would be an off-by-one in the one number on
        # the page that describes how checkable the rest of the page is.
        "primary_verifiable": counts.get("PRIMARY", 0),
        "primary_verifiable_fraction": round(
            counts.get("PRIMARY", 0) / len(rows), 4) if rows else 0.0,
        "why_the_total_is_banked": (
            "Outbound copy cites this total ('179 banked artifacts') and "
            "the reconciliation gate had been matching it against an "
            "unrelated 179 in an unrelated artifact -- a figure that "
            "traced to nothing, passing because some number of that value "
            "happened to exist somewhere. A cited figure has to be a "
            "figure this map deliberately publishes."),
        "artifacts": rows,
        "the_uncomfortable_one": (
            "The +46.5 headline is the least verifiable number here, and "
            "this classifier is stricter about it than the prose first "
            "written beside it was. `novel_schema_summary_2026-08-12.json` "
            "is AGGREGATE -- a summary with no per-item data at all -- and "
            "its ARMS are ARITHMETIC: per-receipt scores, no predictions. "
            "So the step from model output to score cannot be re-checked "
            "by anyone, including us, at any level. It is the most cited "
            "number in this repository and the only headline that is "
            "neither primary-verifiable nor regenerable. Re-running those "
            "arms with predictions stored is the fix, it costs about nine "
            "CPU-hours, and it is not done."),
        "aggregate_but_regenerable": (
            "AGGREGATE is not always as weak as it sounds, and the "
            "distinction matters. Addendum K's artifact "
            "(waybill_grounded_2026-08-22.json) stores no predictions "
            "because the writer excluded them -- but every INPUT it "
            "consumed is published (the banked adapted arm's predictions, "
            "the 30 documents, the gold) and the recipe is pinned to a "
            "named commit, so a reader regenerates it in seconds with the "
            "command in its VERDICT row. Contrast the +46.5 arms, whose "
            "model outputs were discarded at generation time and are gone "
            "for good. Both read AGGREGATE here; only one is recoverable, "
            "and this classifier deliberately does not upgrade either, "
            "because ambiguity downgrades."),
        "downgrade_rule": "Ambiguity downgrades. If stored predictions "
                          "cannot be found this reports ARITHMETIC even "
                          "where the artifact claims more, because the "
                          "failure being guarded against is a page implying "
                          "more checkability than it has.",
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")

    order = ["PRIMARY", "ARITHMETIC", "AGGREGATE", "EXTERNAL", "UNREADABLE"]
    for level in order:
        members = [r for r in rows if r["level"] == level]
        if not members:
            continue
        print(f"\n{level}  ({len(members)})")
        for row in members[:60]:
            print(f"    {row['artifact']}")
    total = len(rows)
    primary = counts.get("PRIMARY", 0)
    print(f"\n{primary}/{total} artifacts are PRIMARY-verifiable "
          f"({primary / total * 100:.0f}%).")
    print(record["the_uncomfortable_one"])
    print(f"\nbanked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
