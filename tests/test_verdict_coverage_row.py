"""VERDICT's coverage row must quote the coverage map it names.

The row said "all 236 banked artifacts" and "AGGREGATE 92" for four weeks
after `experiments/verification_coverage.json` moved to 237 and 93 on
2026-09-23, when the Addendum Z run log was banked (CORRECTIONS
2026-10-21). Nothing read the row against the map. This does: the total,
the four level counts and the rounded primary share are read from the
artifact, never typed here.
"""
from __future__ import annotations

import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
MAP = REPO / "experiments" / "verification_coverage.json"
ROW_START = "| **How far can a stranger actually re-derive each result?**"


def _row() -> str:
    rows = [line for line in (REPO / "VERDICT.md").read_text(encoding="utf-8").splitlines()
            if line.startswith(ROW_START)]
    assert len(rows) == 1, "the coverage row is missing or duplicated"
    return rows[0]


def test_the_coverage_row_quotes_the_map():
    record = json.loads(MAP.read_text(encoding="utf-8"))
    row = _row()
    total = re.search(r"classifies all (\d+) banked artifacts", row)
    assert total, "the row no longer states the total in a shape this test reads"
    assert int(total.group(1)) == record["total_artifacts"]
    levels = re.search(r"PRIMARY (\d+) · ARITHMETIC (\d+) · AGGREGATE (\d+) · EXTERNAL (\d+)", row)
    assert levels, "the row no longer states the four levels in a shape this test reads"
    quoted = dict(zip(("PRIMARY", "ARITHMETIC", "AGGREGATE", "EXTERNAL"),
                      map(int, levels.groups())))
    assert quoted == {k: record["counts"].get(k, 0) for k in quoted}, (quoted, record["counts"])
    assert sum(quoted.values()) == record["total_artifacts"]
    share = re.search(r"Only (\d+)% of what we have banked", row)
    assert share and int(share.group(1)) == round(100 * record["primary_verifiable_fraction"])
