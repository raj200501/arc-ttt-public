#!/usr/bin/env python3
"""Count the corrections, so "we correct ourselves" stops being an adjective.

`CORRECTIONS.md` opens by saying self-correction is only evidence if it is
**countable**, and then never counts. Every outbound document that has
cited this page has cited it qualitatively -- "a dated corrections
ledger" -- which is exactly the shape of claim this repository refuses to
accept from anyone else.

So this counts it: dated rows, by section, with the subset that moved a
number this project had already published in an outbound document
separated out, because that subset is the only part that costs anything.
A correction to a typo is housekeeping. A correction that withdraws a
headline you have already sent to a stranger is the asset.

The count is banked as an artifact so any figure quoted from it can be
reconciled by `scripts/reconcile_outbound.py` like every other number
here, rather than being a hand-count that drifts the moment a row lands.

    PYTHONPATH=src python3 scripts/self_correction_ledger.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
LEDGER = REPO / "CORRECTIONS.md"

# A row counts only if its first cell opens with an ISO date. Header rows,
# alignment rows and prose tables are excluded by construction rather than
# by a keyword list, so a new section cannot silently inflate the count.
DATED = re.compile(r"^\*{0,2}(\d{4}-\d{2}-\d{2})")
SEPARATOR = re.compile(r"^\|[\s:\-|]+\|$")

# Rows whose correction reached, or would have reached, someone outside
# this repository. Detected from the row's own text rather than declared,
# so the classifier can be checked against the page by eye.
OUTWARD_MARKERS = ("email", "deck", "one-pager", "outbound", "sent",
                   "public", "readme", "headline", "verdict", "press")


def rows() -> list[dict]:
    out: list[dict] = []
    section = "(before any heading)"
    for line in LEDGER.read_text(encoding="utf-8").split("\n"):
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        if not line.startswith("|") or SEPARATOR.match(line):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if not cells:
            continue
        match = DATED.match(cells[0])
        if not match:
            continue
        text = " ".join(cells).lower()
        out.append({
            "date": match.group(1),
            "section": section,
            "outward_facing": any(m in text for m in OUTWARD_MARKERS),
            "claim": cells[1][:200] if len(cells) > 1 else "",
        })
    return out


APPLICATION = REPO / "docs" / "strategy" / "APPLICATION_DRAFT.md"

# Copy that quotes the count. Group 1 is the number; everything else is
# matched so the surrounding sentence is preserved exactly.
#
# This exists for the same reason `sync_test_counts.py` does. The test
# count drifted six times and every hand-sync missed a phrasing, so the
# repository stopped hand-syncing it. A correction count moves strictly
# more often than a test count -- it moves every time this project finds
# a mistake, which is the one thing it does reliably -- so leaving the
# copy to be updated by hand would reproduce a solved problem.
COPY_PATTERNS = (
    (re.compile(r"(\d+)(\s+dated corrections)"), "total"),
    (re.compile(r"(?<=carrying )(\d+)(\s*\n?>?\s*entries)"), "total"),
    (re.compile(r"(\d+)(\s+of them to something already)"), "outward"),
    (re.compile(r"(\d+)(\s+of which corrected something)"), "outward"),
)


# Any number sitting next to the word "correction(s)". Never rewritten --
# only used to refuse silence when a phrasing escapes COPY_PATTERNS.
#
# This is the same fix, for the same reason, as SUSPECT in
# sync_test_counts.py, and it is here because the identical defect
# happened a THIRD time within one session: copy was rewritten as "the
# ledger carries 59 dated entries", COPY_PATTERNS owns "dated
# corrections" and "carrying N entries", neither matched, and the number
# silently went stale while the sync reported success. Adding a fourth
# pattern would fix that sentence and lose the next one.
SUSPECT_COPY = re.compile(
    r"(?<![\d.])(\d{1,4})(?:\s+[a-z]+)?\s+(?:corrections?|dated entries)\b")


def sync_copy(total: int, outward: int) -> list[str]:
    """Rewrite quoted counts in outbound copy. Returns what changed.

    Raises if the copy contains a correction count in a shape this does
    not understand: a fixer that silently skips the sentence it was
    written for is worse than no fixer, because it reports success.
    """
    changed = []
    if not APPLICATION.exists():
        return changed
    text = APPLICATION.read_text(encoding="utf-8")
    original = text
    for pattern, which in COPY_PATTERNS:
        want = str(total if which == "total" else outward)

        def replace(match: re.Match, want: str = want,
                    which: str = which) -> str:
            if match.group(1) != want:
                changed.append(f"{which}: {match.group(1)} -> {want}")
            return want + match.group(2)

        text = pattern.sub(replace, text)
    if text != original:
        APPLICATION.write_text(text, encoding="utf-8")

    escaped = [" ".join(m.group(0).split())
               for m in SUSPECT_COPY.finditer(text)
               if int(m.group(1)) not in (total, outward)]
    if escaped:
        raise SystemExit(
            "a correction count in the copy is in a shape this fixer does "
            "not understand, so it was NOT updated and is now stale:\n  "
            + "\n  ".join(escaped)
            + f"\n\nCurrent: {total} dated corrections, {outward} "
              "outward-facing. Rephrase to a shape COPY_PATTERNS owns, or "
              "add the shape deliberately.")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(
        REPO / "experiments" / "self_correction_ledger.json"))
    parser.add_argument("--sync-copy", action="store_true",
                        help="rewrite counts quoted in outbound copy to "
                             "match this run, the way sync_test_counts.py "
                             "does for the test count")
    args = parser.parse_args()

    entries = rows()
    if not entries:
        raise SystemExit(
            "CORRECTIONS.md parsed to zero dated rows. Either the page has "
            "been restructured or this parser is broken, and a ledger that "
            "silently reports zero corrections is the most flattering "
            "possible failure -- refusing.")

    by_section: dict[str, int] = {}
    by_month: dict[str, int] = {}
    for entry in entries:
        by_section[entry["section"]] = by_section.get(entry["section"], 0) + 1
        month = entry["date"][:7]
        by_month[month] = by_month.get(month, 0) + 1
    outward = sum(1 for e in entries if e["outward_facing"])
    dates = sorted(e["date"] for e in entries)

    record = {
        "what": "Every dated row in CORRECTIONS.md, counted. The page says "
                "self-correction is only evidence if it is countable; this "
                "is the count.",
        "total_dated_corrections": len(entries),
        "outward_facing": outward,
        "outward_facing_note": (
            "Rows whose text refers to an email, deck, one-pager, README, "
            "public page, headline or VERDICT row -- corrections that "
            "reached, or would have reached, someone outside this "
            "repository. A typo fixed in a private note is housekeeping; "
            "withdrawing a number already sent to a stranger is the part "
            "that costs something, and it is the only part worth citing."),
        "by_section": by_section,
        "by_month": by_month,
        "first": dates[0],
        "latest": dates[-1],
        "how_counted": "A row counts only if its first cell opens with an "
                       "ISO date. Header rows, alignment rows and prose "
                       "tables are excluded by construction, so adding a "
                       "section cannot inflate the count.",
        "what_this_does_not_show": (
            "A high count is not self-evidently good and must not be sold "
            "as though it were. It is consistent with a careful instrument "
            "and equally consistent with a project that published too fast "
            "the first time. The defensible reading is narrower: these are "
            "errors that were found HERE, by this process, and published "
            "with dates, rather than found by a reader later. It says "
            "nothing about the errors still standing."),
        "source": "CORRECTIONS.md",
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")
    if args.sync_copy:
        for line in sync_copy(len(entries), outward) or ["copy already current"]:
            print(f"  copy: {line}")

    print(f"{len(entries)} dated corrections "
          f"({outward} outward-facing), {dates[0]} to {dates[-1]}")
    for section, count in sorted(by_section.items(), key=lambda kv: -kv[1]):
        print(f"  {count:3d}  {section}")
    print(f"\nbanked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
