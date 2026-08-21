#!/usr/bin/env python3
"""Rewrite every documented test count to the count the suite really has.

    python3 scripts/sync_test_counts.py            # fix in place
    python3 scripts/sync_test_counts.py --check    # report, change nothing

This module owns the patterns; `tests/test_doc_counts_agree.py` imports
them, so the fixer and the checker cannot disagree about what a claim
looks like. That is the whole point of the file.

The number has now drifted six times (83 -> 128 -> 137 -> 144 -> 151 ->
167 -> ...), twice within hours of being "re-synced by hand", and two
outside readers caught it by running `pytest -q`, which is the one
command a repo like this invites. Each hand-sync was a `sed` written
from the phrasings someone remembered:

  * the first missed the hyphenated form entirely, so "167-test suite"
    survived in our presentation and talking-point documents -- the
    numbers said out loud in a room;
  * the second missed "192-test\\n   suite", wrapped across a line.

A person re-deriving the match rule each time will keep missing a form.
So there is one rule, in one place, used by both sides.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Each pattern captures the number in group 1. Whitespace between the
# count and its noun is \s+ so a claim wrapped across lines still counts.
CLAIM_PATTERNS = (
    re.compile(r"(\d+)\s+offline tests"),
    re.compile(r"(\d+)\s+tests?\s+green"),
    re.compile(r"Test suite:\s*(\d+)\s+green"),
    re.compile(r"(\d+)-tests?\b"),
    re.compile(r"(\d+)-test\s+suite"),
)

ROOT_DOCS = ("README.md", "EVIDENCE.md", "ROADMAP.md", "paper/DRAFT.md")
# Scratch notes and dated postmortems describe a past state.
SKIP_SUBSTRINGS = ("AUDIT_RESPONSE", "MONDAY_BRIEF", "OVERNIGHT",
                   "snapshots_", "POSTMORTEM")


def is_superseded(text: str) -> bool:
    """A document that declares itself superseded describes a past state.

    Opting out has to be visible IN the document, not hidden in a skip
    list -- otherwise a stale claim is exempted by a name nobody reading
    the document can see.
    """
    return "SUPERSEDED" in text[:600]


def documents() -> list[Path]:
    """Every doc a reader is invited to check.

    Swept rather than listed, so a new document cannot opt out of the
    check by not being on someone's list.
    """
    paths = [REPO / name for name in ROOT_DOCS]
    docs_dir = REPO / "docs"
    if docs_dir.is_dir():
        paths += sorted(docs_dir.rglob("*.md"))
    return [p for p in paths
            if p.exists()
            and not any(s in p.name for s in SKIP_SUBSTRINGS)
            and not is_superseded(p.read_text(encoding="utf-8"))]


def collected_test_count() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--collect-only",
         "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=REPO, timeout=900,
    )
    match = re.search(r"(\d+)\s+tests? collected", result.stdout)
    if not match:
        sys.exit(f"could not read a collection count:\n{result.stdout[-2000:]}")
    return int(match.group(1))


def stale_claims(path: Path, actual: int) -> list[tuple[int, int]]:
    """(line number, claimed count) for every claim that disagrees."""
    text = path.read_text(encoding="utf-8")
    found = []
    for pattern in CLAIM_PATTERNS:
        for match in pattern.finditer(text):
            claimed = int(match.group(1))
            if claimed != actual:
                found.append((text[:match.start()].count("\n") + 1, claimed))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="report drift and exit nonzero; change nothing")
    args = parser.parse_args()

    actual = collected_test_count()
    print(f"suite collects {actual} tests")
    drifted = 0

    for path in documents():
        stale = stale_claims(path, actual)
        if not stale:
            continue
        drifted += len(stale)
        name = path.relative_to(REPO)
        for line, claimed in stale:
            print(f"  {name}:{line} claims {claimed}")
        if args.check:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in CLAIM_PATTERNS:
            text = pattern.sub(
                lambda m: m.group(0).replace(m.group(1), str(actual), 1), text)
        path.write_text(text, encoding="utf-8")

    if not drifted:
        print("every documented count matches the suite")
        return 0
    if args.check:
        print(f"\n{drifted} stale claim(s). Run without --check to fix.")
        return 1
    print(f"\nrewrote {drifted} stale claim(s) to {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
