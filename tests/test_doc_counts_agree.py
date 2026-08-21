"""The test count printed in the docs must equal the test count that exists.

This drifted five times (83 -> 128 -> 137 -> 144 -> 151 -> 167), was
re-synced by hand on 2026-08-21 with a `CORRECTIONS.md` row saying so,
and drifted AGAIN inside the same day: the docs said 167 while the suite
collected 170, because the commit that re-synced them also added tests.

Two outside readers found it independently by running `pytest -q` -- the
one command a claim like ours invites. Their note was the right one: it
is a trivial number, and it is the exact class of drift the corrections
page claims to have closed, and nothing pinned it. A hand-sync is not a
fix; a hand-sync is the thing that already failed twice. So the count is
pinned here, where it fails instead of drifting.

Deliberately narrow: only phrases that are unambiguously about THIS test
suite are matched. Prose like "167 attempted tasks" in the paper is a
different 167 and is none of this test's business.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Phrasings that mean "the size of the offline test suite". Each pattern
# must capture the number in group 1.
CLAIM_PATTERNS = (
    re.compile(r"(\d+)\s+offline tests"),
    re.compile(r"(\d+)\s+tests? green"),
    re.compile(r"Test suite:\s*(\d+)\s+green"),
    re.compile(r"(\d+)\s+offline tests green"),
)

# Documents a reader is invited to check. Every markdown file under
# docs/ is swept rather than listed, so a new document cannot quietly
# opt out of the check by not being on a list -- and so this file names
# no internal document paths, which the public-export leak gate rejects.
# Scratch notes and dated postmortems describe a past state and are not
# resynced, so they are excluded by name below.
ROOT_DOCS = ("README.md", "EVIDENCE.md", "ROADMAP.md", "paper/DRAFT.md")
SKIP_SUBSTRINGS = ("AUDIT_RESPONSE", "MONDAY_BRIEF", "OVERNIGHT",
                   "snapshots_", "POSTMORTEM")


def _is_superseded(text: str) -> bool:
    """A document that declares itself superseded describes a past state.

    Opting out has to be visible IN the document, not hidden in this
    test's skip list -- otherwise a stale claim gets exempted by a name
    nobody reading the document can see. A file that says SUPERSEDED at
    the top has told its own readers not to trust its numbers.
    """
    return "SUPERSEDED" in text[:600]


def documents() -> list[Path]:
    paths = [REPO / name for name in ROOT_DOCS]
    docs_dir = REPO / "docs"
    if docs_dir.is_dir():
        paths += sorted(docs_dir.rglob("*.md"))
    return [p for p in paths
            if p.exists()
            and not any(s in p.name for s in SKIP_SUBSTRINGS)
            and not _is_superseded(p.read_text(encoding="utf-8"))]


def collected_test_count() -> int:
    """Ask pytest, in a subprocess, how many tests it collects."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--collect-only",
         "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=REPO, timeout=600,
    )
    match = re.search(r"(\d+)\s+tests? collected", result.stdout)
    assert match, f"could not read a collection count:\n{result.stdout[-2000:]}"
    return int(match.group(1))


def test_every_documented_test_count_is_the_real_one() -> None:
    actual = collected_test_count()
    wrong: list[str] = []
    for path in documents():
        name = path.relative_to(REPO)
        text = path.read_text(encoding="utf-8")
        for pattern in CLAIM_PATTERNS:
            for match in pattern.finditer(text):
                claimed = int(match.group(1))
                if claimed != actual:
                    line = text[:match.start()].count("\n") + 1
                    wrong.append(
                        f"{name}:{line} claims {claimed} tests, "
                        f"the suite collects {actual}")
    assert not wrong, (
        "documented test counts have drifted from the suite:\n  "
        + "\n  ".join(wrong)
        + f"\n\nUpdate them to {actual}. This test exists because "
          "hand-syncing this number has failed twice.")


def test_the_pin_can_actually_fail() -> None:
    """A guard that never fires is decoration -- prove the matcher bites."""
    sample = "the harness ships 999 offline tests, no downloads"
    found = [int(m.group(1)) for p in CLAIM_PATTERNS for m in p.finditer(sample)]
    assert found == [999], f"the claim matcher no longer matches: {found}"
