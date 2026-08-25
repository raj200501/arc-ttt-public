"""The test count printed in the docs must equal the test count that exists.

This drifted six times (83 -> 128 -> 137 -> 144 -> 151 -> 167 -> ...),
twice within hours of being "re-synced by hand", and two outside readers
caught it independently by running `pytest -q` -- the one command a
repo whose whole pitch is checkability invites.

A hand-sync was never a fix; a hand-sync is the thing that already
failed. Worse, each one was written from the phrasings someone
remembered, so each left a hole: the first missed the hyphenated form
and left "167-test suite" standing in our presentation and
talking-point documents; the second missed a claim wrapped across a
line break.

So the match rule lives in ONE place -- `scripts/sync_test_counts.py`,
which both fixes and is imported here to check. A checker and a fixer
that re-derive the rule separately will disagree, and the disagreement
will always favour the stale number.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SYNC = REPO / "scripts" / "sync_test_counts.py"

_spec = importlib.util.spec_from_file_location("sync_test_counts", SYNC)
sync = importlib.util.module_from_spec(_spec)
sys.modules["sync_test_counts"] = sync
_spec.loader.exec_module(sync)


def test_every_documented_test_count_is_the_real_one() -> None:
    actual = sync.collected_test_count()
    wrong: list[str] = []
    for path in sync.documents():
        for line, claimed in sync.stale_claims(path, actual):
            wrong.append(f"{path.relative_to(REPO)}:{line} claims "
                         f"{claimed} tests, the suite collects {actual}")
    assert not wrong, (
        "documented test counts have drifted from the suite:\n  "
        + "\n  ".join(wrong)
        + "\n\nFix them with: python3 scripts/sync_test_counts.py\n"
          "Do NOT hand-edit them -- hand-syncing this number has failed "
          "twice, each time by missing a phrasing.")


def test_the_matcher_recognises_every_phrasing_the_docs_use() -> None:
    """A guard that never fires is decoration, and this one had holes.

    Each string below is a form that actually appeared in the docs while
    some earlier version of the check or the fixer passed straight over
    it. They are kept as a regression list.
    """
    samples = {
        "the harness ships 999 offline tests, no downloads": 999,
        "Test suite: 999 green, pinned by a test": 999,
        "999 tests green at HEAD": 999,
        "CI runs the 999-test suite on every public push": 999,
        "every number reconciles (scripts re-run, 999-tests, VERIFIED)": 999,
        # wrapped across a line, which one hand-written fixer missed
        "six-seed replication run in parallel, 999-test\n   suite,": 999,
    }
    for text, expected in samples.items():
        found = {int(m.group(1))
                 for pattern in sync.CLAIM_PATTERNS
                 for m in pattern.finditer(text)}
        assert found == {expected}, (
            f"the claim matcher no longer recognises {text!r} (found {found})")


def test_a_superseded_document_opts_out_visibly() -> None:
    """Opting out must be readable by a human reading the document."""
    assert sync.is_superseded("> **SUPERSEDED (2026-08-21)** do not send")
    assert not sync.is_superseded("a normal document about 999 offline tests")
    # ...and not by burying the word far below where anyone would see it
    assert not sync.is_superseded("x" * 700 + "SUPERSEDED")


def test_the_matcher_recognises_a_bare_count() -> None:
    """The third phrasing to escape the list, found 2026-08-25.

    "a public harness with 303\\n  tests, 179 banked artifacts" carries no
    "offline", no "green" and no hyphen, so every pattern passed over it
    while the fixer reported a clean run.
    """
    samples = ("a public harness with 999 tests, 179 banked artifacts",
               "a public harness with 999\n  tests, and a ledger",
               "the harness ships 999 tests.")
    for text in samples:
        found = {int(m.group(1))
                 for pattern in sync.CLAIM_PATTERNS
                 for m in pattern.finditer(text)}
        assert found == {999}, f"{text!r} still escapes the matcher: {found}"


def test_an_unknown_phrasing_is_reported_rather_than_passed_over() -> None:
    """The real fix is not another pattern; it is refusing to be silent.

    Three phrasings have escaped CLAIM_PATTERNS, each found only after it
    had already shipped. Enumerating the sentences nobody thought of is
    impossible by construction, so the fixer must report what it can see
    and cannot match instead of calling the run clean.
    """
    import tempfile
    from pathlib import Path as _Path
    with tempfile.TemporaryDirectory() as tmp:
        doc = _Path(tmp) / "note.md"
        # A ratio: the exact shape that left "83/83 tests" standing in
        # outbound copy for six drift cycles.
        doc.write_text("an MIT-licensed harness with 83/83 tests (public)",
                       encoding="utf-8")
        uncovered = sync.uncovered_claims(doc, 303)
    assert uncovered, (
        "a claim shape no pattern matches was passed over in silence — "
        "which is how this number drifted three times")
    assert "83" in uncovered[0][1]


def test_the_suspect_scan_does_not_cry_wolf() -> None:
    """A check that fires on prose trains the reader to ignore it.

    Its first, wider version produced ten hits over the tree, nine of
    them about test *items* rather than the suite. A gate nobody reads is
    the same as no gate.
    """
    import tempfile
    from pathlib import Path as _Path
    noise = ("scored on 100 test receipts\n"
             "10 ranked attempts per test\n"
             "held out 128 for the test split\n"
             "22 scored test documents\n")
    with tempfile.TemporaryDirectory() as tmp:
        doc = _Path(tmp) / "note.md"
        doc.write_text(noise, encoding="utf-8")
        assert sync.uncovered_claims(doc, 303) == []
