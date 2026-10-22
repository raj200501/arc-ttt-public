"""Counting your own corrections is only evidence if the count can be wrong.

`CORRECTIONS.md` opens by saying self-correction is evidence only when it
is countable. The counter that makes it countable is itself a claim about
this project, cited in outbound copy, and so it gets the same treatment as
every other gate here: prove it can fail.

The specific failure mode is one-directional and obvious. A parser that
quietly under-counts makes the project look worse; a parser that quietly
over-counts, or that silently returns zero and gets ignored, makes it look
better. Only the second kind is dangerous, so it is the one pinned.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "self_correction_ledger.py"
LEDGER = REPO / "CORRECTIONS.md"

pytestmark = pytest.mark.skipif(
    not LEDGER.exists(), reason="CORRECTIONS.md is not in this cut")


def _load():
    """Load the counter module once, so the skip conditions below can ask
    it where its copy lives instead of naming an internal document."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_ledger_mod", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_LEDGER = _load() if SCRIPT.exists() else None
_HAS_COPY = bool(_LEDGER and _LEDGER.APPLICATION.exists())


def _run(out: pathlib.Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out), *extra],
        capture_output=True, text=True, cwd=REPO, timeout=120)


@pytest.fixture()
def ledger(tmp_path: pathlib.Path) -> dict:
    out = tmp_path / "ledger.json"
    result = _run(out)
    assert result.returncode == 0, result.stderr
    return json.loads(out.read_text(encoding="utf-8"))


def test_the_count_matches_a_hand_count_of_dated_rows(ledger: dict) -> None:
    """Re-derive it independently rather than trusting the parser."""
    import re
    hand = 0
    for line in LEDGER.read_text(encoding="utf-8").split("\n"):
        if not line.startswith("|"):
            continue
        cells = line.split("|")
        if len(cells) < 3:
            continue
        if re.match(r"^\*{0,2}\d{4}-\d{2}-\d{2}", cells[1].strip()):
            hand += 1
    assert ledger["total_dated_corrections"] == hand


def test_it_refuses_rather_than_reporting_zero(tmp_path: pathlib.Path) -> None:
    """MUTATION TEST: a ledger that reports zero corrections is the most
    flattering possible failure, so it must be an error, not a result."""
    original = LEDGER.read_text(encoding="utf-8")
    stripped = "\n".join(line for line in original.split("\n")
                         if not line.startswith("|"))
    out = tmp_path / "ledger.json"
    # A COPY, never the tracked page: this test used to rewrite the real
    # CORRECTIONS.md and restore it in a finally, and two suite runs in
    # one tree raced -- a killed run left the page with every row gone.
    copy = tmp_path / "CORRECTIONS.md"
    copy.write_text(stripped, encoding="utf-8")
    result = _run(out, "--ledger", str(copy))
    assert result.returncode != 0, (
        "the counter reported success on a page with no correction rows — "
        "a silent zero here reads as 'nothing to correct'")


def test_the_outward_facing_subset_is_smaller_than_the_whole(
        ledger: dict) -> None:
    """The cited subset must be a real narrowing, not a relabelling.

    'Corrections that already went out' is the figure worth citing, and a
    classifier that marked everything outward-facing would inflate it
    while looking rigorous.
    """
    assert 0 < ledger["outward_facing"] < ledger["total_dated_corrections"]


def test_the_page_refuses_to_sell_the_count(ledger: dict) -> None:
    """A high correction count is not self-evidently good, and the
    artifact must say so where anyone quoting it will see it."""
    caveat = ledger["what_this_does_not_show"].lower()
    assert "not self-evidently good" in caveat
    assert "still standing" in caveat, (
        "the artifact must name what the count is silent about — the "
        "errors that have not been found")


def _copies_in(module, tmp_path: pathlib.Path, monkeypatch) -> dict:
    """Point the fixer at COPIES of the synced pages, never the tracked
    ones. sync_copy writes every page before it refuses, and the test
    below used to restore only the application: each run left the slide
    page saying "65 dated corrections ... 46 of them reached", two
    made-up figures in outbound copy (found 2026-10-22)."""
    copies = {}
    for page in module.SYNCED_COPIES:
        if page.exists():
            copy = tmp_path / page.name
            copy.write_text(page.read_text(encoding="utf-8"), encoding="utf-8")
            copies[page.name] = copy
    monkeypatch.setattr(module, "SYNCED_COPIES", tuple(copies.values()))
    return copies


@pytest.mark.skipif(
    not _HAS_COPY,
    reason="the copy this fixer syncs is internal and not in this cut")
def test_the_fixer_refuses_a_phrasing_it_cannot_update(
        tmp_path: pathlib.Path, monkeypatch) -> None:
    """MUTATION TEST: silence on an escaped shape is the failure mode.

    This happened three times in one session across two scripts: copy was
    rewritten into a phrasing the fixer did not own, the fixer reported
    success, and the number went stale inside a document written to be
    read by a stranger. Adding a pattern per miss fixes the sentence and
    loses the next one; refusing to be silent does not.
    """
    module = _load()
    tracked = {p: p.read_text(encoding="utf-8")
               for p in module.SYNCED_COPIES if p.exists()}
    application = _copies_in(module, tmp_path, monkeypatch)[module.APPLICATION.name]
    application.write_text(
        application.read_text(encoding="utf-8")
        + "\n\nthe ledger carries 7 dated entries.\n", encoding="utf-8")
    try:
        module.sync_copy(65, 46)
    except SystemExit as refusal:
        message = str(refusal)
    else:
        pytest.fail("the fixer reported success on a shape it cannot update")
    assert "does not understand" in message
    assert "7 dated entries" in message
    assert all(p.read_text(encoding="utf-8") == text
               for p, text in tracked.items()), "a tracked page was written"


@pytest.mark.skipif(
    not _HAS_COPY,
    reason="the copy this fixer syncs is internal and not in this cut")
def test_the_refusal_does_not_fire_on_current_copy(
        tmp_path: pathlib.Path, monkeypatch) -> None:
    """A check that cries wolf trains the reader to ignore it."""
    module = _load()
    tracked = {p: p.read_text(encoding="utf-8")
               for p in module.SYNCED_COPIES if p.exists()}
    _copies_in(module, tmp_path, monkeypatch)
    entries = module.rows()
    total = len(entries)
    outward = sum(1 for e in entries if e["outward_facing"])
    module.sync_copy(total, outward)   # must not refuse
    assert all(p.read_text(encoding="utf-8") == text
               for p, text in tracked.items()), "a tracked page was written"


def test_sent_and_press_are_matched_as_whole_words() -> None:
    """As substrings, "sent" matched "sentence" and "presentation" and
    "press" matched "suppress", and four rows counted as outward-facing
    for that alone (CORRECTIONS 2026-10-21)."""
    module = _load()
    assert not module.is_outward_facing(
        "a sentence in a presentation, to suppress a fence")
    assert module.is_outward_facing("the email body was sent on monday")
    assert module.is_outward_facing("a press release")
    assert module.is_outward_facing("the verdict row said so")


def test_public_is_matched_as_a_whole_word() -> None:
    """As a substring, "public" matched "publication", and three rows
    counted as outward-facing for that alone (CORRECTIONS 2026-10-22).
    One of them corrected a private memo; its only hit was "an
    adversarial check of the publication"."""
    module = _load()
    assert not module.is_outward_facing("an adversarial check of the publication")
    assert not module.is_outward_facing("found while staging the publication text")
    assert module.is_outward_facing("the public page said 237")
    assert module.is_outward_facing("pushed to the public repo")
    assert module.is_outward_facing("the terms did not exist publicly")
    assert module.is_outward_facing("experiments/public_export_counts.json")


def test_every_marker_is_matched_as_a_whole_word() -> None:
    """The same substring defect was fixed for two markers and then found
    again in a third. Pin the rule for every marker, including any added
    later: a letter touching either end means it is a different word."""
    module = _load()
    for marker in module.OUTWARD_MARKERS:
        assert module.is_outward_facing(f"see the {marker} row"), marker
        assert module.is_outward_facing(f"see the {marker}s row"), marker
        assert not module.is_outward_facing(f"see the x{marker} row"), marker
        assert not module.is_outward_facing(f"see the {marker}er row"), marker
        assert not module.is_outward_facing(f"see the {marker}ation row"), marker
    assert not module.is_outward_facing("a decker of the sentences")
    assert module.is_outward_facing("slide 7 of deck_v2.md")
    assert module.is_outward_facing("offered in the emails")
    assert module.is_outward_facing("`scripts/reconcile_outbound.py` passed it")


def test_the_outward_count_matches_a_hand_count(ledger: dict) -> None:
    """Re-derive the outward subset independently of the parser: split
    each row into words (letters, joined by hyphens) and their hyphen
    parts, and look the words up instead of searching with a pattern."""
    import re
    markers = ("email", "deck", "one-pager", "outbound", "sent", "public",
               "readme", "headline", "verdict", "press")
    forms = {m + suffix for m in markers for suffix in ("", "s", "ly")}
    hand = 0
    for line in LEDGER.read_text(encoding="utf-8").split("\n"):
        cells = line.split("|")
        if not line.startswith("|") or len(cells) < 3:
            continue
        if not re.match(r"^\*{0,2}\d{4}-\d{2}-\d{2}", cells[1].strip()):
            continue
        text = " ".join(c.strip() for c in cells[1:-1]).lower()
        words = set()
        for word in re.findall(r"[a-z]+(?:-[a-z]+)*", text):
            words.add(word)
            words.update(word.split("-"))
        if words & forms:
            hand += 1
    assert ledger["outward_facing"] == hand


def test_email_and_headline_keep_their_verb_endings() -> None:
    """Whole-word matching must not drop rows that did reach someone."""
    module = _load()
    assert module.is_outward_facing("the figure was emailed to two people")
    assert module.is_outward_facing("emailing the corrected table")
    assert module.is_outward_facing("the number headlined the page")
    assert not module.is_outward_facing("an emailer and a headliner")
