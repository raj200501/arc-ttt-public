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


@pytest.mark.skipif(
    not _HAS_COPY,
    reason="the copy this fixer syncs is internal and not in this cut")
def test_the_fixer_refuses_a_phrasing_it_cannot_update() -> None:
    """MUTATION TEST: silence on an escaped shape is the failure mode.

    This happened three times in one session across two scripts: copy was
    rewritten into a phrasing the fixer did not own, the fixer reported
    success, and the number went stale inside a document written to be
    read by a stranger. Adding a pattern per miss fixes the sentence and
    loses the next one; refusing to be silent does not.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_ledger", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    original = module.APPLICATION.read_text(encoding="utf-8")
    try:
        module.APPLICATION.write_text(
            original + "\n\nthe ledger carries 7 dated entries.\n",
            encoding="utf-8")
        with pytest.raises(SystemExit) as caught:
            module.sync_copy(65, 46)
    finally:
        module.APPLICATION.write_text(original, encoding="utf-8")
    assert "does not understand" in str(caught.value)
    assert "7 dated entries" in str(caught.value)


@pytest.mark.skipif(
    not _HAS_COPY,
    reason="the copy this fixer syncs is internal and not in this cut")
def test_the_refusal_does_not_fire_on_current_copy() -> None:
    """A check that cries wolf trains the reader to ignore it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_ledger2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entries = module.rows()
    total = len(entries)
    outward = sum(1 for e in entries if e["outward_facing"])
    module.sync_copy(total, outward)   # must not raise
