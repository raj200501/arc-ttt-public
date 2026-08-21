"""The published rehearsal material must reproduce the published result.

A result nobody can re-score is a claim, not evidence. So the corpus,
the withheld gold and both arms' raw outputs ship in the repository, and
these tests hold three things true:

  1. the gold file published now is BYTE-IDENTICAL to the one whose
     sha256 was hash-committed before the submission was made — which is
     the entire provenance claim of a blind run, and the one thing a
     reader cannot check without the file;
  2. re-scoring the raw outputs reproduces the numbers on the page;
  3. the field-audit viewer regenerates from those same artifacts, so
     the demo is a view of the evidence rather than a drawing of it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "experiments" / "blind_rehearsal_2026-08-20_raw"
REHEARSAL = REPO / "experiments" / "blind_rehearsal_2026-08-20.json"
PAIRED = REPO / "experiments" / "blind_rehearsal_baseline_2026-08-21.json"

pytestmark = pytest.mark.skipif(
    not RAW.is_dir(), reason="rehearsal raw material not in this tree")


def test_published_gold_matches_the_hash_committed_before_submission() -> None:
    """The provenance claim of the whole blind run, made checkable.

    The rehearsal recorded a sha256 of its gold file BEFORE the
    submission existed. If the file published later does not hash to
    that value, the blindness claim is worthless — so this is the one
    test whose failure would invalidate the row rather than annoy us.
    """
    committed = json.loads(REHEARSAL.read_text())["setup"]["challenger_gold_sha256"]
    published = hashlib.sha256(
        (RAW / "gold_holdout.jsonl").read_bytes()).hexdigest()
    assert published == committed, (
        "the published gold is NOT the file that was hash-committed before "
        f"submission\n  committed: {committed}\n  published: {published}")


def test_holdout_carries_no_gold() -> None:
    """What we received must not contain what we were not supposed to see."""
    for line in (RAW / "holdout.jsonl").read_text().splitlines():
        if line.strip():
            assert set(json.loads(line)) == {"id", "text"}, (
                "holdout.jsonl carries a field beyond id/text")


def test_train_and_holdout_are_disjoint() -> None:
    def ids(name: str) -> set[str]:
        return {json.loads(line)["id"]
                for line in (RAW / name).read_text().splitlines() if line.strip()}

    assert not ids("train.jsonl") & ids("holdout.jsonl")
    assert ids("gold_holdout.jsonl") == ids("holdout.jsonl")


@pytest.mark.parametrize("arm,expected", [
    ("predictions_prompted_greedy.jsonl", 0.7836),
    ("predictions_adapted_greedy.jsonl", 0.8833),
])
def test_rescoring_the_raw_outputs_reproduces_the_published_means(
        arm: str, expected: float) -> None:
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "make_challenge.py"), "score",
         "--pred", str(RAW / arm), "--gold", str(RAW / "gold_holdout.jsonl")],
        capture_output=True, text=True, cwd=REPO, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    line = [l for l in result.stdout.splitlines() if "mean per-document" in l]
    assert line, result.stdout[-1000:]
    value = float(line[0].split(":")[1].strip().split()[0])
    assert abs(value - expected) < 5e-4, (
        f"{arm} re-scores to {value}, the repo publishes {expected}")


def test_the_paired_verdict_is_still_a_fail() -> None:
    """Guard the direction, not just the digits.

    A future change that quietly turned this row into a pass would be
    the single most damaging edit available in this repository.
    """
    record = json.loads(PAIRED.read_text())
    assert record["verdict"] == "FAIL"
    assert record["why"]["clears_mean_bar"] is True
    assert record["why"]["sign_test_agrees"] is False


def test_the_field_audit_page_regenerates_from_the_artifacts() -> None:
    """The demo must be a VIEW of the evidence, not a drawing of it."""
    page = REPO / "demo" / "waybill_field_audit.html"
    if not page.exists():
        pytest.skip("field-audit page not in this tree")
    marker = "const DOCS = "
    text = page.read_text(encoding="utf-8")
    assert marker in text, "the page lost its generated-data marker"
    data = json.loads(text[text.index(marker) + len(marker):
                           text.index(";\n", text.index(marker))])
    scores = json.loads(PAIRED.read_text())["per_doc"]
    assert len(data) == len(scores)
    for row in data:
        assert row["sp"] == scores[row["id"]]["baseline"]
        assert row["sa"] == scores[row["id"]]["adapted"]
    # the losses and ties must still be in the page a reader is shown
    assert sum(1 for r in data if r["sa"] < r["sp"]) == 5
    assert sum(1 for r in data if r["sa"] == r["sp"]) == 17
