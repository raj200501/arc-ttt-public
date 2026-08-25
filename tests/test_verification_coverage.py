"""The coverage map must never flatter the evidence.

Its whole value is that it is stricter than the prose. A classifier that
can be talked into PRIMARY by an artifact's own adjectives would be worse
than no classifier, because it would launder aggregate results into
verifiable-looking ones -- which is precisely the impression this
repository has been corrected for creating before.
"""

from __future__ import annotations

import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
ARTIFACT = REPO / "experiments" / "verification_coverage.json"

pytestmark = pytest.mark.skipif(
    not ARTIFACT.exists(), reason="coverage map has not been generated")


@pytest.fixture(scope="module")
def coverage() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_every_artifact_is_classified(coverage: dict) -> None:
    levels = {"PRIMARY", "ARITHMETIC", "AGGREGATE", "EXTERNAL", "UNREADABLE"}
    assert coverage["artifacts"], "the map is empty"
    for row in coverage["artifacts"]:
        assert row["level"] in levels, row


def test_a_claim_of_primary_requires_stored_predictions(coverage: dict) -> None:
    """The upgrade path must be evidence, never wording.

    Read each PRIMARY artifact back off disk and require that it really
    does carry predictions. If this ever passes on an artifact that only
    *says* it is verifiable, the map is laundering.
    """
    for row in coverage["artifacts"]:
        if row["level"] != "PRIMARY":
            continue
        path = REPO / "experiments" / row["artifact"]
        blob = path.read_text(encoding="utf-8")
        assert '"prediction"' in blob or '"predictions"' in blob, (
            f"{row['artifact']} is classified PRIMARY but stores no "
            "predictions — the classifier is upgrading on wording")


def test_the_headline_is_not_claimed_as_primary(coverage: dict) -> None:
    """The +46.5 arms discarded their model outputs. No amount of later
    tooling makes them re-derivable, and the map must keep saying so."""
    headline = [row for row in coverage["artifacts"]
                if row["artifact"].startswith("novel_schema_summary")]
    assert headline, "the headline summary is missing from the map"
    for row in headline:
        assert row["level"] != "PRIMARY", (
            "the most-cited number in this repository must not be reported "
            "as primary-verifiable; its predictions were never stored")


def test_the_uncomfortable_finding_is_published(coverage: dict) -> None:
    """A coverage map that buried its worst row would be decoration."""
    text = coverage["the_uncomfortable_one"].lower()
    assert "46.5" in text
    assert "not done" in text, (
        "the outstanding fix must be named as outstanding")


def test_ambiguity_downgrades(coverage: dict) -> None:
    assert "downgrade" in coverage["downgrade_rule"].lower()
    # AGGREGATE-but-regenerable is a real distinction and must not be used
    # to quietly promote anything.
    note = coverage["aggregate_but_regenerable"]
    assert "does not upgrade" in note or "deliberately does not upgrade" in note
