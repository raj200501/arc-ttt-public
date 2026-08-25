"""The impact table must not overstate a pattern it has not measured.

Its headline sentence -- "demonstrations suppress the fence" -- is a claim
about two regimes. If only one regime is banked, the sentence is an
extrapolation from half a table, which is the shape of over-claim this
repository keeps correcting.
"""

from __future__ import annotations

import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
ARTIFACT = REPO / "experiments" / "fence_impact_table.json"

pytestmark = pytest.mark.skipif(
    not ARTIFACT.exists(), reason="the impact table has not been generated")


@pytest.fixture(scope="module")
def table() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_the_finding_is_pending_until_both_regimes_exist(table: dict) -> None:
    regimes = {row["regime"] for row in table["arms"]}
    stated = not table["the_finding"].startswith("PENDING")
    both = any(r == "schema" for r in regimes) and any(
        r.startswith("k-shot") for r in regimes)
    assert stated == both, (
        "the finding is stated over one regime, or withheld when both are "
        "present — either way the sentence and the data disagree")


def test_every_arm_reports_both_columns(table: dict) -> None:
    for row in table["arms"]:
        assert row["n"] > 0
        assert 0.0 <= row["fence_rate"] <= 1.0
        # The tax is the whole point and it must be derived, never typed.
        assert row["fence_tax"] == pytest.approx(
            row["mean_fence_stripped"] - row["mean_as_emitted"], abs=1e-4)


def test_an_arm_without_raw_text_is_named_not_dropped(table: dict) -> None:
    """An arm that cannot answer the question must say so.

    Silently omitting arms that banked parsed objects would make the table
    look like a complete sweep of everything that was run.
    """
    for row in table["not_measurable"]:
        assert row["why"], "an arm was excluded without a reason"


def test_the_scope_names_what_it_does_not_show(table: dict) -> None:
    text = table["what_it_does_not_show"].lower()
    assert "does not show any published benchmark number is wrong" in text
    assert "multiplied together" in text, (
        "the table must refuse the multiplication a reader will reach for — "
        "fence rate times exposed-package count is not a real quantity")
