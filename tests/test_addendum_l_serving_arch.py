"""Addendum L is a COST claim, which is the kind this repo has broken most.

Three of the corrections in `CORRECTIONS.md` are cost claims: a
cross-corpus conflation committed three times, a serving-mode conflation
that stapled one arm's quality to another arm's payload, and a
payload-asymmetry argument withdrawn when the assumption under it was
finally measured. Every one of them shared a shape: **a cost was
reported without the quality that bought it, or beside a quality
measured on something else.**

So the tests here are not about arithmetic. They are structural: they
fail if the artifact ever lets a cost be read without its quality, if
any arm's quality comes from anywhere but this run, or if the batching
scope is dropped.
"""

from __future__ import annotations

import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
ARTIFACT = REPO / "experiments" / "waybill_serving_arch_2026-08-22.json"

pytestmark = pytest.mark.skipif(
    not ARTIFACT.exists(), reason="Addendum L has not been run yet")


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_every_arm_reports_quality_beside_its_cost(record: dict) -> None:
    """The defect class three cost corrections share. No arm may carry a
    price without the score that price bought."""
    for arm in record["arms"]:
        assert "cost_per_1k_documents_usd" in arm, arm["arm"]
        assert "mean_micro_f1" in arm, (
            f"{arm['arm']} reports a cost with no quality beside it — the "
            "exact shape of three corrections already in CORRECTIONS.md")
        assert 0.0 <= arm["mean_micro_f1"] <= 1.0
        assert arm["cost_per_1k_documents_usd"] > 0


def test_the_reference_arm_reproduces_the_banked_score(record: dict) -> None:
    """fp32/batch1 with the retrained adapter must land on the number the
    banked greedy arm published, or the adapter is not the same recipe and
    nothing downstream of it means anything."""
    reference = record["arms"][0]
    assert reference["batch_size"] == 1
    assert reference["mean_micro_f1"] == pytest.approx(0.8833, abs=0.01), (
        "the retrained adapter does not reproduce the banked 0.8833; every "
        "comparison in this artifact is against a reference that is not the "
        "published arm")


def test_the_adapter_is_real_and_retained(record: dict) -> None:
    """The previous cost artifact had to time an UNTRAINED LoRA because the
    rehearsal adapter was not kept, and had to correct itself for the drift.
    This pins that it cannot happen silently again."""
    adapter = record["adapter"]
    assert adapter["tensors_loaded"] > 0
    assert "untrained" not in json.dumps(adapter).lower().replace(
        "untrained lora because", "")


def test_both_latency_numbers_are_published(record: dict) -> None:
    """Batching cuts cost and RAISES per-document latency. Publishing only
    the number that flatters the claim is how a trade gets hidden."""
    for arm in record["arms"]:
        assert "seconds_per_document_amortised" in arm
        assert "seconds_per_batch_latency" in arm
    batched = [a for a in record["arms"] if a["batch_size"] > 1]
    assert batched, "no batched arm ran; the claim rests on batching"
    for arm in batched:
        assert arm["seconds_per_batch_latency"] >= arm[
            "seconds_per_document_amortised"], (
            "a batched arm cannot have per-batch latency below its "
            "amortised per-document cost")


def test_the_batching_scope_is_stated(record: dict) -> None:
    scope = record["batching_scope"].lower()
    assert "concurrent" in scope
    assert "latency" in scope


def test_no_quality_claim_against_the_hosted_tier(record: dict) -> None:
    """Cheaper is not better. Addendum I and J are untouched by anything
    here, and the artifact has to say so itself."""
    disclaimer = record["what_this_does_not_claim"].lower()
    assert "quality" in disclaimer
    hosted = record["comparator_hosted_arm"]
    assert hosted["mean_micro_f1"] > record["arms"][0]["mean_micro_f1"], (
        "the hosted comparator must still outscore us — if it does not, "
        "this artifact is describing a different comparison than Addendum J")


def test_the_verdict_matches_the_frozen_bar(record: dict) -> None:
    """Recompute the reading from the arms rather than trusting the string."""
    bar = record["bar"]
    reference = record["arms"][0]
    clearing = [a for a in record["arms"]
                if a["cost_per_1k_documents_usd"] < bar["cost_per_1k_usd"]]
    verdict = record["verdict"]
    if not clearing:
        assert verdict.startswith("(c)")
        return
    best = min(clearing, key=lambda a: a["cost_per_1k_documents_usd"])
    within = abs(best["mean_micro_f1"] - reference["mean_micro_f1"]) <= bar[
        "quality_tolerance_vs_fp32_reference"]
    assert verdict.startswith("(a)" if within else "(b)"), (
        f"verdict {verdict!r} does not follow from the arms: cheapest "
        f"clearing arm scores {best['mean_micro_f1']} against reference "
        f"{reference['mean_micro_f1']}")


def test_the_instance_rate_is_labelled_as_a_quote(record: dict) -> None:
    rate = record["instance_rate"]
    assert "not a measurement" in rate["source"].lower()
    assert rate["quoted"]
