"""The reconciliation gate must not vouch for figures it cannot fail on.

Simulated review round 10 (2026-10-22) found that
`scripts/reconcile_outbound.py` passed most fabricated figures, because
some artifact somewhere holds the same number: every integer from 1 to
247 is in the evidence. These tests pin the tightening made that day and
the probe that measures what is left
(`experiments/reconcile_false_pass_2026-10-22.json`).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "reconcile_outbound.py"
PROBE = REPO / "scripts" / "reconcile_false_pass_probe.py"
BANKED = REPO / "experiments" / "reconcile_false_pass_2026-10-22.json"


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_small_count_is_uncheckable_not_reconciled():
    ro = _load(SCRIPT, "ro_r10a")
    numbers = [float(n) for n in range(1, 50)] + [77.0]
    assert ro._count_floor(numbers) == 50
    assert ro._verdict("counted", "12 design partners", "", numbers) == "uncheckable"
    assert ro._verdict("counted", "77 tests", "", numbers) == "reconciled"
    assert ro._verdict("counted", "78 tests", "", numbers) == "missing"
    unsure: list[str] = []
    reconciled, missing = ro._check("We have 12 design partners.", None, "",
                                    numbers, {}, uncheckable=unsure)
    assert (reconciled, missing, unsure) == ([], [], ["12 design partners"])


def test_n_of_m_is_extracted_and_both_sides_must_trace():
    ro = _load(SCRIPT, "ro_r10b")
    assert ("fraction", "6 of 35") in ro._claims("6 of 35 sites held up")
    numbers = [float(n) for n in range(1, 10)] + [35.0, 120.0, 300.0]
    assert ro._verdict("fraction", "120 of 300", "", numbers) == "reconciled"
    assert ro._verdict("fraction", "6 of 35", "", numbers) == "uncheckable"
    assert ro._verdict("fraction", "6 of 36", "", numbers) == "missing"


def test_a_cited_decimal_is_checked_against_its_own_artifact_only():
    """Before 2026-10-22 only counts took their clause's citation; a
    decimal next to a citation matched anything in the evidence."""
    ro = _load(SCRIPT, "ro_r10c")
    text = "F1 was 0.8123 (`experiments/project_timeline.json`)."
    claims = ro._cited_claims(text)
    assert claims == [("decimal", "0.8123", "project_timeline.json")]
    assert not ro._reconciles("decimal", "0.8123", "", [0.8123],
                              cited_numbers=[0.5])
    assert ro._reconciles("decimal", "0.8123", "", [0.5],
                          cited_numbers=[0.8123])


def test_a_whole_percent_does_not_match_inside_a_larger_one():
    ro = _load(SCRIPT, "ro_r10d")
    assert not ro._reconciles("percent", "5%", "rose 25% overall", [])
    assert ro._reconciles("percent", "5%", "rose 5% overall", [])


def test_a_sources_line_is_not_a_claims_own_citation():
    ro = _load(SCRIPT, "ro_r10e")
    text = "> 3 | runs pending, +46.5\n\nSources: `experiments/project_timeline.json`"
    assert all(cited is None for _, _, cited in ro._cited_claims(text))


# The banked rates were measured on the private tree, whose evidence holds
# artifacts the public export drops; a public cut measures different rates.
_PRIVATE = (REPO / "experiments" / "outbound_reconciliation.json").exists()


@pytest.mark.skipif(not _PRIVATE, reason="rates were measured on the private tree")
def test_the_probe_reproduces_its_banked_after_rates():
    """The banked measurement still describes this gate.

    The probe matches fabricated figures against the live evidence, which
    grows with every banked artifact, so exact counts drift by design: a
    new artifact can only add numbers a fabricated figure might match.
    What must hold is that the gate catches about as much as it did when
    the rates were banked (within 10 of 200 per shape) and that the
    tightening below still holds. Re-run the probe to re-bank the rates."""
    probe = _load(PROBE, "probe_r10")
    banked = json.loads(BANKED.read_text(encoding="utf-8"))
    gate = probe._load_gate(SCRIPT.read_text(encoding="utf-8"))
    now = probe.measure(gate)
    for key, tally in banked["after"]["shapes"].items():
        assert abs(now[key]["caught"] - tally["caught"]) <= 10, (key, now[key], tally)
    # the tightening: uncited counts are no longer vouched for wholesale,
    # and a cited figure of any shape is checked against its artifact
    before = banked["before"]["shapes"]
    assert now["integer_count"]["vouched_rate"] < before["integer_count"]["vouched_rate"]
    assert now["decimal_cited"]["vouched_rate"] <= before["decimal_cited"]["vouched_rate"]
    assert now["n_of_m_cited"]["caught"] > before["n_of_m_cited"]["caught"]
