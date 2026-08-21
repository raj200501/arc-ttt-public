"""EVIDENCE.md is the one page most likely to be read without the repo
around it — so every figure on it is pinned to its artifact here.

The card claims "every number below is reconciled to a named artifact."
That sentence is only true if something enforces it. This is that
something: each assertion recomputes the value from the banked artifact
and requires the card to contain it verbatim. A number that drifts (or
a rounding that flatters) fails the suite before it reaches a reader.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CARD_PATH = REPO / "EVIDENCE.md"
EXP = REPO / "experiments"


@pytest.fixture(scope="module")
def card() -> str:
    return CARD_PATH.read_text(encoding="utf-8")


def _load(name: str) -> dict:
    return json.loads((EXP / name).read_text())


def _pts(x: float) -> float:
    """F1 points, as the card writes them (0.4035 -> 40.35)."""
    return x * 100


def test_gate_b_headline_matches_artifact(card: str) -> None:
    gate = _load("novel_schema_summary_2026-08-12.json")["gate_k30"]
    assert f"+{_pts(gate['mean_delta']):.1f} F1" in card
    for delta in gate["seed_deltas"]:
        assert f"+{_pts(delta):.1f}" in card
    sign = gate["receipt_level"]["sign_test"]
    assert (f"{sign['wins']}W/{sign['losses']}L/{sign['ties']}T" in card
            or f"{sign['wins']}W/{sign['losses']}L/{sign['ties']}T"
            .replace("/", " / ") in card)
    assert f"{gate['receipt_level']['n']} scored pairs" in card
    excluded = sum(a["excluded_no_completion"] for a in gate["attrition"])
    assert str(excluded) in card


def test_gate_e_headline_matches_artifact(card: str) -> None:
    e = _load("novel_schema_e_summary_2026-08-20.json")
    assert e["verdict"] == "PASS"
    assert f"+{_pts(e['mean_delta']):.1f} F1" in card
    for delta in e["seed_deltas"]:
        assert f"+{_pts(delta):.1f}" in card
    lo, hi = e["cluster_level"]["ci95"]
    # written to the artifact's own precision — no rounding that flatters
    assert f"[+{_pts(lo):.2f}, +{_pts(hi):.2f}]" in card
    rlo, rhi = e["receipt_level"]["ci95"]
    assert f"[+{_pts(rlo):.1f}, +{_pts(rhi):.1f}]" in card
    sign = e["receipt_level"]["sign_test"]
    assert f"{sign['wins']}W/{sign['losses']}L/{sign['ties']}T" in card
    assert str(e["receipt_level"]["n"]) in card
    assert sum(a["excluded_no_completion"] for a in e["attrition"]) == 0
    assert "zero documents excluded" in card


def test_gate_f_absolutes_and_the_weak_seed_are_both_shown(card: str) -> None:
    """The 0.53 tenant must appear beside the 0.94 one — the card is not
    allowed to show only the flattering absolute."""

    values = [
        _load(f"novel_schema_f_0.5b_k30_seed{s}_docadapted_2026-08-19.json")
        ["mean_micro_f1"]
        for s in (1, 2, 3)
    ]
    for value in values:
        assert f"{value:.4f}" in card
    assert min(values) < 0.6  # the weak seed is real...
    assert "0.53" in card     # ...and called out in prose, not just the table


def test_addendum_d_failure_is_on_the_page(card: str) -> None:
    for seed in (1, 2, 3):
        record = _load(
            f"novel_schema_d_0.5b_k30_seed{seed}_doconly_2026-08-18.json")
        assert record["mean_micro_f1"] == 0.0
    assert "0.0000 F1" in card
    assert "0/60 valid JSON" in card
    assert "FAIL" in card


def test_rehearsal_figures_and_their_label(card: str) -> None:
    r = _load("blind_rehearsal_2026-08-20.json")
    score = r["score"]
    assert f"{score['mean_micro_f1']:.4f}" in card
    assert f"{score['n_holdout']}/{score['n_holdout']} valid JSON" in card
    for tier, value in score["by_challenger_tier"].items():
        assert f"{value:.3f}" in card, tier
    # the label is load-bearing: the number may never appear unqualified
    assert "dress rehearsal" in card
    assert "agent-authored" in card
    assert "not a real tenant" in card


def test_cost_table_matches_the_banked_rows(card: str) -> None:
    cheap = _load("novel_cheaptier_baseline_2026-08-19.json")
    assert cheap["pooled_mean_micro_f1"] == 1.0
    assert "1.00 by plain prompting" in card
    greedy = _load("novel_greedy_quality_2026-08-19.json")
    assert f"{greedy['mean_micro_f1']:.4f}" in card
    for figure in ("~$1.09", "~$5.13", "~$1.03", "~$0.22"):
        assert figure in card


def test_every_disclosure_that_makes_the_numbers_honest_is_present(
    card: str,
) -> None:
    """The caveats are not decoration; without them the page overclaims."""

    required = [
        "synthetic",              # what the corpora are
        "FAIL",                   # the failures are shown
        "Zero customers",         # the ledger admits the gap
        "0.5B",                   # the scale everything was measured at
        "GPU crossover",          # the open cost question
        "OPEN",                   # the unexplained seed
        "1.67",                   # the ARC score, disclosed
    ]
    for phrase in required:
        assert phrase in card, f"missing disclosure: {phrase}"


def test_card_does_not_cite_a_missing_artifact(card: str) -> None:
    """Every experiments/ path named on the page must exist."""

    import re

    for match in re.findall(r"experiments/[A-Za-z0-9_.*-]+", card):
        if "*" in match:  # glob form, e.g. novel_schema_f_*.json
            pattern = Path(match).name
            assert list(EXP.glob(pattern)), f"no artifact matches {match}"
        else:
            assert (REPO / match).exists(), f"missing artifact: {match}"


def test_referenced_scripts_exist(card: str) -> None:
    import re

    for match in re.findall(r"scripts/[a-z_]+\.py", card):
        assert (REPO / match).exists(), f"card names a missing script: {match}"
