"""Addendum H decided. Pin the verdict and the magnitude that qualifies it.

The verdict is (a) SURVIVES, which is the outcome that flatters us. That
is exactly why the qualifying number needs a test: the ablation shows the
corpus's arbitrary label->key mapping was worth more than half the
measured delta, and a future edit that quotes +41 without +18.75 would be
true and misleading.
"""

from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
SUMMARY = REPO / "experiments" / "novel_schema_h_summary_2026-08-22.json"


def _load() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_the_verdict_is_the_preregistered_reading():
    record = _load()
    assert record["verdict"].startswith("(a)")
    assert record["mnemonic"]["seeds_missing"] == []
    assert record["control"]["seeds_missing"] == []


def test_the_saturation_guard_did_not_have_to_fire():
    """(u) would have made the sweep uninformative. It did not apply."""
    record = _load()
    assert record["mnemonic"]["baseline_saturated"] is False
    assert record["mnemonic"]["seed_mean_baseline"] < 0.95


def test_the_ablation_moved_the_baseline_as_designed():
    """Realistic labels must make prompting easier, or nothing was tested."""
    record = _load()
    assert (record["mnemonic"]["seed_mean_baseline"]
            > record["control"]["seed_mean_baseline"] + 0.15)


def test_the_qualifying_magnitude_is_recorded():
    """More than half the delta was the mapping. Do not lose this number."""
    record = _load()
    assert record["mapping_cost"] > 0.20
    assert (record["mapping_cost"]
            > record["mnemonic"]["seed_mean_delta"] * 0.9), (
        "the arbitrary mapping was worth more than the effect that "
        "survives it; any page quoting the control delta must carry the "
        "mnemonic one beside it")


def test_the_surviving_effect_clears_the_frozen_bar_on_both_statistics():
    record = _load()
    m = record["mnemonic"]
    assert m["seed_mean_delta"] >= 0.05
    st = m["pooled_sign_test"]
    assert st["p_value"] < 0.05 and st["wins"] > st["losses"]
