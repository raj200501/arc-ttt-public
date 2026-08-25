"""The H reader must not be reachable by accident, and must refuse partials.

Written with the reader itself, before the arms landed. What is pinned:

1. A partial sweep returns INCOMPLETE, not a verdict on the cells that
   happen to exist. Stopping a sweep at a flattering moment is the failure
   mode a preregistered bar cannot prevent on its own.
2. The saturation guard fires BEFORE (b) and (c). A ceiling effect that
   looked like a refutation would be the most expensive misreading
   available, and (u) exists to make it unbankable.
3. Each preregistered reading is reachable from the numbers that should
   reach it, and only those.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import addendum_h_summary as H  # noqa: E402


def _arm(delta, baseline, wins, losses, ties=0, missing=()):
    return {
        "arm": "test",
        "seeds_present": [s for s in (1, 2, 3) if s not in missing],
        "seeds_missing": list(missing),
        "seed_mean_delta": delta,
        "seed_mean_baseline": baseline,
        "seed_mean_adapted": baseline + delta,
        "pooled_sign_test": {
            "wins": wins, "losses": losses, "ties": ties,
            "p_value": 0.0001 if wins > losses * 3 else 0.5},
        "baseline_saturated": baseline >= H.SATURATION,
    }


def test_a_partial_sweep_returns_no_verdict():
    verdict, why = H.reading(_arm(0.40, 0.55, 55, 2, missing=(3,)),
                             _arm(0.42, 0.55, 56, 1))
    assert verdict == "INCOMPLETE"
    assert "flattering moment" in why


def test_saturation_guard_fires_before_the_negative_readings():
    """A ceiling must read UNINFORMATIVE, never (c)."""
    # Delta is ZERO and the baseline is saturated: (c) would be the
    # flattering-to-nobody misreading, and it must not be returned.
    verdict, why = H.reading(_arm(0.0, 0.97, 0, 0, ties=60),
                             _arm(0.42, 0.55, 56, 1))
    assert verdict.startswith("(u)")
    assert "NOT reading (c)" in why


def test_saturation_guard_also_blocks_a_flattering_pass():
    """It binds in both directions -- a saturated baseline cannot PASS."""
    verdict, _ = H.reading(_arm(0.30, 0.96, 50, 1),
                           _arm(0.42, 0.55, 56, 1))
    assert verdict.startswith("(u)")


def test_reading_a_requires_both_statistics():
    survives, _ = H.reading(_arm(0.30, 0.55, 50, 1), _arm(0.42, 0.55, 56, 1))
    assert survives.startswith("(a)")
    # Same delta, sign test disagreeing -> not (a).
    other, _ = H.reading(_arm(0.30, 0.55, 30, 28), _arm(0.42, 0.55, 56, 1))
    assert not other.startswith("(a)")


def test_reading_b_and_c_are_reachable_and_distinct():
    b, why_b = H.reading(_arm(0.02, 0.55, 30, 20), _arm(0.42, 0.55, 56, 1))
    assert b.startswith("(b)")
    assert "+0.4200" in why_b, "reading (b) must carry the control's number"
    c, why_c = H.reading(_arm(-0.01, 0.55, 10, 40), _arm(0.42, 0.55, 56, 1))
    assert c.startswith("(c)")
    assert "those words" in why_c


def test_reader_runs_on_an_empty_experiments_set(tmp_path):
    out = tmp_path / "h.json"
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "addendum_h_summary.py"),
         "--out", str(out)],
        capture_output=True, text=True, cwd=str(REPO), check=False)
    assert result.returncode == 0, result.stderr
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["rule"]["bar_delta"] == 0.05
    assert record["rule"]["saturation_guard"] == 0.95
