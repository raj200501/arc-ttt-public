"""G-E2 decision-record guards (ENTERPRISE_EVAL_SPEC Addendum A).

The summary script produces the single number the product wedge rests on.
These tests pin the two ways it must refuse to produce one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "cord_scale_summary.py"
DATE = "2026-08-11"


def _arm(directory: Path, seed: int, arm: str, f1: float, device: str | None = None) -> None:
    record: dict[str, object] = {
        "rung": "4b", "k": 10, "seed": seed, "arm": arm,
        "mean_micro_f1": f1, "invalid_json": 0, "scored": 20,
    }
    if device is not None:  # kernel-produced arms carry a device field
        record["device"] = device
    (directory / f"cord_scale_4b_k10_seed{seed}_{arm}_{DATE}.json").write_text(
        json.dumps(record)
    )


def _run(directory: Path) -> dict:
    subprocess.run(
        [sys.executable, str(SCRIPT), "--dir", str(directory), "--date", DATE],
        check=True, capture_output=True,
    )
    return json.loads((directory / f"cord_scale_summary_{DATE}.json").read_text())


def test_refuses_to_pair_across_environments(tmp_path: Path) -> None:
    """A mixed-environment pair must never enter the G-E2 mean.

    All three seeds here carry a large positive delta, so a script that
    paired blindly would report ~+12.7 F1 and declare G-E2 PASSED - a
    fabricated 'measured mechanism' built on a dtype/library mismatch.
    """

    _arm(tmp_path, 1, "adapted", 0.80, "cuda")
    _arm(tmp_path, 1, "kshot", 0.68, "cuda")
    _arm(tmp_path, 2, "adapted", 0.82)  # local: no device field
    _arm(tmp_path, 2, "kshot", 0.68, "cuda")  # kernel
    _arm(tmp_path, 3, "adapted", 0.79, "cuda")
    _arm(tmp_path, 3, "kshot", 0.67, "cuda")

    summary = _run(tmp_path)
    decision = summary["decisions"]["4b"]
    assert "g_e2_pass" not in decision, "a contaminated pair reached the verdict"
    assert decision["blocked_by_mixed_environment"] == [2]
    mixed = [e for e in summary["incomplete"] if e.get("status") == "mixed_environment"]
    assert len(mixed) == 1 and mixed[0]["seed"] == 2


def test_homogeneous_pairs_decide_normally(tmp_path: Path) -> None:
    """With one environment throughout, the gate evaluates and can pass."""

    for seed, (a, ks) in enumerate([(0.80, 0.68), (0.82, 0.70), (0.79, 0.67)], start=1):
        _arm(tmp_path, seed, "adapted", a, "cuda")
        _arm(tmp_path, seed, "kshot", ks, "cuda")

    decision = _run(tmp_path)["decisions"]["4b"]
    assert decision["g_e2_pass"] is True
    assert decision["k10_mean_delta"] == 0.12


def test_non_arm_incident_records_are_listed_not_parsed(tmp_path: Path) -> None:
    """Hand-written incident records share the artifact directory and glob.

    cord_scale_4b_cpu_oom_<date>.json is a postmortem, not an arm: it has no
    rung/k/seed/arm. Before this guard the script died with KeyError('rung')
    and produced no decision record at all - a real crash on 08-11.
    """

    for seed, (a, ks) in enumerate([(0.80, 0.68), (0.82, 0.70), (0.79, 0.67)], start=1):
        _arm(tmp_path, seed, "adapted", a, "cuda")
        _arm(tmp_path, seed, "kshot", ks, "cuda")
    (tmp_path / f"cord_scale_4b_cpu_oom_{DATE}.json").write_text(
        json.dumps({"outcome": "OOM - process SIGKILLed", "arms_completed": 0})
    )

    summary = _run(tmp_path)
    assert summary["non_arm_files"] == [f"cord_scale_4b_cpu_oom_{DATE}.json"]
    assert summary["decisions"]["4b"]["g_e2_pass"] is True  # arms still decide


def test_gate_fails_when_delta_below_five_points(tmp_path: Path) -> None:
    """The threshold is +5 F1; a smaller positive delta is not a pass."""

    for seed, (a, ks) in enumerate([(0.70, 0.68), (0.71, 0.70), (0.69, 0.67)], start=1):
        _arm(tmp_path, seed, "adapted", a, "cuda")
        _arm(tmp_path, seed, "kshot", ks, "cuda")

    decision = _run(tmp_path)["decisions"]["4b"]
    assert decision["g_e2_pass"] is False
    # deltas .02/.01/.02 -> mean .0167, a third of the +5-point bar
    assert decision["k10_mean_delta"] == 0.0167
