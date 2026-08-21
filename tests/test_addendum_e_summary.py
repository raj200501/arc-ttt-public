"""The Addendum E verdict is computed, never argued — so it is pinned.

Same discipline as the Addendum B verdict test: every branch a motivated
reader could argue about after the numbers landed (an outlier-carried
mean, a partial wave, a validity trip, a seed outside the frozen set) is
decided here in advance, by a test rather than by whoever is reading.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "addendum_e_summary.py"
DATE = "2026-08-19"
FROZEN_SEEDS = (203, 204, 206, 207, 208, 209)


def _arm(directory: Path, seed: int, arm: str, per_receipt: list[float]) -> None:
    record = {
        "rung": "0.5b",
        "k": 30,
        "seed": seed,
        "arm": arm,
        "device": "cpu",
        "dtype": "float32",
        "eval_n": len(per_receipt),
        "mean_micro_f1": round(sum(per_receipt) / len(per_receipt), 4),
        "results": [{"index": i, "micro_f1": v} for i, v in enumerate(per_receipt)],
    }
    name = f"novel_schema_e_0.5b_k30_seed{seed}_{arm}_{DATE}.json"
    (directory / name).write_text(json.dumps(record))


def _run(directory: Path) -> dict:
    out = directory / "summary.json"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--dir", str(directory),
         "--date", DATE, "--out", str(out)],
        check=True, capture_output=True,
    )
    return json.loads(out.read_text())


def test_broad_effect_over_the_six_frozen_seeds_passes(tmp_path: Path) -> None:
    for seed in FROZEN_SEEDS:
        _arm(tmp_path, seed, "adapted", [0.80] * 60)
        _arm(tmp_path, seed, "kshot", [0.70] * 60)
    report = _run(tmp_path)
    assert report["verdict"] == "PASS"
    assert report["pairs_complete"] == 6
    assert report["receipt_level"]["sign_test"]["wins"] == 360


def test_outlier_carried_mean_is_a_fail_not_a_pass(tmp_path: Path) -> None:
    """A mean that clears +5 on a handful of receipts must not pass.

    The sign test is the guard: 30W/330L cannot be a PASS however
    flattering the mean looks.
    """

    adapted = [0.50] * 55 + [1.0] * 5
    kshot = [0.52] * 55 + [0.0] * 5
    for seed in FROZEN_SEEDS:
        _arm(tmp_path, seed, "adapted", adapted)
        _arm(tmp_path, seed, "kshot", kshot)
    report = _run(tmp_path)
    assert report["mean_delta"] >= 0.05  # the mean alone would pass
    assert report["verdict"] == "FAIL"


def test_partial_wave_is_undecidable_never_extrapolated(tmp_path: Path) -> None:
    for seed in FROZEN_SEEDS[:4]:
        _arm(tmp_path, seed, "adapted", [0.80] * 60)
        _arm(tmp_path, seed, "kshot", [0.70] * 60)
    report = _run(tmp_path)
    assert report["verdict"] == "UNDECIDABLE"
    assert report["missing_seeds"] == [208, 209]


def test_saturated_baseline_is_uninformative_not_a_fail(tmp_path: Path) -> None:
    """A k-shot arm above the 0.95 ceiling means the task was not
    measurable at this rung — the branch E.4 already exercised once."""

    for seed in FROZEN_SEEDS:
        _arm(tmp_path, seed, "adapted", [0.99] * 60)
        _arm(tmp_path, seed, "kshot", [0.97] * 60)
    report = _run(tmp_path)
    assert report["verdict"] == "UNINFORMATIVE"


def test_seeds_outside_the_frozen_set_cannot_enter_the_verdict(
    tmp_path: Path,
) -> None:
    """E-r2 froze {203,204,206,207,208,209}; 201/202/205 were excluded by
    the token screen. A late arm on an excluded seed must not be pooled,
    even if it would help."""

    for seed in FROZEN_SEEDS:
        _arm(tmp_path, seed, "adapted", [0.80] * 60)
        _arm(tmp_path, seed, "kshot", [0.70] * 60)
    for seed in (201, 202, 205):
        _arm(tmp_path, seed, "adapted", [1.00] * 60)
        _arm(tmp_path, seed, "kshot", [0.10] * 60)
    report = _run(tmp_path)
    assert report["pairs_complete"] == 6
    assert [r["seed"] for r in report["per_seed"]] == list(FROZEN_SEEDS)


def test_pair_split_across_dtype_is_refused_not_averaged(tmp_path: Path) -> None:
    for seed in FROZEN_SEEDS:
        _arm(tmp_path, seed, "adapted", [0.80] * 60)
        _arm(tmp_path, seed, "kshot", [0.70] * 60)
    path = tmp_path / f"novel_schema_e_0.5b_k30_seed203_adapted_{DATE}.json"
    record = json.loads(path.read_text())
    record["dtype"] = "bfloat16"
    path.write_text(json.dumps(record))
    report = _run(tmp_path)
    assert report["heterogeneous_pairs_refused"][0]["seed"] == 203
    assert report["verdict"] == "UNDECIDABLE"


def test_published_verdict_matches_the_banked_arms(tmp_path: Path) -> None:
    """The live check: recompute from the real artifacts and pin the
    published headline. If the banked arms and VERDICT.md ever disagree,
    this fails first."""

    report = _run(REPO / "experiments")
    assert report["verdict"] == "PASS"
    assert report["pairs_complete"] == 6
    assert round(report["mean_delta"], 4) == 0.4035
    assert report["receipt_level"]["sign_test"]["wins"] == 340
    assert report["receipt_level"]["sign_test"]["losses"] == 5
    assert sum(a["excluded_no_completion"] for a in report["attrition"]) == 0
