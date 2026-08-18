"""Guards on the G-E2 PRECISION report.

This script does not decide the gate - cord_scale_summary.py does - but it
decides whether the gate's verdict is worth anything, so its refusals and
its arithmetic are pinned the same way.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "cord_paired_power.py"
DATE = "2026-08-11"


def _arm(
    directory: Path,
    seed: int,
    arm: str,
    per_receipt: list[float],
    device: str | None = None,
    k: int = 10,
) -> None:
    record: dict = {
        "rung": "4b",
        "k": k,
        "seed": seed,
        "arm": arm,
        "mean_micro_f1": round(sum(per_receipt) / len(per_receipt), 4),
        "invalid_json": 0,
        "scored": len(per_receipt),
        "results": [
            {"index": i, "micro_f1": f1} for i, f1 in enumerate(per_receipt)
        ],
    }
    if device is not None:
        record["device"] = device
    (directory / f"cord_scale_4b_k{k}_seed{seed}_{arm}_{DATE}.json").write_text(
        json.dumps(record)
    )


def _run(directory: Path) -> dict:
    subprocess.run(
        [sys.executable, str(SCRIPT), "--dir", str(directory), "--date", DATE],
        check=True,
        capture_output=True,
    )
    return json.loads((directory / f"cord_paired_power_{DATE}.json").read_text())


def _row(summary: dict, k: int = 10) -> dict:
    return next(r for r in summary["rows"] if r["k"] == k)


def test_pairs_by_receipt_index_not_by_position(tmp_path: Path) -> None:
    """A dropped receipt must not shift the pairing by one.

    kshot here is missing index 1. Pairing by position would compare
    adapted[1] against kshot's index-2 score and report a fabricated delta;
    pairing by index compares only the three receipts both arms scored.
    """

    adapted: dict = {
        "rung": "4b", "k": 10, "seed": 1, "arm": "adapted", "device": "cuda",
        "mean_micro_f1": 0.5, "invalid_json": 0, "scored": 4,
        "results": [{"index": i, "micro_f1": v}
                    for i, v in enumerate([0.5, 0.9, 0.5, 0.5])],
    }
    kshot: dict = {
        "rung": "4b", "k": 10, "seed": 1, "arm": "kshot", "device": "cuda",
        "mean_micro_f1": 0.4, "invalid_json": 0, "scored": 3,
        "results": [{"index": 0, "micro_f1": 0.4},
                    {"index": 2, "micro_f1": 0.4},
                    {"index": 3, "micro_f1": 0.4},
                    {"index": 1, "error": "no completion"}],
    }
    (tmp_path / f"cord_scale_4b_k10_seed1_adapted_{DATE}.json").write_text(
        json.dumps(adapted)
    )
    (tmp_path / f"cord_scale_4b_k10_seed1_kshot_{DATE}.json").write_text(
        json.dumps(kshot)
    )

    receipt = _row(_run(tmp_path))["receipt_level"]
    assert receipt["n"] == 3, "index 1 was scored by only one arm"
    assert receipt["mean"] == 0.1  # (0.5-0.4) on each of the three shared


def test_refuses_to_pool_receipts_across_environments(tmp_path: Path) -> None:
    """Each pair may be homogeneous while the POOL still is not.

    Seed 1 is local/local and seed 3 is kernel/kernel - both are valid pairs
    that cord_scale_summary.py accepts. Pooling their receipt deltas into one
    confidence interval assumes the effect is environment-invariant, so the
    CI is marked non-authoritative and resolvability is withheld.
    """

    _arm(tmp_path, 1, "adapted", [0.8] * 20)
    _arm(tmp_path, 1, "kshot", [0.7] * 20)
    _arm(tmp_path, 3, "adapted", [0.8] * 20, "cuda")
    _arm(tmp_path, 3, "kshot", [0.7] * 20, "cuda")

    receipt = _row(_run(tmp_path))["receipt_level"]
    assert receipt["ci_is_authoritative"] is False
    assert receipt["pooled_across_environments"] == ["kernel", "local"]
    assert _row(_run(tmp_path))["power"]["gate_is_resolvable"] is None


def test_mixed_pair_is_refused_before_it_reaches_the_pool(tmp_path: Path) -> None:
    """A pair split across environments is dropped, exactly as in the gate."""

    _arm(tmp_path, 1, "adapted", [0.8] * 20, "cuda")
    _arm(tmp_path, 1, "kshot", [0.7] * 20, "cuda")
    _arm(tmp_path, 2, "adapted", [0.9] * 20)          # local
    _arm(tmp_path, 2, "kshot", [0.7] * 20, "cuda")    # kernel

    row = _row(_run(tmp_path))
    assert row["refused_mixed_environment"] == [2]
    assert row["receipt_level"]["n"] == 20  # seed 2 contributed nothing


def test_noiseless_effect_is_resolvable_and_noisy_one_is_not(tmp_path: Path) -> None:
    """The resolvability verdict tracks spread, not the point estimate.

    The tight rung (+10 F1 on every receipt) resolves the +5 bar. The noisy
    rung has a TRUE delta of zero but +/-80 F1 per-receipt swings, and its
    interval is far too wide to rule the bar in or out - so it is reported
    unresolvable rather than as a clean negative. Distinguishing "measured
    no effect" from "could not have measured one" is the whole reason this
    report exists.
    """

    for seed in (1, 2, 3):
        _arm(tmp_path, seed, "adapted", [0.80] * 20, "cuda", k=10)
        _arm(tmp_path, seed, "kshot", [0.70] * 20, "cuda", k=10)
    tight = _row(_run(tmp_path), k=10)
    assert tight["receipt_level"]["mean"] == 0.1
    assert tight["power"]["gate_is_resolvable"] is True

    noisy_a = [0.1, 0.9] * 10
    noisy_b = [0.9, 0.1] * 10  # same means, huge per-receipt swing
    for seed in (1, 2, 3):
        _arm(tmp_path, seed, "adapted", noisy_a, "cuda", k=5)
        _arm(tmp_path, seed, "kshot", noisy_b, "cuda", k=5)
    noisy = _row(_run(tmp_path), k=5)
    assert noisy["power"]["gate_is_resolvable"] is False
    assert noisy["receipt_level"]["ci95"][0] < 0 < noisy["receipt_level"]["ci95"][1]


def test_sign_test_and_jackknife_expose_an_outlier_driven_mean(
    tmp_path: Path,
) -> None:
    """A mean can pass while the receipts underneath it disagree.

    Nineteen receipts per seed lose by 3 F1 and one wins outright. The mean
    still comes out POSITIVE, because one receipt going 0.0 -> 1.0 outweighs
    nineteen small losses. The sign test says adaptation lost on 57 of 60
    receipts, and dropping the three winners flips the mean negative. A
    headline built on that mean would be describing three receipts.
    """

    per_receipt_adapted = [0.49] * 19 + [1.0]
    per_receipt_kshot = [0.52] * 19 + [0.0]
    for seed in (1, 2, 3):
        _arm(tmp_path, seed, "adapted", per_receipt_adapted, "cuda")
        _arm(tmp_path, seed, "kshot", per_receipt_kshot, "cuda")

    row = _row(_run(tmp_path))
    assert row["receipt_level"]["mean"] > 0  # the mean reads as a win
    sign = row["robustness"]["sign_test"]
    assert sign["wins"] == 3 and sign["losses"] == 57  # the receipts do not
    assert sign["p_value"] < 0.001
    # and the mean is standing on those three winners alone
    assert row["robustness"]["mean_less_top3_winners"] < 0


def test_report_carries_the_multiplicity_and_gate_warnings(tmp_path: Path) -> None:
    """The guardrails ship inside the artifact, not just in a session's head."""

    _arm(tmp_path, 1, "adapted", [0.8] * 20, "cuda", k=5)
    _arm(tmp_path, 1, "kshot", [0.7] * 20, "cuda", k=5)
    warn = _run(tmp_path)["READ_THIS_BEFORE_QUOTING_ANY_ROW"]
    assert "4b/k=5" in warn["exploratory_rows_in_this_report"]
    assert "cherry-pick" in warn["the_gate_is_k10_only"]
    assert "multiplicity" in warn
