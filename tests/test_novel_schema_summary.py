"""The Addendum B verdict is computed, never argued — so it is pinned.

Every branch a motivated reader might argue about after the numbers land
(promoting k=10, calling a floor-trip a fail, passing on a two-receipt
mean) is decided here, in advance, by a test.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "novel_schema_summary.py"
DATE = "2026-08-12"


def _arm(
    directory: Path,
    k: int,
    seed: int,
    arm: str,
    per_receipt: list[float],
) -> None:
    record = {
        "rung": "0.5b",
        "k": k,
        "seed": seed,
        "arm": arm,
        "device": "cpu",
        "mean_micro_f1": round(sum(per_receipt) / len(per_receipt), 4),
        "results": [{"index": i, "micro_f1": v} for i, v in enumerate(per_receipt)],
    }
    (directory / f"novel_schema_0.5b_k{k}_seed{seed}_{arm}_{DATE}.json").write_text(
        json.dumps(record)
    )


def _run(directory: Path) -> dict:
    subprocess.run(
        [sys.executable, str(SCRIPT), "--dir", str(directory), "--date", DATE],
        check=True,
        capture_output=True,
    )
    return json.loads((directory / f"novel_schema_summary_{DATE}.json").read_text())


def test_go_requires_all_three_statistics_to_agree(tmp_path: Path) -> None:
    """A broad, real +10 F1 effect at k=30 is a GO."""

    for seed in (1, 2, 3):
        _arm(tmp_path, 30, seed, "adapted", [0.80] * 60)
        _arm(tmp_path, 30, seed, "kshot", [0.70] * 60)
    report = _run(tmp_path)
    assert report["VERDICT"] == "GO"
    assert report["gate_k30"]["receipt_level"]["sign_test"]["wins"] == 180


def test_outlier_carried_mean_is_pivot_not_go(tmp_path: Path) -> None:
    """A +6.5 F1 mean carried by five receipts per seed must NOT pass.

    55 receipts lose slightly, five win enormously; the mean clears the +5
    bar but the sign test reads 15W/165L and refuses — the failure mode the
    4B k=5 arms demonstrated on CORD, scaled to this gate's n.
    """

    adapted = [0.50] * 55 + [1.0] * 5
    kshot = [0.52] * 55 + [0.0] * 5
    for seed in (1, 2, 3):
        _arm(tmp_path, 30, seed, "adapted", adapted)
        _arm(tmp_path, 30, seed, "kshot", kshot)
    report = _run(tmp_path)
    assert report["gate_k30"]["mean_delta"] >= 0.05  # the mean alone would pass
    assert report["VERDICT"] == "PIVOT"              # the verdict refuses


def test_floor_trip_is_uninformative_not_pivot(tmp_path: Path) -> None:
    """If the baseline cannot do the task, a zero delta means nothing."""

    for seed in (1, 2, 3):
        _arm(tmp_path, 30, seed, "adapted", [0.10] * 60)
        _arm(tmp_path, 30, seed, "kshot", [0.08] * 60)  # below 0.15 floor
    report = _run(tmp_path)
    assert report["VERDICT"] == "UNINFORMATIVE"
    assert "NOT a fail" in report["verdict_detail"]


def test_missing_pair_is_undecidable_never_extrapolated(tmp_path: Path) -> None:
    _arm(tmp_path, 30, 1, "adapted", [0.9] * 60)
    _arm(tmp_path, 30, 1, "kshot", [0.7] * 60)
    report = _run(tmp_path)
    assert report["VERDICT"] == "UNDECIDABLE"


def test_k10_positive_cannot_rescue_a_k30_fail(tmp_path: Path) -> None:
    """The 4B-k=5 trap, pre-committed against: garnish never becomes gate."""

    for seed in (1, 2, 3):
        _arm(tmp_path, 30, seed, "adapted", [0.70] * 60)  # k=30 flat
        _arm(tmp_path, 30, seed, "kshot", [0.70] * 60)
        _arm(tmp_path, 10, seed, "adapted", [0.85] * 60)  # k=10 glowing
        _arm(tmp_path, 10, seed, "kshot", [0.70] * 60)
    report = _run(tmp_path)
    assert report["VERDICT"] == "PIVOT"
    assert report["comparability_k10"]["mean_delta"] > 0.05  # tempting…
    assert "may not be quoted" in report["k10_promotion_bar"]  # …and barred


def test_pair_split_across_dtype_is_refused_not_averaged(tmp_path: Path) -> None:
    """B.7-r3 moved k=30 pairs to fp16; a stale bf16 kshot must not pair.

    The adapted arm below is fp16, the kshot arm bf16. Averaging them would
    launder a dtype difference into the gate delta, so the pair is refused
    and the verdict stays UNDECIDABLE rather than deciding on a corrupt pair.
    """

    for seed in (1, 2, 3):
        a = {
            "rung": "0.5b", "k": 30, "seed": seed, "arm": "adapted",
            "device": "cuda", "dtype": "torch.float16",
            "mean_micro_f1": 0.9,
            "results": [{"index": i, "micro_f1": 0.9} for i in range(60)],
        }
        b = {
            "rung": "0.5b", "k": 30, "seed": seed, "arm": "kshot",
            "device": "cuda", "dtype": "torch.bfloat16",
            "mean_micro_f1": 0.5,
            "results": [{"index": i, "micro_f1": 0.5} for i in range(60)],
        }
        (tmp_path / f"novel_schema_0.5b_k30_seed{seed}_adapted_{DATE}.json").write_text(json.dumps(a))
        (tmp_path / f"novel_schema_0.5b_k30_seed{seed}_kshot_{DATE}.json").write_text(json.dumps(b))
    report = _run(tmp_path)
    assert report["VERDICT"] == "UNDECIDABLE"
    assert len(report["gate_k30"]["heterogeneous_pairs_refused"]) == 3


def test_banking_supersedes_stale_dtype_and_error_but_refuses_same_env_conflicts(
    tmp_path: Path,
) -> None:
    """The 4am command must make only the decisions that are mechanical."""

    exp = tmp_path / "experiments"; exp.mkdir()
    pull = tmp_path / "pull"; pull.mkdir()

    def write(where: Path, seed: int, arm: str, **extra) -> None:
        rec = {
            "rung": "0.5b", "k": 30, "seed": seed, "arm": arm,
            "device": "cuda",
            "results": [{"index": 0, "micro_f1": extra.get("mean_micro_f1", 0.5)}],
        }
        rec.update(extra)
        (where / f"novel_schema_0.5b_k30_seed{seed}_{arm}_{DATE}.json").write_text(
            json.dumps(rec)
        )

    # seed1: existing error record, incoming scoreable -> supersede
    write(exp, 1, "adapted", error="oom")
    write(pull, 1, "adapted", dtype="torch.float16", mean_micro_f1=0.9)
    # seed2: existing bf16 kshot, incoming fp16 (B.7-r3 migration) -> supersede
    write(exp, 2, "kshot", dtype="torch.bfloat16", mean_micro_f1=0.5)
    write(pull, 2, "kshot", dtype="torch.float16", mean_micro_f1=0.52)
    # seed3: same env, DIFFERENT numbers -> refuse with exit code 2
    write(exp, 3, "kshot", dtype="torch.float16", mean_micro_f1=0.40)
    write(pull, 3, "kshot", dtype="torch.float16", mean_micro_f1=0.60)

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "bank_novel_schema.py"),
            "--pull-dir", str(pull),
            "--experiments", str(exp),
            "--date", DATE,
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr  # conflict refused
    # supersessions happened and preserved the originals
    sup = exp / "superseded_novel"
    assert (sup / f"novel_schema_0.5b_k30_seed1_adapted_{DATE}.json").exists()
    assert (sup / f"novel_schema_0.5b_k30_seed2_kshot_{DATE}.json").exists()
    banked1 = json.loads((exp / f"novel_schema_0.5b_k30_seed1_adapted_{DATE}.json").read_text())
    assert banked1["mean_micro_f1"] == 0.9
    # the conflicted file was NOT touched
    kept3 = json.loads((exp / f"novel_schema_0.5b_k30_seed3_kshot_{DATE}.json").read_text())
    assert kept3["mean_micro_f1"] == 0.40
