"""The B.8 readout is claim-bearing, so its refusals and labels are pinned."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "novel_schema_rep_summary.py"
DATE = "2026-08-12"


def _arm(d: Path, seed: int, arm: str, f1: float, n: int = 10, dtype="torch.bfloat16") -> None:
    rec = {
        "rung": "0.5b", "k": 10, "seed": seed, "arm": arm,
        "device": "cuda", "dtype": dtype, "mean_micro_f1": f1,
        "results": [{"index": i, "micro_f1": f1} for i in range(n)],
    }
    (d / f"novel_schema_0.5b_k10_seed{seed}_{arm}_{DATE}.json").write_text(json.dumps(rec))


def _run(d: Path) -> dict:
    subprocess.run(
        [sys.executable, str(SCRIPT), "--dir", str(d), "--date", DATE],
        check=True, capture_output=True,
    )
    return json.loads((d / f"novel_schema_rep_summary_{DATE}.json").read_text())


def test_flipped_tenant_is_named_not_averaged_away(tmp_path: Path) -> None:
    for seed in (1, 2, 3, 4):
        _arm(tmp_path, seed, "adapted", 0.9)
        _arm(tmp_path, seed, "kshot", 0.5)
    _arm(tmp_path, 5, "adapted", 0.4)  # tenant 5 flips negative
    _arm(tmp_path, 5, "kshot", 0.6)
    report = _run(tmp_path)
    assert report["tenants_negative"] == [5]
    assert report["pooled"]["n_tenants"] == 5
    assert "NOT the gate" in report["ROLE"]


def test_heterogeneous_pair_refused_from_pool(tmp_path: Path) -> None:
    _arm(tmp_path, 1, "adapted", 0.9)
    _arm(tmp_path, 1, "kshot", 0.5)
    _arm(tmp_path, 2, "adapted", 0.9, dtype="torch.float16")  # mismatched pair
    _arm(tmp_path, 2, "kshot", 0.5, dtype="torch.bfloat16")
    report = _run(tmp_path)
    assert report["refused_heterogeneous"] == [2]
    assert report["pooled"]["n_tenants"] == 1


def test_incomplete_pair_simply_absent(tmp_path: Path) -> None:
    _arm(tmp_path, 1, "adapted", 0.9)
    _arm(tmp_path, 1, "kshot", 0.5)
    _arm(tmp_path, 2, "adapted", 0.9)  # kshot missing
    report = _run(tmp_path)
    assert [t["tenant_seed"] for t in report["tenants"]] == [1]
