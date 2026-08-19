"""The verification scripts a cold reader runs are themselves verified.

verify_verdict.py and verify_from_primary.py are the two commands readers
are explicitly told to run. This pins both directions:

- happy path: run against the shipped experiments/ artifacts and pass;
- mutation: a perturbed copy (one receipt score, the published summary,
  one stored prediction, one gold-regeneration field) must FAIL loudly.

Mutations run only on copies in a temp tree — the real artifacts are never
touched.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
VERIFY_VERDICT = REPO / "scripts" / "verify_verdict.py"
VERIFY_PRIMARY = REPO / "scripts" / "verify_from_primary.py"
EXP = REPO / "experiments"
DATE = "2026-08-12"
ARM_FILES = [
    f"novel_schema_0.5b_k30_seed{seed}_{arm}_{DATE}.json"
    for seed in (1, 2, 3)
    for arm in ("adapted", "kshot")
]
SUMMARY_FILE = f"novel_schema_summary_{DATE}.json"
PRIMARY_FILES = sorted(EXP.glob("novel_schema_f_*_docadapted_2026-08-19.json"))


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


# -- happy path against the shipped artifacts -------------------------------


def test_verify_verdict_passes_on_shipped_artifacts() -> None:
    proc = _run(VERIFY_VERDICT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "GO" in proc.stdout
    assert "MATCHES" in proc.stdout
    assert "MISMATCH\n" not in proc.stdout


def test_verify_from_primary_passes_on_shipped_artifacts() -> None:
    assert PRIMARY_FILES, "shipped docadapted artifacts must exist"
    for path in PRIMARY_FILES:
        proc = _run(VERIFY_PRIMARY, str(path))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "PRIMARY-VERIFIED" in proc.stdout


# -- mutation: tampering must be detected (temp copies only) -----------------


def _verdict_tree(tmp_path: Path) -> Path:
    """Copy verify_verdict.py + its inputs into tmp, preserving layout
    (the script resolves ROOT from __file__, so a copied tree works)."""

    (tmp_path / "scripts").mkdir()
    (tmp_path / "experiments").mkdir()
    shutil.copy(VERIFY_VERDICT, tmp_path / "scripts" / "verify_verdict.py")
    for name in [*ARM_FILES, SUMMARY_FILE]:
        shutil.copy(EXP / name, tmp_path / "experiments" / name)
    return tmp_path / "scripts" / "verify_verdict.py"


def test_verify_verdict_detects_a_tampered_receipt_score(tmp_path: Path) -> None:
    script = _verdict_tree(tmp_path)
    target = tmp_path / "experiments" / ARM_FILES[0]  # seed1 adapted
    record = json.loads(target.read_text())
    row = next(r for r in record["results"] if "micro_f1" in r)
    # a value guaranteed to differ from the original (many receipts sit at
    # the 1.0 cap, so "+0.25 clamped" could be a no-op)
    row["micro_f1"] = 0.1234 if abs(row["micro_f1"] - 0.1234) > 0.01 else 0.4321
    target.write_text(json.dumps(record))

    proc = _run(script)
    assert proc.returncode != 0, proc.stdout + proc.stderr


def test_verify_verdict_detects_a_tampered_summary(tmp_path: Path) -> None:
    script = _verdict_tree(tmp_path)
    target = tmp_path / "experiments" / SUMMARY_FILE
    summary = json.loads(target.read_text())
    summary["gate_k30"]["mean_delta"] = round(
        summary["gate_k30"]["mean_delta"] + 0.05, 4
    )
    target.write_text(json.dumps(summary))

    proc = _run(script)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "MISMATCH" in proc.stdout


@pytest.mark.skipif(not PRIMARY_FILES, reason="no shipped docadapted artifacts")
def test_verify_from_primary_detects_a_tampered_prediction(tmp_path: Path) -> None:
    copy = tmp_path / PRIMARY_FILES[0].name
    record = json.loads(PRIMARY_FILES[0].read_text())
    row = next(
        r
        for r in record["results"]
        if r.get("prediction") and r.get("micro_f1", 0) > 0.2
    )
    row["prediction"] = '{"tampered": "prediction"}'
    copy.write_text(json.dumps(record))

    proc = _run(VERIFY_PRIMARY, str(copy))
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "MISMATCH" in proc.stdout


@pytest.mark.skipif(not PRIMARY_FILES, reason="no shipped docadapted artifacts")
def test_verify_from_primary_detects_tampered_regeneration_metadata(
    tmp_path: Path,
) -> None:
    # The gold labels are regenerated from the seed; a record whose seed was
    # altered can no longer reproduce its own stored schema.
    copy = tmp_path / PRIMARY_FILES[0].name
    record = json.loads(PRIMARY_FILES[0].read_text())
    record["seed"] = int(record["seed"]) + 40
    copy.write_text(json.dumps(record))

    proc = _run(VERIFY_PRIMARY, str(copy))
    assert proc.returncode != 0, proc.stdout + proc.stderr
