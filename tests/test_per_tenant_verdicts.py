"""Per-tenant readings must stay honest and must not drift from artifacts.

Pinned here: the one tenant that fails the frozen rule on its own is
actually reported as failing. A version of this script that quietly passed
everything would be worse than not having it.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "per_tenant_verdicts.py"
BANKED = REPO / "experiments" / "per_tenant_verdicts_2026-08-22.json"


def _run(tmp_path: pathlib.Path) -> dict:
    out = tmp_path / "ptv.json"
    result = subprocess.run([sys.executable, str(SCRIPT), "--out", str(out)],
                            capture_output=True, text=True, cwd=str(REPO),
                            check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(out.read_text(encoding="utf-8"))


def test_the_failing_tenant_is_reported_as_failing(tmp_path):
    record = _run(tmp_path)
    assert record["tenants_failing_individually"] == [
        "gate4_addendum_F_document_mode:2"]
    assert record["tenants_passing_individually"] == "11/12"


def test_gate4_seed2_numbers_are_the_ones_we_publish(tmp_path):
    record = _run(tmp_path)
    gate = next(g for g in record["gates"] if g["gate"].startswith("gate4"))
    seed2 = next(t for t in gate["tenants"] if t["tenant"] == "2")
    assert seed2["per_tenant_verdict"] == "FAIL"
    assert seed2["sign_test"]["wins"] == 18
    assert seed2["sign_test"]["losses"] == 15
    assert seed2["sign_test"]["ties"] == 5
    assert seed2["clears_bar"] is True      # the mean clears
    assert seed2["sign_test_agrees"] is False   # the sign test does not


def test_record_does_not_claim_to_restate_a_gate(tmp_path):
    record = _run(tmp_path)
    assert "does NOT restate any gate" in record["status"]


def test_banked_record_matches_a_fresh_run(tmp_path):
    assert _run(tmp_path) == json.loads(BANKED.read_text(encoding="utf-8"))
