"""The two verdict readers must not disagree about the same interval.

`verify_verdict.py` used the normal quantile (1.96) for the receipt-level
interval while using a t quantile for the cluster interval in the same
function — and `novel_schema_summary.py`, the authorized reader that
produced the banked artifact, used t throughout. The gap was 0.03 F1
points: comfortably inside the script's own 1e-3 cross-check tolerance,
so it never failed, and it put VERDICT.md's number (42.8, from the
artifact) at odds with the command VERDICT.md tells you to run (which
printed 42.9).

That is the exact failure the whole verification stack exists to
prevent: two authorities, one number, no alarm. These tests make any
future divergence loud.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARTIFACT = REPO / "experiments" / "novel_schema_summary_2026-08-12.json"


def _verify_verdict_output() -> str:
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "verify_verdict.py")],
        capture_output=True, text=True, cwd=REPO, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


def test_receipt_interval_matches_the_banked_artifact() -> None:
    stored = json.loads(ARTIFACT.read_text())["gate_k30"]["receipt_level"]["ci95"]
    printed = re.search(r"receipt-level 95% CI \[([\d.]+), ([\d.]+)\]",
                        _verify_verdict_output())
    assert printed, "verify_verdict.py no longer prints the receipt CI"
    for stored_value, printed_value in zip(stored, printed.groups()):
        assert round(stored_value * 100, 1) == float(printed_value), (
            f"reader disagreement: artifact {stored_value * 100:.4f} vs "
            f"printed {printed_value}")


def test_verdict_md_quotes_the_same_interval_the_script_prints() -> None:
    """The 'check it' column must not contradict the number beside it."""
    printed = re.search(r"receipt-level 95% CI \[([\d.]+), ([\d.]+)\]",
                        _verify_verdict_output())
    assert printed
    lo, hi = printed.groups()
    verdict = (REPO / "VERDICT.md").read_text(encoding="utf-8")
    assert f"receipt-level [{lo}, {hi}]" in verdict, (
        f"VERDICT.md does not carry the interval the script prints "
        f"([{lo}, {hi}])")


def test_both_readers_use_the_same_multiplier() -> None:
    """Guard the cause, not just the symptom."""
    source = (REPO / "scripts" / "verify_verdict.py").read_text()
    assert "1.96 * sd" not in source, (
        "verify_verdict.py is using the normal quantile for the receipt "
        "interval again; the authorized reader uses t")
