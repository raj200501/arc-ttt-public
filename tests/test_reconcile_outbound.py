"""The outbound-copy gate, and proof it can actually fail.

`tests/test_verify_scripts.py` exists because a verifier that cannot fail
is decoration. The same applies here, and more sharply: this gate guards
the one document a stranger reads first, so if it silently passes
everything it is worse than nothing — it would license the copy.

So these tests mutate the copy and require the gate to catch it.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "reconcile_outbound.py"
PACKAGE = REPO / "docs" / "strategy" / "SEND_PACKAGE_2026-08-20.md"

pytestmark = pytest.mark.skipif(
    not PACKAGE.exists(),
    reason="outbound package is not in this cut (it is internal)")


def _run(out: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out)],
        capture_output=True, text=True, cwd=REPO, timeout=300)


def test_the_live_copy_reconciles(tmp_path: pathlib.Path) -> None:
    """Every distinctive figure in every body traces to an artifact."""
    out = tmp_path / "reconciliation.json"
    result = _run(out)
    record = json.loads(out.read_text(encoding="utf-8"))
    assert result.returncode == 0, (
        "outbound copy carries a figure that traces to no artifact:\n"
        + json.dumps(record["bodies"], indent=2))
    assert record["unreconciled_total"] == 0


def test_the_gate_checks_a_meaningful_number_of_claims(
        tmp_path: pathlib.Path) -> None:
    """A gate that finds nothing to check passes vacuously.

    If a rewrite ever strips the copy of checkable figures, or the
    section parser silently stops matching, this fails loudly instead of
    reporting a clean run over an empty set.
    """
    out = tmp_path / "reconciliation.json"
    _run(out)
    record = json.loads(out.read_text(encoding="utf-8"))
    assert set(record["bodies"]) == {"SHORT", "TECHNICAL", "GENERALIST"}, (
        "a body stopped parsing; the gate would pass it vacuously")
    for name, block in record["bodies"].items():
        assert block["claims_checked"] >= 3, (
            f"{name} offers only {block['claims_checked']} checkable "
            "figures — the gate is passing on an almost empty set")


def test_the_gate_catches_an_invented_figure(tmp_path: pathlib.Path) -> None:
    """MUTATION TEST: plant a number no artifact contains, demand a catch.

    This is the whole point. The failure being guarded against is a
    number appearing in outbound copy that exists nowhere in the
    evidence — which is exactly what an outside reader found in the
    closing line of these emails.
    """
    original = PACKAGE.read_text(encoding="utf-8")
    marker = "0.7391"  # not a figure this project has ever produced
    assert marker not in original
    anchor = "## GENERALIST BODY"
    assert anchor in original
    mutated = original.replace(
        anchor, anchor + f"\n\nAdapted arms reach {marker} on held-out data.", 1)
    out = tmp_path / "reconciliation.json"
    try:
        PACKAGE.write_text(mutated, encoding="utf-8")
        result = _run(out)
        record = json.loads(out.read_text(encoding="utf-8"))
    finally:
        PACKAGE.write_text(original, encoding="utf-8")
    assert result.returncode != 0, (
        "the gate passed copy containing an invented figure — it cannot fail, "
        "so it licenses the copy instead of checking it")
    tokens = [item["token"]
              for item in record["bodies"]["GENERALIST"]["unreconciled"]]
    assert marker in tokens, f"caught something, but not the plant: {tokens}"


def test_a_range_is_not_read_as_a_signed_number(
        tmp_path: pathlib.Path) -> None:
    """Regression: `0.53-0.94` is a range, not the signed number `-0.94`.

    The first run of this gate reported three false failures of exactly
    this shape. A gate that cries wolf trains the reader to ignore it,
    which is the failure mode the web-form length warning was written to
    avoid, so the false-positive case is pinned as tightly as the true one.
    """
    out = tmp_path / "reconciliation.json"
    _run(out)
    record = json.loads(out.read_text(encoding="utf-8"))
    for name, block in record["bodies"].items():
        for item in block["unreconciled"]:
            assert not (item["kind"] == "signed"
                        and item["token"].startswith("-")), (
                f"{name}: {item['token']} looks like the right half of a "
                "range being read as a negative number")
