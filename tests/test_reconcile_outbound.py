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


def test_the_outbound_strategy_documents_are_inside_the_gate():
    """Round 9 (2026-09-23): the outbound strategy documents were outside
    every mechanical check while the emails and the application were
    inside it. Their names are a private list, absent from the public
    tree, so the membership check runs only where the list exists."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ro", SCRIPT)
    ro = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ro)
    listing = REPO / "docs" / "strategy" / "OUTBOUND_DOCS.txt"
    if listing.exists():
        assert len(ro.OUTBOUND_STRATEGY_DOCS) == 3
    # a section reference is not a count; a thousands-separated count still is
    tokens = [t for k, t in ro._claims("see section 1-2, artifacts named there") if k == "counted"]
    assert tokens == []
    tokens = [t for k, t in ro._claims("across 1,950 documents banked") if k == "counted"]
    assert tokens == ["1,950 documents"]


def _module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ro_cited", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_counted_claim_is_tied_to_the_artifact_its_own_clause_cites():
    """Slide copy said 236 banked artifacts while the artifact it cited
    said 237, and the gate passed it because an unrelated 236 existed
    elsewhere in the evidence (CORRECTIONS 2026-10-21). A claim takes the
    citation in its own clause, never a neighbour's."""
    ro = _module()
    text = ("fourteen results (`experiments/results_against_thesis_2026-09-17.json`), "
            "107 dated corrections (`experiments/self_correction_ledger.json`), "
            "236 banked artifacts in 48 days (`experiments/project_timeline.json`), "
            "a public repository whose CI re-derives the numbers. "
            "312 tests green, 236 banked artifacts (`experiments/project_timeline.json`).")
    counted = [(token, cited) for kind, token, cited in ro._cited_claims(text)
               if kind == "counted"]
    assert counted == [("107 dated corrections", "self_correction_ledger.json"),
                       ("236 banked artifacts", "project_timeline.json"),
                       ("312 tests", None)], counted
    # the cited artifact decides; the rest of the evidence does not rescue it
    assert not ro._reconciles("counted", "236 banked artifacts", "", [236.0],
                              cited_numbers=[237.0, 48.0])
    assert ro._reconciles("counted", "237 banked artifacts", "", [236.0],
                          cited_numbers=[237.0, 48.0])
    # no citation: any figure of that value in the evidence, as before
    assert ro._reconciles("counted", "236 banked artifacts", "", [236.0])


def test_the_gate_catches_a_count_that_disagrees_with_its_cited_artifact():
    """MUTATION TEST on the real artifacts: plant the public export's
    artifact count beside a citation of the source timeline. The value
    exists in the evidence, so only the citation check can catch it."""
    ro = _module()
    timeline = json.loads((REPO / "experiments" / "project_timeline.json").read_text())
    export = json.loads((REPO / "experiments" / "public_export_counts.json").read_text())
    planted = export["public_export_artifacts"]
    if planted == timeline["banked_artifacts"]:
        pytest.skip("the two counts agree today, so the plant would be true")
    haystack = ro._haystack()
    numbers = ro._haystack_numbers(haystack)
    body = f"{planted} banked artifacts in 48 days (`experiments/project_timeline.json`)."
    reconciled, missing = ro._check(body, None, haystack, numbers, {})
    assert [m["token"] for m in missing] == [f"{planted} banked artifacts"], (reconciled, missing)
    assert missing[0]["checked_against"] == "experiments/project_timeline.json"
    # and the true count passes
    body = f"{timeline['banked_artifacts']} banked artifacts (`experiments/project_timeline.json`)."
    assert ro._check(body, None, haystack, numbers, {})[1] == []
