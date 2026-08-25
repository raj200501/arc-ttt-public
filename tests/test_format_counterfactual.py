"""The constrained-decoding counterfactual must stay honest and reproducible.

Three things are pinned here:

1. The script reproduces the PUBLISHED paired result as its control reading.
   If reading A ever drifts from `blind_rehearsal_baseline_2026-08-21.json`,
   one of the two is wrong and this fails.
2. Every grant to the rival explanation moves the delta DOWN, never up. A
   "counterfactual" that flattered us would be a bug, not a finding.
3. The banked record says out loud that it is post-hoc, not a gate.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "format_counterfactual.py"
BANKED = REPO / "experiments" / "format_counterfactual_2026-08-22.json"
PAIRED = REPO / "experiments" / "blind_rehearsal_baseline_2026-08-21.json"


def _run(tmp_path: pathlib.Path) -> dict:
    out = tmp_path / "cf.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out)],
        capture_output=True, text=True, cwd=str(REPO), check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(out.read_text(encoding="utf-8"))


def _by_name(record: dict) -> dict:
    return {r["reading"]: r for r in record["readings"]}


def test_control_reading_reproduces_the_published_paired_result(tmp_path):
    record = _run(tmp_path)
    published = json.loads(PAIRED.read_text(encoding="utf-8"))
    control = _by_name(record)["A_as_measured"]

    assert control["baseline_mean"] == published["baseline_kshot_greedy"][
        "mean_micro_f1"]
    assert control["adapted_mean"] == published["adapted_greedy"][
        "mean_micro_f1"]
    assert control["mean_delta"] == published["paired"]["mean_delta"]
    assert control["sign_test"]["wins"] == published["paired"]["sign_test"][
        "wins"]
    assert control["sign_test"]["losses"] == published["paired"]["sign_test"][
        "losses"]


def test_every_grant_to_the_rival_explanation_lowers_our_delta(tmp_path):
    record = _run(tmp_path)
    r = _by_name(record)
    # Each reading grants the rival explanation strictly more than the last.
    assert (r["A_as_measured"]["mean_delta"]
            >= r["B_schema_key_pruned"]["mean_delta"]
            >= r["C_pruned_plus_imputed"]["mean_delta"]
            >= r["D_pruned_plus_perfect"]["mean_delta"])


def test_the_assumption_free_reading_is_reported_and_is_the_quoted_one(
        tmp_path):
    record = _run(tmp_path)
    r = _by_name(record)
    neutral = r["F_format_neutral_subset_nothing_granted"]
    # No imputation: it must drop exactly the unparseable documents.
    assert neutral["n"] == 30 - len(record["unparseable_prompted_documents"])
    assert "NOTHING" in neutral["what_it_assumes"]
    assert record["bottom_line"]["format_neutral_delta"] == neutral[
        "mean_delta"]
    # The reading that DOES grant something must not claim otherwise, and
    # granting the key-pruning must lower the delta relative to F. This is
    # the mislabel an outside auditor caught on 2026-08-22.
    pruned = r["E_format_neutral_subset_key_pruned"]
    assert "not\nassumption-free" in pruned["what_it_assumes"].replace(
        " ", "\n") or "not assumption-free" in pruned["what_it_assumes"]
    assert pruned["mean_delta"] < neutral["mean_delta"]
    assert pruned["baseline_mean"] > neutral["baseline_mean"]


def test_the_nothing_granted_reading_agrees_with_the_verdict_page(tmp_path):
    """F must reproduce the +4.14 VERDICT.md carried for this subset."""
    record = _run(tmp_path)
    f = _by_name(record)["F_format_neutral_subset_nothing_granted"]
    assert abs(f["mean_delta"] - 0.0414) < 5e-4


def test_record_declares_itself_post_hoc_and_not_a_gate(tmp_path):
    record = _run(tmp_path)
    status = record["status"].lower()
    assert "post-hoc" in status
    assert "not a preregistered gate" in status
    assert "not a gate pass" in status


def test_banked_record_matches_a_fresh_run(tmp_path):
    """The committed artifact is not allowed to drift from its generator."""
    fresh = _run(tmp_path)
    banked = json.loads(BANKED.read_text(encoding="utf-8"))
    assert fresh == banked
