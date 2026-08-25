"""The headline gates' answer to the constrained-decoding objection.

`format_counterfactual.py` showed the objection lands on the waybill corpus.
This decomposition asks whether it reaches gates 1/4/5. What is pinned here:

1. The repair function actually repairs — a baseline with a wrong key set
   scores HIGHER after repair. Without this, "0% of the gap closed" would be
   indistinguishable from a no-op bug, and that is the number the pitch
   leans on.
2. The gates' format-neutral deltas are recomputed, not asserted.
3. The record declares itself post-hoc, not a gate.
4. The committed artifact does not drift from its generator.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "schema_conformance_decomposition.py"
BANKED = REPO / "experiments" / "schema_conformance_decomposition_2026-08-22.json"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def _run(tmp_path: pathlib.Path) -> dict:
    out = tmp_path / "decomp.json"
    env_script = [sys.executable, str(SCRIPT), "--out", str(out)]
    result = subprocess.run(env_script, capture_output=True, text=True,
                            cwd=str(REPO), check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(out.read_text(encoding="utf-8"))


def _gate(record: dict, name: str) -> dict:
    return next(g for g in record["gates"] if g["gate"].startswith(name))


def test_repair_is_not_a_no_op():
    """A wrong key set must be measurably improved by the repair."""
    from schema_conformance_decomposition import (leaf_paths, micro_f1_flat,
                                                  repair)

    gold = {"outer": {"kept": "1", "missed": "2"}}
    pred = {"outer": {"kept": "1"}, "invented": "junk"}
    raw = micro_f1_flat(leaf_paths(pred), gold)
    r1 = micro_f1_flat(repair(pred, gold, by_name=False), gold)
    assert r1 > raw, "path repair must drop the invented key and score higher"

    # And the name-repair must forgive a nesting mistake the path repair cannot.
    nested_wrong = {"elsewhere": {"missed": "2"}, "outer": {"kept": "1"}}
    r1n = micro_f1_flat(repair(nested_wrong, gold, by_name=False), gold)
    r2n = micro_f1_flat(repair(nested_wrong, gold, by_name=True), gold)
    assert r2n > r1n


def test_headline_gates_have_no_format_failures_to_explain(tmp_path):
    record = _run(tmp_path)
    for name in ("gate1", "gate5"):
        gate = _gate(record, name)
        assert gate["total_invalid_json_baseline"] == 0
        assert gate["total_invalid_json_adapted"] == 0
        # With nothing invalid on either arm the restriction is a no-op.
        assert (gate["seed_mean_delta_format_neutral"]
                == gate["seed_mean_delta_as_measured"])


def test_gate5_baseline_gets_the_schema_right_and_the_values_wrong(tmp_path):
    """The load-bearing claim: constrained decoding has nothing to fix here."""
    record = _run(tmp_path)
    gate = _gate(record, "gate5")
    for arm in gate["arms"]:
        conf = arm["baseline_schema_conformance"]
        assert conf["key_path_recall"] == 1.0
        assert conf["key_path_precision"] == 1.0
        assert conf["wrong_value_share_of_shared_leaves"] > 0.2
    assert gate["share_of_gap_closed_by_R2"] == 0.0
    assert (gate["seed_mean_delta_vs_R2"]
            == gate["seed_mean_delta_as_measured"])


def test_record_declares_itself_post_hoc(tmp_path):
    record = _run(tmp_path)
    status = record["status"].lower()
    assert "post-hoc" in status
    assert "not a preregistered gate" in status


def test_banked_record_matches_a_fresh_run(tmp_path):
    assert _run(tmp_path) == json.loads(BANKED.read_text(encoding="utf-8"))
