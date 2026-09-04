"""Addendum U pins: the per-record status, the rates, the three frozen
readings at their exact boundaries, and the corpus builder's refusals.
None of these touch the banked corpus or any parser library."""
import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pr = _load(REPO / "scripts" / "parser_robustness.py", "parser_robustness")
fcorp = _load(REPO / "tools" / "fence_corpus.py", "fence_corpus")


def test_status_is_the_five_way_split():
    ref = {"a": 1}
    assert pr.status(ref, {"a": 1}) == "ok"
    assert pr.status(ref, None) == "lost"
    assert pr.status(ref, {"a": 2}) == "diverged"
    assert pr.status(None, None) == "agree_none"
    assert pr.status(None, {"a": 1}) == "fabricated"


def test_rates_denominators_are_ref_and_noref():
    r = pr.rates(["ok", "lost", "diverged", "agree_none", "fabricated", "fabricated"])
    assert (r["n"], r["n_ref"], r["n_noref"]) == (6, 3, 3)
    assert r["lost_rate"] == round(1 / 3, 4)
    assert r["diverged_rate"] == round(1 / 3, 4)
    assert r["fabricated_rate"] == round(2 / 3, 4)
    assert r["hazard_rate"] == round(3 / 6, 4)
    empty = pr.rates(["agree_none"])
    assert empty["lost_rate"] is None and empty["fabricated_rate"] == 0.0


def test_strict_parser_is_the_evals_shape():
    assert pr.parser_strict('{"a": 1}') == {"a": 1}
    assert pr.parser_strict('```json\n{"a": 1}\n```') is None   # the fail-open
    assert pr.parser_strict('[1, 2]') is None                    # non-object
    assert pr.parser_strict('') is None


def test_u1_boundaries_and_combination():
    assert pr.family_strict_reading(0.50) == "LOSES"
    assert pr.family_strict_reading(0.4999) == "PARTIAL"
    assert pr.family_strict_reading(0.10) == "PARTIAL"
    assert pr.family_strict_reading(0.0999) == "DOES NOT LOSE"
    L, P, D = "LOSES", "PARTIAL", "DOES NOT LOSE"
    assert pr.read_u1({"a": L, "b": L, "c": L, "d": P}).startswith("U1 HOLDS")
    assert pr.read_u1({"a": L, "b": L, "c": P, "d": P}).startswith("U1 MIXED")
    r = pr.read_u1({"a": L, "b": L, "c": L, "d": D})
    assert r.startswith("U1 EXCEPTION IN d") and "HOLDS" not in r
    assert "never 'across families'" in r


def test_u2_material_needs_n_at_least_30():
    # a 0.10 hazard on a 29-record slice is not MATERIAL
    r = pr.read_u2(0.004, {"s1": (0.10, 29), "s2": (0.0, 100)})
    assert r.startswith("U2 HARMLESS")
    r = pr.read_u2(0.02, {"s1": (0.10, 30), "s2": (0.0, 100)})
    assert r.startswith("U2 MATERIAL") and "s1" in r
    r = pr.read_u2(0.02, {"s1": (0.04, 100)})
    assert r.startswith("U2 PRESENT")
    assert pr.read_u2(0.0099, {"s1": (0.0499, 100)}).startswith("U2 HARMLESS")
    assert pr.read_u2(0.01, {"s1": (0.0, 100)}).startswith("U2 PRESENT")


def test_u3_boundary():
    assert pr.read_u3(0.0499, 100, {}).startswith("U3 SAME")
    assert pr.read_u3(0.05, 100, {"fenced": 5}).startswith("U3 RESIDUAL")


def test_builder_refuses_parsed_objects(tmp_path, monkeypatch):
    art = tmp_path / "cord_fence_tax_cells"
    art.mkdir()
    (art / "0.5b_schema.json").write_text(json.dumps({
        "model": "Qwen/Qwen2.5-0.5B-Instruct", "regime": "schema", "k": 0,
        "decode": "greedy, max_new_tokens=512, float32, CPU",
        "predictions": {"cord-000": {"already": "parsed"}}}))
    monkeypatch.setattr(fcorp, "EXP", tmp_path)
    monkeypatch.setattr(fcorp, "REGISTRY", [
        ("cord_fence_tax_cells/0.5b_schema.json", "qwen2.5", "0.5B", False, "cord", "regime")])
    with pytest.raises(fcorp.CorpusError, match="not raw text"):
        fcorp.build()


def test_builder_refuses_registry_contradiction(tmp_path, monkeypatch):
    art = tmp_path / "cord_fence_tax_cells"
    art.mkdir()
    (art / "0.5b_schema.json").write_text(json.dumps({
        "model": "Qwen/Qwen2.5-1.5B-Instruct", "regime": "schema", "k": 0,
        "decode": "greedy, max_new_tokens=512, float32, CPU",
        "predictions": {"cord-000": "{}"}}))
    monkeypatch.setattr(fcorp, "EXP", tmp_path)
    monkeypatch.setattr(fcorp, "REGISTRY", [
        ("cord_fence_tax_cells/0.5b_schema.json", "qwen2.5", "0.5B", False, "cord", "regime")])
    with pytest.raises(fcorp.CorpusError, match="contradicts registry"):
        fcorp.build()


def test_builder_names_absent_artifacts(tmp_path, monkeypatch):
    art = tmp_path / "cord_fence_tax_cells"
    art.mkdir()
    (art / "0.5b_schema.json").write_text(json.dumps({
        "model": "Qwen/Qwen2.5-0.5B-Instruct", "regime": "schema", "k": 0, "dtype": "float32",
        "decode": "greedy, max_new_tokens=512, float32, CPU",
        "predictions": {"cord-001": "{}", "cord-000": "```json\n{}\n```"}}))
    monkeypatch.setattr(fcorp, "EXP", tmp_path)
    monkeypatch.setattr(fcorp, "REGISTRY", [
        ("cord_fence_tax_cells/0.5b_schema.json", "qwen2.5", "0.5B", False, "cord", "regime"),
        ("cord_fence_tax_cells/missing.json", "phi3", "3.8B", False, "cord", "regime")])
    records, manifest = fcorp.build()
    assert manifest["artifacts_absent"] == ["cord_fence_tax_cells/missing.json"]
    assert [r["doc_id"] for r in records] == ["cord-000", "cord-001"]  # sorted, deterministic
    assert manifest["families_present"] == ["qwen2.5"]
