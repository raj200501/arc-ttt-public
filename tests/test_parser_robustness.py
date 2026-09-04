"""Addendum U pins: the per-record status, the rates, the three frozen
readings at their exact boundaries, the substance decomposition, and the
corpus builder's refusals. None of these touch the banked corpus or any
parser library."""
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
    assert "NO combined headline" not in r  # three LOSES: the exception form is licensed


def test_u1_two_loses_plus_exceptions_has_no_headline_in_either_form():
    # the case this corpus produced: {L, L, D, D}. The exception is named
    # (the frozen letter) AND the reading says there is no combined
    # headline at all (the non-flattering reading, which governs).
    L, D = "LOSES", "DOES NOT LOSE"
    r = pr.read_u1({"a": L, "b": L, "c": D, "d": D})
    assert r.startswith("U1 EXCEPTION IN c, d")
    assert "NO combined headline in either form" in r
    assert "HOLDS" not in r


def test_u2_material_needs_n_at_least_30_but_harmless_needs_no_slice_at_all():
    # a 0.10 hazard on a 29-record slice is not MATERIAL (n < 30) ...
    r = pr.read_u2(0.004, {"s1": (0.10, 29), "s2": (0.0, 100)})
    assert not r.startswith("U2 MATERIAL")
    # ... and by the protocol's letter it also blocks HARMLESS ("no slice >= 0.05")
    assert r.startswith("U2 PRESENT")
    r = pr.read_u2(0.02, {"s1": (0.10, 30), "s2": (0.0, 100)})
    assert r.startswith("U2 MATERIAL") and "s1" in r
    assert pr.read_u2(0.02, {"s1": (0.04, 100)}).startswith("U2 PRESENT")
    assert pr.read_u2(0.0099, {"s1": (0.0499, 100)}).startswith("U2 HARMLESS")
    assert pr.read_u2(0.01, {"s1": (0.0, 100)}).startswith("U2 PRESENT")


def test_u3_boundary_and_empty_pool():
    assert pr.read_u3(0.0499, 100, {}).startswith("U3 SAME")
    assert pr.read_u3(0.05, 100, {"fenced": 5}).startswith("U3 RESIDUAL")
    assert pr.read_u3(None, 0, {}).startswith("U3 NOT READABLE")


def test_decompose_fabricated_does_not_call_a_stripped_fence_recovery():
    # a leading fence the reference stripped, body malformed (an expression):
    # this is a REPAIR, never "the reference's undercount"
    text = '```json\n{"a": 2 * 13000}\n```'
    assert pr.decompose_fabricated(text, {"a": 2}, fenced=True) == "leading_fenced_malformed"
    # prose, then a fence: the reference's leading-fence scope missed it
    text = 'Here it is:\n```json\n{"a": 1,}\n```'
    assert pr.decompose_fabricated(text, {"a": 1}, fenced=False) == "fence_after_prose"
    # an exact object inside prose: recovered, whatever the fence flag says
    text = 'Sure! {"a": 1} hope that helps'
    assert pr.decompose_fabricated(text, {"a": 1}, fenced=False) == "exact_object_present"
    # no fence, malformed
    assert pr.decompose_fabricated('{"a": 1', {"a": 1}, fenced=False) == "unfenced_malformed"


def test_brace_depth_ignores_braces_inside_strings():
    assert pr._brace_depth('{"a": 1}', 0) == 0
    assert pr._brace_depth('{"a": {"b": 1}}', 6) == 1          # nested
    assert pr._brace_depth('{"a": "{not a brace", "b": {', 27) == 1
    assert pr._brace_depth('prose {"a":1} more {"b":2}', 19) == 0  # two top-level


def test_decompose_marks_a_nested_fragment_as_not_recovery():
    # the model's top-level object is unparseable (single-quoted key later);
    # a last-span helper returns the inner menu item -- a fragment, not the answer
    text = '{"menu":{"cnt":"1","nm":"Tea"},\'sub_total\':{\'price\':\'1\'}}'
    assert pr.decompose_fabricated(text, {"cnt": "1", "nm": "Tea"}, fenced=False) == "nested_fragment_returned"
    # trailing prose after a complete top-level object: recovery
    text = '{"cnt":"1","nm":"Tea"}\nHope this helps!'
    assert pr.decompose_fabricated(text, {"cnt": "1", "nm": "Tea"}, fenced=False) == "exact_object_present"


def test_ext_panel_wrappers_return_object_or_none():
    pytest.importorskip("instructor"); pytest.importorskip("smolagents"); pytest.importorskip("llama_index.core")
    parsers, versions, raw = pr.make_parsers_ext()
    fenced = '```json\n{"a": 1}\n```'
    for name, fn in parsers.items():
        assert fn(fenced) == {"a": 1}, name          # every helper handles a leading fence
        assert fn("") is None, name                   # empty input is never an object
    # non-object -> None (the frozen rule) where the helper returns the array;
    # smolagents slices first "{" to last "}" and so returns the inner object
    assert parsers["instructor_extract_json_from_codeblock"]('[{"a": 1}]') is None
    assert parsers["llama_index_parse_json_markdown"]('[{"a": 1}]') is None
    assert parsers["smolagents_parse_json_blob"]('[{"a": 1}]') == {"a": 1}
    # the shipped behaviours the protocol wrote down before the run
    assert parsers["instructor_extract_json_from_codeblock"]('{"a":1}{"b":2}') == {"b": 2}   # LAST span
    assert parsers["smolagents_parse_json_blob"]('{"a": 2 * 3}') is None                   # no repair
    assert parsers["llama_index_parse_json_markdown"]('{"a": 2 * 3}') == {"a": "2 * 3"}    # yaml fallback
    assert set(versions) == {"instructor", "smolagents", "llama_index_core"}


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
