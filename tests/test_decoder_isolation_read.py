"""Addendum AA: the successor reader of Addendum X's banked cells.

Every test here runs on synthetic cells, synthetic proofs or precondition
flags. None computes a reading on the real cells: the reading runs once,
after the protocol is anchored and Bitcoin-attested, and is banked.
"""
import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aa = _load("decoder_isolation_read")
x = aa.x
BITCOIN = aa.ATTESTATIONS["bitcoin"]
PENDING = aa.ATTESTATIONS["pending"]


def test_scope_is_the_four_readable_families_and_phi3_is_excluded():
    assert aa.FAMILIES == ("qwen2.5-0.5b", "smollm2-1.7b", "granite-2b", "falcon3-1b")
    assert set(aa.EXCLUDED) == {"phi3-mini"}
    assert set(aa.FAMILIES) | set(aa.EXCLUDED) == set(x.CELLS)
    # the exclusion note says what Y did NOT explain, not only what it did
    assert "remain unexplained" in aa.EXCLUDED["phi3-mini"]


# -- the anchor ------------------------------------------------------------------


def test_anchored_digest_reads_a_real_proof_and_refuses_malformed_ones(tmp_path):
    z = REPO / "docs" / "research" / "ADDENDUM_Z_PROTOCOL.md.2026-09-23T0219Z.ots"
    # the digest quoted in anchor commit 7919d36, not the living file's
    assert aa.anchored_digest(z) == "511db57e530a6ebc07de18f2563629ce12b3b9b664cc822221f2a921f6c87cd1"
    digest = bytes(32)
    cases = {
        "header_only": aa.OTS_MAGIC + b"\x01\x08" + digest,
        "truncated": aa.OTS_MAGIC + b"\x01\x08" + digest[:10],
        "version_2": aa.OTS_MAGIC + b"\x02\x08" + digest + b"\x00" + BITCOIN,
        "not_sha256": aa.OTS_MAGIC + b"\x01\x02" + digest + b"\x00" + BITCOIN,
        "garbage": b"not a proof",
    }
    for name, blob in cases.items():
        path = tmp_path / f"{name}.ots"
        path.write_bytes(blob)
        assert aa.anchored_digest(path) is None, name


def _anchor_fixture(tmp_path, monkeypatch, *, attestation=BITCOIN, edit_after=b"", wrong_digest=False,
                    placeholder=False):
    research = tmp_path / "docs" / "research"
    research.mkdir(parents=True)
    stamp = "2026-09-23T0900Z"
    reader_sha = hashlib.sha256(aa.READER_PATH.read_bytes()).hexdigest()
    x_sha = hashlib.sha256(aa.X_MODULE_PATH.read_bytes()).hexdigest()
    name = "STAMP_PLACEHOLDER" if placeholder else stamp
    text = (f"Frozen. Anchored as `ADDENDUM_AA_PROTOCOL.md.{name}.ots`.\n"
            f"`scripts/decoder_isolation_read.py` sha256 `{reader_sha}` and "
            f"`scripts/cord_decoder_isolation.py` sha256 `{x_sha}`.\n").encode()
    protocol = research / "ADDENDUM_AA_PROTOCOL.md"
    protocol.write_bytes(text + edit_after)
    (research / f"snapshots_ADDENDUM_AA_PROTOCOL_{stamp}.md").write_bytes(text)
    committed = bytes(32) if wrong_digest else hashlib.sha256(text).digest()
    (research / f"ADDENDUM_AA_PROTOCOL.md.{stamp}.ots").write_bytes(
        aa.OTS_MAGIC + b"\x01\x08" + committed + b"\xf0\x10" + b"\x00" + attestation + b"\x01")
    monkeypatch.setattr(aa, "REPO", tmp_path)
    monkeypatch.setattr(aa, "PROTOCOL_PATH", protocol)
    monkeypatch.setattr(aa, "READER_PATH", tmp_path / "scripts" / "decoder_isolation_read.py")
    monkeypatch.setattr(aa, "X_MODULE_PATH", tmp_path / "scripts" / "cord_decoder_isolation.py")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "decoder_isolation_read.py").write_bytes(REPO.joinpath(
        "scripts", "decoder_isolation_read.py").read_bytes())
    (tmp_path / "scripts" / "cord_decoder_isolation.py").write_bytes(REPO.joinpath(
        "scripts", "cord_decoder_isolation.py").read_bytes())
    return tmp_path


def test_the_anchor_gate_passes_only_when_everything_agrees(tmp_path, monkeypatch):
    _anchor_fixture(tmp_path, monkeypatch)
    record = aa.anchor_record()
    assert record["ok"], record
    assert record["attestations"] == ["bitcoin"] and record["bitcoin_attested"]


def test_an_erratum_appended_later_does_not_break_the_gate(tmp_path, monkeypatch):
    _anchor_fixture(tmp_path, monkeypatch, edit_after=b"\n## Erratum 2026-10-01\n...\n")
    assert aa.anchor_record()["ok"]


@pytest.mark.parametrize("kwargs, why", [
    ({"attestation": PENDING}, "a calendar promise only"),
    ({"wrong_digest": True}, "the proof commits to other bytes"),
    ({"placeholder": True}, "the stamp placeholder was never replaced"),
])
def test_the_anchor_gate_refuses(tmp_path, monkeypatch, kwargs, why):
    _anchor_fixture(tmp_path, monkeypatch, **kwargs)
    assert not aa.anchor_record()["ok"], why


def test_the_anchor_gate_refuses_an_in_place_edit_and_a_code_change(tmp_path, monkeypatch):
    root = _anchor_fixture(tmp_path, monkeypatch)
    protocol = root / "docs" / "research" / "ADDENDUM_AA_PROTOCOL.md"
    protocol.write_bytes(protocol.read_bytes().replace(b"Frozen.", b"Frozen!"))
    assert not aa.anchor_record()["ok"]
    root = _anchor_fixture(tmp_path / "second", monkeypatch)
    reader = root / "scripts" / "decoder_isolation_read.py"
    reader.write_bytes(reader.read_bytes() + b"\n# edited after the freeze\n")
    assert not aa.anchor_record()["code"]["matches"]


def test_the_real_protocol_quotes_its_stamp_once_it_is_frozen():
    text = aa.PROTOCOL_PATH.read_text()
    stamp = aa.stamp_of(text)
    if stamp is None:
        assert "STAMP_PLACEHOLDER" in text  # not yet frozen: nothing may be read
        return
    assert "STAMP_PLACEHOLDER" not in text and "_SHA_PLACEHOLDER" not in text
    assert (aa.PROTOCOL_PATH.parent / f"snapshots_ADDENDUM_AA_PROTOCOL_{stamp}.md").exists()
    assert (aa.PROTOCOL_PATH.parent / f"ADDENDUM_AA_PROTOCOL.md.{stamp}.ots").exists()


# -- preconditions ---------------------------------------------------------------


def test_a_recomputation_is_refused_until_the_banked_reading_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(aa, "OUT", tmp_path / "banked.json")
    problems = aa.preconditions(x._scope(), tmp_path / "mine.json", banked_reading=False)
    assert any("does not exist yet" in p for p in problems)
    (tmp_path / "banked.json").write_text("{}")
    problems = aa.preconditions(x._scope(), tmp_path / "mine.json", banked_reading=False)
    assert not any("does not exist yet" in p for p in problems)


def test_the_banked_reading_is_made_once(tmp_path, monkeypatch):
    out = tmp_path / "banked.json"
    out.write_text("{}")
    monkeypatch.setattr(aa, "OUT", out)
    monkeypatch.setattr(aa, "anchor_record", lambda: {"ok": True})
    problems = aa.preconditions(x._scope(), out, banked_reading=True)
    assert any("already banked" in p for p in problems)


def test_gate_status_names_granite_not_applicable_and_never_passed():
    scope = x._scope()
    granite = aa.gate_status("granite-2b", scope)
    assert granite["status"] == "NOT APPLICABLE" and granite["n"] == 0
    for family in ("qwen2.5-0.5b", "smollm2-1.7b", "falcon3-1b"):
        assert aa.gate_status(family, scope)["status"] == "PASSED"


# -- X's moved code on synthetic cells, the EOS erratum included -----------------


@pytest.fixture
def synthetic_x(monkeypatch, tmp_path):
    out_dir, split = tmp_path / "cells", tmp_path / "split"
    out_dir.mkdir()
    split.mkdir()
    docs = ["d1", "d2", "d3"]
    gold = {"d1": {"a": 1}, "d2": {"a": 2}, "d3": {"a": 3}}
    (split / "gold.jsonl").write_text("".join(json.dumps({"id": d, "gold": gold[d]}) + "\n" for d in docs))
    (split / "train.jsonl").write_text("")
    same = {d: "h" for d in docs}
    plain = {"predictions": {"d1": '{"a": 1}', "d2": "not json", "d3": '{"a": 2}'},
             "per_document_prompt_sha256_16": same,
             "per_document_token_ids": {"d1": [1, 2, 3], "d2": [5, 6], "d3": [7, 8]},
             "per_document_decode": {"d1": {"stopped_on": "complete"}, "d2": {"stopped_on": "eos"},
                                     "d3": {"stopped_on": "complete"}}}
    const = {"predictions": {"d1": '{"a": 1}', "d2": "not json", "d3": '{"a": 3}'},
             "per_document_prompt_sha256_16": same,
             "per_document_decode": {"d1": {"constrained_steps": 0}, "d2": {"constrained_steps": 0},
                                     "d3": {"constrained_steps": 1}}}
    gen = {"predictions": {"d1": '{"a": 1}', "d2": "not json", "d3": '{"b": 1}'},
           "per_document_prompt_sha256_16": same,
           # d1: generate runs on past the closing brace, then stops: agreement
           # d2: our loop stopped on EOS; generate kept its stop token 9 -- the erratum
           # d3: a real divergence at index 1
           "per_document_token_ids": {"d1": [1, 2, 3, 4, 9], "d2": [5, 6, 9], "d3": [7, 1, 9]},
           "effective_generation_config": {"eos_token_id": [9, 10]}}
    g0 = {"predictions": {d: "" for d in docs}}
    (out_dir / "fam_plain.json").write_text(json.dumps(plain))
    (out_dir / "fam_constrained.json").write_text(json.dumps(const))
    (out_dir / "fam_generate_neutral.json").write_text(json.dumps(gen))
    (tmp_path / "g0.json").write_text(json.dumps(g0))
    monkeypatch.setattr(x, "REPO", tmp_path)
    monkeypatch.setattr(x, "OUT_DIR", out_dir)
    monkeypatch.setattr(x, "CELLS", {"fam": ("m", "float32", tmp_path / "g0.json")})
    monkeypatch.setattr(x.cft, "SPLIT_DIR", split)
    return {"fam": {"arm_C_reused_from_V": False, "arm_G_runs": True, "modifiers": {}}}


def test_read_families_runs_on_synthetic_cells_and_applies_the_eos_erratum(synthetic_x, capsys):
    rows, x1, x2, x3, mismatches = x.read_families(["fam"], synthetic_x)
    (row,) = rows
    assert mismatches == [] and x3 is None
    assert row["x1_identical_documents"] == 2 and row["x1_differing_documents"] == ["d3"]
    assert row["accounting_consistent"]  # d3 differs and carries a constrained step
    assert row["x1_invalid_plain"] == 1 and row["x1_invalid_constrained"] == 1
    assert x1["fam"].startswith("UNTESTABLE AT SIZE")
    # the erratum: as first coded, d2 (our EOS stop vs generate's kept stop
    # token) was a divergence; protocol-conforming, only d3 is
    assert row["x2_divergent_documents_as_first_coded"] == 2
    assert row["x2_divergent_documents"] == 1 and row["x2_first_divergence_index"] == {"d3": 1}
    assert row["x2_documents_changed_by_the_eos_erratum"] == ["d2"]
    assert x2["fam"] == "PATH DIVERGES (1)"


def test_read_families_only_reads_the_families_asked_for(synthetic_x):
    rows, *_ = x.read_families([], synthetic_x)
    assert rows == []


def test_the_generate_stop_is_stripped_once_and_only_if_it_is_a_stop():
    assert x._without_generate_stop([1, 2, 9], {9}) == [1, 2]
    assert x._without_generate_stop([1, 9, 9], {9}) == [1, 9]
    assert x._without_generate_stop([1, 2, 3], {9}) == [1, 2, 3]
    assert x._without_generate_stop([], {9}) == []


# -- the readings layered on X's -------------------------------------------------


def test_the_unscoped_x1_holds_sentence_is_unreachable():
    four = {f: "CONSTRAINT REMOVES" for f in aa.FAMILIES}
    finding = aa.scoped_x1_finding(four)
    assert finding.startswith("X1 HOLDS ON THE FAMILIES READ ONLY: CONSTRAINT REMOVES in 4 of 5")
    assert "unverified on 1 of 5" in finding and "is NOT licensed" in finding
    assert "no family is an exception" not in finding.split("is NOT licensed")[1]
    three = dict(four) | {"granite-2b": "CONSTRAINT INERT"}
    assert aa.scoped_x1_finding(three).startswith("X1 MIXED: CONSTRAINT REMOVES in 3 of 5")
    hurts = {f: "CONSTRAINT INERT" for f in aa.FAMILIES}
    hurts["falcon3-1b"] = "CONSTRAINT HURTS (invalid 1 -> 3, regressions 2)"
    failed = aa.scoped_x1_finding(hurts)
    assert failed.startswith("X1 FAILS IN falcon3-1b") and "families tested" not in failed


def _row(family, x1="CONSTRAINT INERT", x2="PATH REPRODUCES", basis="token ids",
         consistent=True, differing=()):
    return {"family": family, "n": 50, "x1_reading": x1, "x2_reading": x2,
            "x1_invalid_plain": 0, "x1_invalid_constrained": 0, "x2_basis": basis,
            "x2_invalid_plain": 3, "x2_invalid_other_arm": 5, "x2_compared_against": "G0",
            "accounting_consistent": consistent, "x1_differing_documents": list(differing),
            "documents_with_zero_steps_but_differing_arms": [] if consistent else ["cord-001"],
            "documents_with_steps_but_identical_arms": [],
            "x2_divergent_documents_as_first_coded": 0, "x2_divergent_documents": 0,
            "x2_documents_changed_by_the_eos_erratum": []}


def _rows(**over):
    rows = [
        _row("qwen2.5-0.5b"),
        _row("smollm2-1.7b", x2="PATH REPRODUCES (WEAKER TEST: decoded text, not token ids)",
             basis="decoded text"),
        _row("granite-2b"),
        _row("falcon3-1b", x1="CONSTRAINT HELPS", differing=["cord-003"] * 7,
             x2="PATH DIVERGES (2) (WEAKER TEST: decoded text, not token ids)", basis="decoded text"),
    ]
    for r in rows:
        r.update(over.get(r["family"], {}))
    return rows


def test_prediction_checks_follow_x_verbatim_and_mark_phi3_uncheckable():
    checks = aa.prediction_checks(_rows())
    assert checks["x1_inert_holds"] == {f: True for f in aa.X1_INERT_PREDICTED}
    assert checks["x2_where_predicted"]["smollm2-1.7b"]["basis"] == "decoded text"
    assert "granite-2b" not in checks["x2_where_predicted"]
    assert len(checks["not_checkable"]) == 3
    assert "Granite are not predicted" in checks["verbatim"]["X2"]


def test_licensed_sentences_carry_the_caveats_and_void_a_family():
    scope = x._scope()
    identity = {f: {"C vs P": "verified"} for f in aa.FAMILIES}
    identity["granite-2b"] = {"C vs P": "MISMATCHED"}
    sentences = aa.licensed_sentences(_rows(), "X1 MIXED: ... Per-family: {}", None, identity, scope)
    text = " ".join(sentences)
    assert "granite-2b: VOID" in text and "granite-2b (X2)" not in text
    assert "token-for-token" in [s for s in sentences if s.startswith("qwen2.5-0.5b (X2)")][0]
    smol = [s for s in sentences if s.startswith("smollm2-1.7b (X2)")][0]
    assert "WEAKER TEST" in smol and "token-for-token" not in smol
    falcon = [s for s in sentences if s.startswith("falcon3-1b (X2)")][0]
    assert "no environment record" in falcon and "invalid outputs 3" in falcon
    falcon_x1 = [s for s in sentences if s.startswith("falcon3-1b (X1)")][0]
    assert "cannot separate the two" in falcon_x1  # arm C is V's reused cell
    assert "phi3" not in " ".join(s for s in sentences if not s.startswith("X1 (combined")).lower()


def test_obligations_fire_from_the_readings():
    owed = aa.obligations(_rows(), None)
    assert [o["file"] for o in owed] == ["tools/README.md"]
    hurts = _rows(**{"falcon3-1b": {"x1_reading": "CONSTRAINT HURTS (invalid 1 -> 3, regressions 2)"}})
    x3 = {"reading": "V'S ATTRIBUTION IS TOO STRONG: 3 of 5 ..."}
    files = [o["file"] for o in aa.obligations(hurts, x3)]
    assert files.count("tools/README.md") == 2
    assert any(f.startswith("VERDICT.md") for f in files) and any(f.startswith("CORRECTIONS.md") for f in files)


def test_read_withholds_and_writes_nothing_when_a_precondition_fails(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(aa, "OUT", tmp_path / "out.json")
    monkeypatch.setattr(aa, "preconditions", lambda scope, out, banked: ["anchor: not attested"])

    def forbidden(*a, **k):
        raise AssertionError("the reading was computed before its preconditions held")

    monkeypatch.setattr(aa.x, "read_families", forbidden)
    assert aa.read(tmp_path / "out.json") == 2
    assert "WITHHELD" in capsys.readouterr().out
    assert not (tmp_path / "out.json").exists()


def test_read_voids_mismatched_families_publishes_first_and_withholds_x3(monkeypatch, tmp_path):
    out = tmp_path / "out.json"
    monkeypatch.setattr(aa, "OUT", out)
    monkeypatch.setattr(aa, "preconditions", lambda scope, o, banked: [])
    monkeypatch.setattr(aa, "anchor_record", lambda: {"ok": True, "stamp": "t"})
    rows = _rows(**{"smollm2-1.7b": {"accounting_consistent": False}})
    x3 = {"family": "qwen2.5-0.5b", "reading": "WITHHELD: ...", "v_regression_count_reproduces": False,
          "regressions_explained_by_the_defaults": 4, "residual_documents": ["a"],
          "residual_attributed_by_arithmetic": {"a": "x"}}

    def fake(families, scope):
        print("console line from read_families")
        return rows, {r["family"]: r["x1_reading"] for r in rows}, \
            {r["family"]: r["x2_reading"] for r in rows}, x3, ["granite-2b/cord-001 (C vs P)"]

    monkeypatch.setattr(aa.x, "read_families", fake)
    monkeypatch.setattr(aa, "prompt_identity", lambda f, scope, mm: {
        "C vs P": "MISMATCHED" if f == "granite-2b" else "verified"})
    assert aa.read(out) == 0
    record = json.loads(out.read_text())
    assert record["void_families"] == ["granite-2b"]
    assert record["published_first"][0].startswith("PROMPT IDENTITY FAILED")
    assert "accounting is wrong" in record["published_first"][1]
    assert "does not reproduce" in record["published_first"][2]
    assert record["x3"]["regressions_explained_by_the_defaults"] is None
    assert record["console_of_read_families"].startswith("console line")
    assert record["read_started_utc"] <= record["read_finished_utc"]
    assert record["determinism_gate_from_x"]["granite-2b"]["status"] == "NOT APPLICABLE"


def test_the_protocol_names_its_choices_and_owes_the_disclosure():
    text = aa.PROTOCOL_PATH.read_text()
    for needle in ("Nothing here is read blind", "HURTS veto", "NOT APPLICABLE",
                   "moves X2 toward\n   PATH REPRODUCES", "not verifiable",
                   "cord_decoder_isolation_read_2026-09-23.json", "Granite-3.1-2B is 2.53B",
                   "remain" if False else "did **not** explain"):
        assert needle in text, needle
    assert "cannot help us" not in text and "strictly harder" not in text
