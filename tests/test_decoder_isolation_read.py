"""Addendum AA: the successor reader of Addendum X's banked cells.

Every test here runs on synthetic cells, synthetic proofs, synthetic git
repositories or precondition flags. None opens a real cell: the reading
runs once, after the protocol is anchored and Bitcoin-attested, and is
banked. The real files these tests touch are the protocol, the pinned code,
X's banked artifact (for its gate entries), the scope record, the input
manifest's key list and the committed OpenTimestamps proofs.
"""
import hashlib
import importlib.util
import inspect
import json
import os
import pathlib
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

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
STAMP = "2026-10-21T2300Z"


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


def _git_repo(root: pathlib.Path):
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fixture")


def test_scope_is_the_four_readable_families_and_phi3_is_excluded():
    assert aa.FAMILIES == ("qwen2.5-0.5b", "smollm2-1.7b", "granite-2b", "falcon3-1b")
    assert set(aa.EXCLUDED) == {"phi3-mini"}
    assert set(aa.FAMILIES) | set(aa.EXCLUDED) == set(x.CELLS)
    # the exclusion note says what Y did NOT explain, not only what it did
    assert "remain unexplained" in aa.EXCLUDED["phi3-mini"]


# -- the proof --------------------------------------------------------------------


def _varuint(n: int) -> bytes:
    out = b""
    while True:
        byte = n & 0x7F
        n >>= 7
        out += bytes([byte | (0x80 if n else 0)])
        if not n:
            return out


def _attestation(kind: str, height: int = 968300) -> bytes:
    if kind == "bitcoin":
        payload = _varuint(height)
        return b"\x00" + BITCOIN + _varuint(len(payload)) + payload
    uri = b"https://alice.btc.calendar.opentimestamps.org"
    payload = _varuint(len(uri)) + uri
    return b"\x00" + PENDING + _varuint(len(payload)) + payload


def _proof(digest: bytes, *kinds: str, height: int = 968300) -> bytes:
    """A well-formed version-1 proof: append a nonce, sha256, then one
    attestation per kind (forked with 0xff when there are several)."""
    tail = b""
    for kind in kinds[:-1]:
        tail += b"\xff" + _attestation(kind, height)
    tail += _attestation(kinds[-1], height)
    nonce = bytes(range(16))
    return aa.OTS_MAGIC + b"\x01\x08" + digest + b"\xf0" + _varuint(16) + nonce + b"\x08" + tail


def test_the_proof_parser_reads_every_committed_proof_and_names_z_s_block():
    z = REPO / "docs" / "research" / "ADDENDUM_Z_PROTOCOL.md.2026-09-23T0219Z.ots"
    # the digest quoted in anchor commit 7919d36, not the living file's
    assert aa.anchored_digest(z) == "511db57e530a6ebc07de18f2563629ce12b3b9b664cc822221f2a921f6c87cd1"
    # ANCHORS.md names block 968216 for Z; the proof itself says so
    assert aa.bitcoin_heights(z) == [968216]
    proofs = list((REPO / "docs" / "research").glob("*.ots")) + list((REPO / "experiments").glob("*.ots"))
    assert proofs and all(aa.parse_ots(p.read_bytes()) is not None for p in proofs)


def test_the_proof_parser_refuses_malformed_proofs(tmp_path):
    digest = bytes(32)
    good = _proof(digest, "bitcoin")
    cases = {
        "header_only": aa.OTS_MAGIC + b"\x01\x08" + digest,
        "truncated": good[:-2],
        "trailing_bytes": good + b"\x00",
        "version_2": aa.OTS_MAGIC + b"\x02\x08" + good[len(aa.OTS_MAGIC) + 2:],
        "not_sha256": aa.OTS_MAGIC + b"\x01\x02" + good[len(aa.OTS_MAGIC) + 2:],
        "unknown_op": aa.OTS_MAGIC + b"\x01\x08" + digest + b"\x99" + _attestation("bitcoin"),
        "garbage": b"not a proof",
    }
    for name, blob in cases.items():
        path = tmp_path / f"{name}.ots"
        path.write_bytes(blob)
        assert aa.anchored_digest(path) is None, name
    path = tmp_path / "good.ots"
    path.write_bytes(_proof(digest, "pending", "bitcoin", height=968301))
    assert aa.anchored_digest(path) == digest.hex()
    assert aa.attestation_kinds(path) == ["bitcoin", "pending"]
    assert aa.bitcoin_heights(path) == [968301]


# -- the anchor ------------------------------------------------------------------


def _anchor_fixture(root, monkeypatch, *, kinds=("bitcoin",), edit_after=b"", wrong_digest=False,
                    placeholder=False, drop=None, extra=None):
    """A git repository with every pinned file (synthetic bytes), a protocol
    quoting their digests and a stamp, its snapshot and a proof."""
    root.mkdir(parents=True, exist_ok=True)
    for rel in aa.PINNED.values():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic {rel}\n")
    research = root / "docs" / "research"
    research.mkdir(parents=True, exist_ok=True)
    name = "STAMP_PLACEHOLDER" if placeholder else STAMP
    lines = [f"Frozen. Anchored as `ADDENDUM_AA_PROTOCOL.md.{name}.ots`."]
    for rel in aa.PINNED.values():
        if rel == drop:
            continue
        digest = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        lines.append(f"- `{rel}` sha256 `{digest}`")
    if extra:
        lines.append(f"- `{extra}` sha256 `{'0' * 64}`")
    text = ("\n".join(lines) + "\n").encode()
    protocol = research / "ADDENDUM_AA_PROTOCOL.md"
    protocol.write_bytes(text + edit_after)
    (research / f"snapshots_ADDENDUM_AA_PROTOCOL_{STAMP}.md").write_bytes(text)
    committed = bytes(32) if wrong_digest else hashlib.sha256(text).digest()
    (research / f"ADDENDUM_AA_PROTOCOL.md.{STAMP}.ots").write_bytes(_proof(committed, *kinds))
    monkeypatch.setattr(aa, "REPO", root)
    monkeypatch.setattr(aa, "PROTOCOL_PATH", protocol)
    _git_repo(root)
    return root


def test_the_anchor_gate_passes_only_when_everything_agrees(tmp_path, monkeypatch):
    root = _anchor_fixture(tmp_path, monkeypatch)
    record = aa.anchor_record()
    assert record["ok"], record
    assert record["attestations"] == ["bitcoin"] and record["bitcoin_attested"]
    proof = root / "docs" / "research" / f"ADDENDUM_AA_PROTOCOL.md.{STAMP}.ots"
    # the record pins the proof it was gated on, and the block it names
    assert record["proof_sha256"] == hashlib.sha256(proof.read_bytes()).hexdigest()
    assert record["bitcoin_block_heights"] == [968300]
    assert set(record["pinned"]["quoted_in_anchored_protocol"]) == set(aa.PINNED.values())


def test_an_erratum_appended_later_does_not_break_the_gate(tmp_path, monkeypatch):
    _anchor_fixture(tmp_path, monkeypatch, edit_after=b"\n## Erratum 2026-10-30\n...\n")
    assert aa.anchor_record()["ok"]


@pytest.mark.parametrize("kwargs, why", [
    ({"kinds": ("pending",)}, "a calendar promise only"),
    ({"wrong_digest": True}, "the proof commits to other bytes"),
    ({"placeholder": True}, "the stamp placeholder was never replaced"),
    ({"drop": "src/arcttt/scoring.py"}, "a pinned file the protocol does not quote"),
    ({"extra": "tools/fencecheck.py"}, "a quoted file the reader does not pin"),
])
def test_the_anchor_gate_refuses(tmp_path, monkeypatch, kwargs, why):
    _anchor_fixture(tmp_path, monkeypatch, **kwargs)
    assert not aa.anchor_record()["ok"], why


def test_the_anchor_gate_refuses_an_in_place_edit_and_any_pinned_file_change(tmp_path, monkeypatch):
    root = _anchor_fixture(tmp_path / "first", monkeypatch)
    protocol = root / "docs" / "research" / "ADDENDUM_AA_PROTOCOL.md"
    protocol.write_bytes(protocol.read_bytes().replace(b"Frozen.", b"Frozen!"))
    assert not aa.anchor_record()["ok"]
    for rel in aa.PINNED.values():
        root = _anchor_fixture(tmp_path / rel.replace("/", "_"), monkeypatch)
        (root / rel).write_bytes((root / rel).read_bytes() + b"# edited after the freeze\n")
        pinned = aa.anchor_record()["pinned"]
        assert not pinned["matches"] and pinned["mismatched"] == [rel], rel


def test_quoted_digests_are_read_for_any_path_with_posix_keys():
    digest = "a" * 64
    text = (f"`src/arcttt/scoring.py` sha256 `{digest}` and "
            f"`scripts/aa_pinned/fencecheck_x.py` sha256 `{digest}` and "
            f"`tools/fencecheck.py` sha256 `{digest}`")
    quoted = aa.pinned_digests(text)["quoted_in_anchored_protocol"]
    assert set(quoted) == {"src/arcttt/scoring.py", "scripts/aa_pinned/fencecheck_x.py",
                           "tools/fencecheck.py"}
    for rel in aa.PINNED.values():
        assert "\\" not in rel and not rel.startswith("/") and pathlib.PurePosixPath(rel).as_posix() == rel


def test_the_live_fencecheck_is_not_pinned_and_the_vendored_one_is_x_s():
    assert "tools/fencecheck.py" not in aa.PINNED.values()
    vendored = REPO / aa.PINNED["FENCECHECK"]
    blob = _git(REPO, "hash-object", "--", aa.PINNED["FENCECHECK"]).strip()
    assert blob == aa.FENCECHECK_BLOB_AT_X_RUN
    at_x = subprocess.run(["git", "rev-parse", f"{aa.X_RUN_COMMIT}:tools/fencecheck.py"], cwd=REPO,
                          capture_output=True, text=True)
    if at_x.returncode != 0:
        pytest.skip("X's run commit is not in this clone's history")
    assert at_x.stdout.strip() == blob, "the vendored copy is not byte-identical to X's"
    assert vendored.read_bytes().startswith(b"#!/usr/bin/env python3")


def test_every_pinned_file_is_protected_from_line_ending_conversion():
    manifest = json.loads((REPO / aa.PINNED["INPUTS"]).read_text())["files"]
    for rel in [*aa.PINNED.values(), *manifest]:
        attr = _git(REPO, "check-attr", "text", "--", rel)
        assert attr.strip().endswith("text: unset"), attr


# -- the real protocol, in each of its states ------------------------------------


def _state():
    """draft (placeholders), prepared (stamp quoted, no proof) or stamped."""
    text = aa.PROTOCOL_PATH.read_text()
    stamp = aa.stamp_of(text)
    if stamp is None:
        return "draft", None
    proof, _ = aa.proof_and_snapshot(stamp)
    return ("stamped" if proof.exists() else "prepared"), stamp


def test_the_real_protocol_quotes_its_stamp_once_it_is_frozen():
    state, stamp = _state()
    text = aa.PROTOCOL_PATH.read_text()
    if state == "draft":
        assert "STAMP_PLACEHOLDER" in text  # not yet frozen: nothing may be read
        assert not list(aa.PROTOCOL_PATH.parent.glob("ADDENDUM_AA_PROTOCOL.md.*.ots"))
        return
    assert "STAMP_PLACEHOLDER" not in text and "_SHA_PLACEHOLDER" not in text
    proof, snapshot = aa.proof_and_snapshot(stamp)
    assert snapshot.exists() and aa.PROTOCOL_PATH.read_bytes().startswith(snapshot.read_bytes())
    if state == "prepared":
        # the one commit between `prepare` and `stamp`: allowed for a day
        made = datetime.strptime(stamp, "%Y-%m-%dT%H%MZ").replace(tzinfo=timezone.utc)
        assert datetime.now(timezone.utc) - made < timedelta(hours=24), (
            f"prepared at {stamp} and never stamped: run freeze_protocol.py stamp")
        return
    assert proof.exists()


def test_the_real_protocol_pins_exactly_the_readers_list():
    state, stamp = _state()
    if state == "draft":
        text = aa.PROTOCOL_PATH.read_text()
        for name, rel in aa.PINNED.items():
            assert f"`{rel}` sha256 `{name}_SHA_PLACEHOLDER`" in text, rel
        assert len(re.findall(r"[A-Z_]+_SHA_PLACEHOLDER", text)) == len(aa.PINNED)
        return
    snapshot = aa.proof_and_snapshot(stamp)[1].read_text()
    pinned = aa.pinned_digests(snapshot)
    assert set(pinned["quoted_in_anchored_protocol"]) == set(aa.PINNED.values())
    if not aa.banked_readings():
        # until the reading is banked, a change to any pinned file voids it
        assert pinned["matches"], pinned["mismatched"]


def test_the_freeze_command_names_exactly_the_pinned_files():
    doc = (REPO / "scripts" / "freeze_protocol.py").read_text()
    named = re.findall(r"--code (\S+)", doc)
    assert sorted(named) == sorted(aa.PINNED.values())


def test_the_manifest_lists_exactly_the_inputs_the_reading_opens():
    manifest = json.loads((REPO / aa.PINNED["INPUTS"]).read_text())
    assert manifest["x_run_commit"] == aa.X_RUN_COMMIT
    expected = [aa._rel(p) for p in aa.input_paths(x._scope())]
    assert list(manifest["files"]) == expected
    assert all(re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) for f in manifest["files"].values())
    # the gate cells are not read: X's artifact supplies the gate status
    assert not any(k.endswith("_determinism.json") for k in manifest["files"])


def test_the_reading_loads_no_repository_module_the_anchor_does_not_pin():
    probe = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['transformers'] = None\n"
        f"sys.path.insert(0, {str(REPO / 'scripts')!r})\n"
        "import decoder_isolation_read as aa\n"
        "import arcttt.scoring, arcttt.text_task\n"
        "aa.pinned_fencecheck()\n"
        "print(aa.loaded_but_unpinned())\n")
    out = subprocess.run([sys.executable, "-c", probe], cwd=REPO / "tests", capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "[]", out


def test_loaded_but_unpinned_names_a_stray_repository_module(tmp_path, monkeypatch):
    monkeypatch.setattr(aa, "REPO", tmp_path)
    stray = type(sys)("aa_test_stray")
    stray.__file__ = str(tmp_path / "scripts" / "stray.py")
    monkeypatch.setitem(sys.modules, "aa_test_stray", stray)
    assert aa.loaded_but_unpinned() == ["scripts/stray.py"]


# -- preconditions, --out and the once-only write --------------------------------


@pytest.fixture
def gated(tmp_path, monkeypatch):
    """A repository root for the reader with an anchor that holds, a clean
    tree, and X's gate as X's artifact banks it."""
    (tmp_path / "experiments").mkdir()
    protocol = tmp_path / "ADDENDUM_AA_PROTOCOL.md"
    protocol.write_text("synthetic\n")
    monkeypatch.setattr(aa, "REPO", tmp_path)
    monkeypatch.setattr(aa, "PROTOCOL_PATH", protocol)
    monkeypatch.setattr(aa, "anchor_record", lambda: {"ok": True, "stamp": STAMP})
    monkeypatch.setattr(aa, "uncommitted", lambda rels: [])
    monkeypatch.setattr(aa, "input_paths", lambda scope: [])
    monkeypatch.setattr(aa, "x_gates", lambda: {f: {"n": 0 if f == "granite-2b" else 5,
                                                    "mismatches": 0, "passed": True}
                                                for f in aa.FAMILIES})
    return tmp_path


def _bank(root, date="2026-10-22"):
    path = root / "experiments" / f"cord_decoder_isolation_read_{date}.json"
    path.write_text("{}")
    return path


def test_a_recomputation_is_refused_until_the_banked_reading_exists(gated):
    out = gated / "mine.json"
    problems = aa.preconditions(x._scope(), out, banked_reading=False)
    assert any("does not exist yet" in p for p in problems)
    _bank(gated)
    assert aa.preconditions(x._scope(), out, banked_reading=False) == []


def test_a_recomputation_needs_the_same_anchor_and_clean_tree(gated, monkeypatch):
    _bank(gated)
    monkeypatch.setattr(aa, "anchor_record", lambda: {"ok": False, "why": "not attested", "stamp": STAMP})
    assert any(p.startswith("anchor:") for p in aa.preconditions(x._scope(), gated / "m.json", False))
    monkeypatch.setattr(aa, "anchor_record", lambda: {"ok": True, "stamp": STAMP})
    monkeypatch.setattr(aa, "uncommitted", lambda rels: ["src/arcttt/scoring.py"])
    problems = aa.preconditions(x._scope(), gated / "m.json", False)
    assert any("not committed" in p and "scoring.py" in p for p in problems)


def test_the_banked_reading_is_made_once_whatever_its_date(gated):
    _bank(gated, "2026-10-22")
    target = aa.banked_path("2026-11-05")
    problems = aa.preconditions(x._scope(), target, banked_reading=True)
    assert any("already banked" in p for p in problems)


def test_out_refuses_every_spelling_of_a_banked_path(gated, monkeypatch):
    banked = _bank(gated)
    (gated / "elsewhere").mkdir()
    os.symlink(gated / "experiments", gated / "elsewhere" / "link")
    monkeypatch.chdir(gated)
    spellings = [banked, pathlib.Path("experiments") / banked.name,
                 pathlib.Path("./experiments/../experiments") / banked.name,
                 gated / "elsewhere" / "link" / banked.name,
                 gated / "experiments" / "cord_decoder_isolation_read_2026-12-01.json",
                 gated / "elsewhere" / "link" / "new.json"]
    for out in spellings:
        assert aa.out_path_problems(out), out
    (gated / "elsewhere" / "taken.json").write_text("x")
    assert any("exists" in p for p in aa.out_path_problems(gated / "elsewhere" / "taken.json"))
    assert aa.out_path_problems(gated / "elsewhere" / "fresh.json") == []


def test_check_opens_no_cell_before_the_anchor_holds(gated, monkeypatch):
    monkeypatch.setattr(aa, "anchor_record", lambda: {"ok": False, "why": "no stamp", "stamp": None})

    def forbidden(*a, **k):
        raise AssertionError("a cell was looked at before the anchor held")

    monkeypatch.setattr(aa, "x_gates", forbidden)
    monkeypatch.setattr(aa, "input_paths", forbidden)
    problems = aa.preconditions(x._scope(), aa.banked_path("2026-10-22"), banked_reading=True)
    assert problems and problems[0].startswith("anchor:")


def test_uncommitted_names_untracked_and_dirty_files(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("a\n")
    (tmp_path / "b.py").write_text("b\n")
    _git_repo(tmp_path)
    monkeypatch.setattr(aa, "REPO", tmp_path)
    assert aa.uncommitted(["a.py", "b.py"]) == []
    (tmp_path / "b.py").write_text("b, edited\n")
    (tmp_path / "c.py").write_text("c\n")
    assert aa.uncommitted(["a.py", "b.py", "c.py"]) == ["b.py", "c.py"]


def test_gate_status_reads_x_s_artifact_and_names_granite_not_applicable():
    scope = x._scope()
    granite = aa.gate_status("granite-2b", scope)
    assert granite["status"] == "NOT APPLICABLE" and granite["n"] == 0
    # X's artifact banks passed = true; this reading never calls that a pass
    assert granite["x_artifact_banks"] == {"n": 0, "mismatches": 0, "passed": True}
    assert granite["not_applicable_because"]["chat_template_clock_calls"] == ["strftime_now"]
    for family in ("qwen2.5-0.5b", "smollm2-1.7b", "falcon3-1b"):
        assert aa.gate_status(family, scope)["status"] == "PASSED"
    assert aa.gate_status("phi3-mini", scope)["status"] == "FAILED"


def test_the_input_manifest_is_checked_byte_for_byte(tmp_path, monkeypatch):
    for name in ("a.json", "b.jsonl"):
        (tmp_path / name).write_text(name)
    manifest = {"files": {n: {"sha256": hashlib.sha256(n.encode()).hexdigest()}
                          for n in ("a.json", "b.jsonl")}}
    (tmp_path / "scripts" / "aa_pinned").mkdir(parents=True)
    (tmp_path / aa.PINNED["INPUTS"]).write_text(json.dumps(manifest))
    monkeypatch.setattr(aa, "REPO", tmp_path)
    monkeypatch.setattr(aa, "input_paths", lambda scope: [tmp_path / "a.json", tmp_path / "b.jsonl"])
    assert aa.inputs_match_the_manifest({})[0] == []
    (tmp_path / "b.jsonl").write_text("b, edited")
    assert aa.inputs_match_the_manifest({})[0] == ["b.jsonl: differs from the manifest"]
    monkeypatch.setattr(aa, "input_paths", lambda scope: [tmp_path / "a.json"])
    assert any("differ" in p for p in aa.inputs_match_the_manifest({})[0])


def test_write_manifest_refuses_an_input_that_differs_from_x_s_commit(tmp_path, monkeypatch, capsys):
    (tmp_path / "in.json").write_text("v1\n")
    _git_repo(tmp_path)
    first = _git(tmp_path, "rev-parse", "HEAD").strip()
    monkeypatch.setattr(aa, "REPO", tmp_path)
    monkeypatch.setattr(aa, "X_RUN_COMMIT", first)
    monkeypatch.setattr(aa.x, "_scope", lambda: {})
    monkeypatch.setattr(aa, "input_paths", lambda scope: [tmp_path / "in.json"])
    assert aa.write_manifest() == 0
    written = json.loads((tmp_path / aa.PINNED["INPUTS"]).read_text())
    assert written["files"]["in.json"]["sha256"] == hashlib.sha256(b"v1\n").hexdigest()
    assert aa.write_manifest() == 2  # written once
    (tmp_path / aa.PINNED["INPUTS"]).unlink()
    (tmp_path / "in.json").write_text("v2\n")
    _git(tmp_path, "commit", "-qam", "changed after X")
    assert aa.write_manifest() == 2 and "not byte-identical" in capsys.readouterr().out


def test_the_exclusive_write_never_replaces_a_file_and_leaves_no_temporary(tmp_path):
    target = tmp_path / "out.json"
    x.write_new_file(target, "first\n")
    with pytest.raises(FileExistsError):
        x.write_new_file(target, "second\n")
    assert target.read_text() == "first\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.json"]


# -- X's moved code on synthetic cells, the EOS erratum included -----------------


def _cells(out_dir, name, payload):
    (out_dir / name).write_text(json.dumps(payload))


@pytest.fixture
def synthetic_x(monkeypatch, tmp_path):
    out_dir, split, v_cells = tmp_path / "cells", tmp_path / "split", tmp_path / "v_cells"
    for d in (out_dir, split, v_cells):
        d.mkdir()
    monkeypatch.setattr(x, "REPO", tmp_path)
    monkeypatch.setattr(x, "OUT_DIR", out_dir)
    monkeypatch.setattr(x, "V_CELLS", v_cells)
    monkeypatch.setattr(x.cft, "SPLIT_DIR", split)
    monkeypatch.setattr(x, "PLAIN_LOOP_EOS_ID", {"fam": 9, "qfam": 9})
    return {"root": tmp_path, "out_dir": out_dir, "split": split, "v_cells": v_cells}


def _gold(split, gold):
    (split / "gold.jsonl").write_text("".join(json.dumps({"id": d, "gold": g}) + "\n" for d, g in gold.items()))
    (split / "train.jsonl").write_text("")


def _token_family(env, monkeypatch):
    docs = ["d1", "d2", "d3", "d4", "d5"]
    _gold(env["split"], {d: {"a": i} for i, d in enumerate(docs, 1)})
    same = {d: "h" for d in docs}
    plain = {"predictions": {"d1": '{"a": 1}', "d2": "not json", "d3": '{"a": 2}', "d4": "nope",
                             "d5": "cap"},
             "per_document_prompt_sha256_16": same,
             "per_document_token_ids": {"d1": [1, 2, 3], "d2": [5, 6], "d3": [7, 8], "d4": [5, 6],
                                        "d5": [5, 10]},
             "per_document_decode": {"d1": {"stopped_on": "complete"}, "d2": {"stopped_on": "eos"},
                                     "d3": {"stopped_on": "complete"}, "d4": {"stopped_on": "eos"},
                                     "d5": {"stopped_on": "max_new_tokens"}}}
    const = {"predictions": dict(plain["predictions"], d3='{"a": 3}'),
             "per_document_prompt_sha256_16": same,
             "per_document_decode": {d: {"constrained_steps": 1 if d == "d3" else 0} for d in docs}}
    gen = {"predictions": dict(plain["predictions"], d3='{"b": 1}'),
           "per_document_prompt_sha256_16": same,
           # d1: generate runs on past the closing brace, then stops: agreement
           # d2: our loop stopped on its EOS (9); generate kept the same stop token: the erratum
           # d3: a real divergence at index 1
           # d4: our loop stopped on 9; generate stopped on 10, another id in ITS stop set:
           #     a different token at the stop position, which must stay a divergence
           # d5: both emitted 10 as their last token, ours as an ordinary token at the cap,
           #     generate as its stop: identical ids, agreement, and nothing is dropped
           "per_document_token_ids": {"d1": [1, 2, 3, 4, 9], "d2": [5, 6, 9], "d3": [7, 1, 9],
                                      "d4": [5, 6, 10], "d5": [5, 10]},
           "effective_generation_config": {"eos_token_id": [9, 10]}}
    _cells(env["out_dir"], "fam_plain.json", plain)
    _cells(env["out_dir"], "fam_constrained.json", const)
    _cells(env["out_dir"], "fam_generate_neutral.json", gen)
    (env["root"] / "g0.json").write_text(json.dumps({"predictions": {d: "" for d in docs}}))
    monkeypatch.setattr(x, "CELLS", {"fam": ("m", "float32", env["root"] / "g0.json")})
    return {"fam": {"arm_C_reused_from_V": False, "arm_G_runs": True, "modifiers": {}}}


def test_read_families_on_the_token_basis_strips_only_our_own_stop_id(synthetic_x, monkeypatch):
    scope = _token_family(synthetic_x, monkeypatch)
    rows, x1, x2, x3, mismatches = x.read_families(["fam"], scope, fencecheck=aa.pinned_fencecheck())
    (row,) = rows
    assert mismatches == [] and x3 is None
    assert row["x1_identical_documents"] == 4 and row["x1_differing_documents"] == ["d3"]
    assert row["accounting_consistent"]  # d3 differs and carries a constrained step
    assert row["x2_basis"] == "token ids"
    # as first coded: d2 (generate kept our stop token), d3 and d4 diverge
    assert row["x2_divergent_documents_as_first_coded"] == 3
    # corrected: d2 agrees; d4 (a different stop id) and d3 still diverge; d5 never did
    assert row["x2_first_divergence_index"] == {"d3": 1, "d4": 2}
    assert row["x2_documents_changed_by_the_eos_erratum"] == ["d2"]
    assert row["x2_eos_erratum_stripped_ids"] == {"d2": 9}
    # both counts are in the reading string itself
    assert x2["fam"] == ("PATH DIVERGES (2) (EOS erratum 2026-09-23: 2 documents diverge as "
                         "corrected, 3 as first coded)")


def test_the_generate_stop_is_stripped_once_and_only_where_our_loop_stopped_on_it():
    assert x._without_generate_stop([1, 2, 9], 9, "eos") == [1, 2]
    assert x._without_generate_stop([1, 9, 9], 9, "eos") == [1, 9]
    assert x._without_generate_stop([1, 2, 10], 9, "eos") == [1, 2, 10]   # another stop id
    assert x._without_generate_stop([1, 2, 9], 9, "complete") == [1, 2, 9]
    assert x._without_generate_stop([1, 2, 9], 9, "max_new_tokens") == [1, 2, 9]
    assert x._without_generate_stop([], 9, "eos") == []


def test_read_families_refuses_a_token_basis_family_without_a_pinned_stop_id(synthetic_x, monkeypatch):
    scope = _token_family(synthetic_x, monkeypatch)
    monkeypatch.setattr(x, "PLAIN_LOOP_EOS_ID", {})
    with pytest.raises(SystemExit):
        x.read_families(["fam"], scope, fencecheck=aa.pinned_fencecheck())


def test_read_families_on_the_text_basis(synthetic_x, monkeypatch):
    docs = ["t1", "t2", "t3"]
    _gold(synthetic_x["split"], {d: {"a": 1} for d in docs})
    plain = {"predictions": {"t1": '{"a": 1}', "t2": "abc", "t3": '{"a": 3}'},
             "per_document_decode": {"t1": {"stopped_on": "complete"}, "t2": {"stopped_on": "eos"},
                                     "t3": {"stopped_on": "complete"}}}
    # V's reused arm C banks no prompt digest: C vs P is not checked
    const = {"predictions": dict(plain["predictions"]),
             "per_document_decode": {d: {"constrained_steps": 0} for d in docs}}
    g0 = {"predictions": {"t1": '{"a": 1} and more', "t2": "abcd", "t3": ' {"b": 1} '}}
    _cells(synthetic_x["out_dir"], "tfam_plain.json", plain)
    _cells(synthetic_x["v_cells"], "tfam_schema_constrained.json", const)
    (synthetic_x["root"] / "g0.json").write_text(json.dumps(g0))
    monkeypatch.setattr(x, "CELLS", {"tfam": ("m", "float32", synthetic_x["root"] / "g0.json")})
    scope = {"tfam": {"arm_C_reused_from_V": True, "arm_G_runs": False, "modifiers": {}}}
    rows, x1, x2, x3, mismatches = x.read_families(["tfam"], scope, fencecheck=aa.pinned_fencecheck())
    (row,) = rows
    assert mismatches == [] and x3 is None
    assert row["x2_basis"] == "decoded text" and row["x2_compared_against"] == "G0"
    # t1: ours shorter and a prefix, agreement; t2: ours took EOS, G0 ran on; t3: differs at 2
    assert row["x2_first_divergence_index"] == {"t2": 3, "t3": 2}
    assert row["x2_divergent_documents_as_first_coded"] == 2
    assert row["x2_documents_changed_by_the_eos_erratum"] == [] and row["x2_eos_erratum_stripped_ids"] == {}
    assert x2["tfam"] == "PATH DIVERGES (2) (WEAKER TEST: decoded text, not token ids)"
    assert x1["tfam"] == "CONSTRAINT INERT"


def _x3_family(env, monkeypatch, v_published):
    docs = ["q1", "q2", "q3"]
    _gold(env["split"], {d: {"a": 1} for d in docs})
    same = {d: "h" for d in docs}
    plain = {"predictions": {"q1": "bad", "q2": '{"a": 1}', "q3": '{"a": 1}'},
             "per_document_prompt_sha256_16": same,
             "per_document_token_ids": {"q1": [1], "q2": [2], "q3": [3]},
             "per_document_decode": {d: {"stopped_on": "complete"} for d in docs}}
    # V's constrained cell: invalid on q1 and q2, where G0 is valid: two of V's regressions
    const = {"predictions": {"q1": "bad", "q2": "also bad", "q3": '{"a": 1}'},
             "per_document_decode": {"q1": {"constrained_steps": 0}, "q2": {"constrained_steps": 1},
                                     "q3": {"constrained_steps": 0}}}
    g0 = {"predictions": {"q1": '{"a": 1}', "q2": '{"a": 1}', "q3": '{"a": 1}'}}
    # without the defaults: q1 still invalid (explained by the defaults), q2 valid (residual)
    gen = {"predictions": {"q1": "bad", "q2": '{"a": 1}', "q3": '{"a": 1}'},
           "per_document_prompt_sha256_16": same,
           "per_document_token_ids": {"q1": [1], "q2": [2], "q3": [3]},
           "effective_generation_config": {"eos_token_id": [9]}}
    _cells(env["out_dir"], "qfam_plain.json", plain)
    _cells(env["out_dir"], "qfam_generate_neutral.json", gen)
    _cells(env["v_cells"], "qfam_schema_constrained.json", const)
    (env["root"] / "g0.json").write_text(json.dumps(g0))
    v_artifact = env["root"] / "v.json"
    v_artifact.write_text(json.dumps({"rows": [{"family": "qfam", "regressions": v_published}]}))
    monkeypatch.setattr(x, "V_ARTIFACT", v_artifact)
    monkeypatch.setattr(x, "CELLS", {"qfam": ("m", "float32", env["root"] / "g0.json")})
    return {"qfam": {"arm_C_reused_from_V": True, "arm_G_runs": True,
                     "modifiers": {"temperature": 0.7}}}


def test_read_families_x3_when_v_s_count_reproduces(synthetic_x, monkeypatch):
    scope = _x3_family(synthetic_x, monkeypatch, v_published=2)
    rows, x1, x2, x3, mismatches = x.read_families(["qfam"], scope, fencecheck=aa.pinned_fencecheck())
    assert x3["v_regression_count_reproduces"] and x3["v_regressions_recomputed_here"] == 2
    assert x3["regressions_explained_by_the_defaults"] == 1 and x3["residual_documents"] == ["q2"]
    assert x3["reading"].startswith("V'S ATTRIBUTION IS TOO STRONG: 1 of 2")
    # q1: zero steps, but arms C and P agree; q2: steps and differing arms; consistent
    assert rows[0]["accounting_consistent"]


def test_read_families_x3_when_v_s_count_does_not_reproduce(synthetic_x, monkeypatch):
    scope = _x3_family(synthetic_x, monkeypatch, v_published=3)
    rows, x1, x2, x3, mismatches = x.read_families(["qfam"], scope, fencecheck=aa.pinned_fencecheck())
    assert x3["v_regression_count_reproduces"] is False and x3["reading"].startswith("WITHHELD")
    banked = aa.x3_as_banked(x3, {})
    for key in aa.X3_WITHHELD_KEYS:
        assert banked[key] is None, key  # nothing the explained count can be rebuilt from
    assert banked["v_regressions_recomputed_here"] == 2 and banked["v_published_regressions"] == 3


def test_read_families_only_reads_the_families_asked_for(synthetic_x, monkeypatch):
    scope = _token_family(synthetic_x, monkeypatch)
    rows, *_ = x.read_families([], scope, fencecheck=aa.pinned_fencecheck())
    assert rows == []


def test_the_reading_never_loads_the_live_fencecheck(synthetic_x, monkeypatch):
    scope = _token_family(synthetic_x, monkeypatch)
    with aa._live_fencecheck_refused():
        with pytest.raises(SystemExit):
            x.read_families(["fam"], scope)  # no copy passed: X's default is the live tool
        rows, *_ = x.read_families(["fam"], scope, fencecheck=aa.pinned_fencecheck())
    assert rows


def test_no_reading_takes_decode_seconds_as_an_input():
    """The 2026-10-21 disclosure says the per_document_seconds field cannot
    bear on X1-X3. That holds only while the reading code never reads it."""
    source = inspect.getsource(x.read_families)
    assert "seconds" not in source
    assert "per_document_seconds" in aa.SECONDS_DISCLOSURE


# -- the readings layered on X's -------------------------------------------------


def test_the_unscoped_x1_holds_sentence_is_unreachable():
    four = {f: "CONSTRAINT REMOVES" for f in aa.FAMILIES}
    finding = aa.scoped_x1_finding(four)
    assert finding.startswith("X1 HOLDS ON THE FAMILIES READ ONLY: CONSTRAINT REMOVES in 4 of 5")
    assert "unverified on 1 of 5" in finding and "is NOT licensed" in finding
    assert "no family is an exception" not in finding
    sentences = aa.licensed_sentences(_rows(), finding, None, {}, x._scope())
    assert not any("no family is an exception" in s for s in sentences)
    three = dict(four) | {"granite-2b": "CONSTRAINT INERT"}
    assert aa.scoped_x1_finding(three).startswith("X1 MIXED: CONSTRAINT REMOVES in 3 of 5")
    hurts = {f: "CONSTRAINT INERT" for f in aa.FAMILIES}
    hurts["falcon3-1b"] = "CONSTRAINT HURTS (invalid 1 -> 3, regressions 2)"
    failed = aa.scoped_x1_finding(hurts)
    assert failed.startswith("X1 FAILS IN falcon3-1b") and "families tested" not in failed


def test_a_void_family_counts_against_removes_and_leaves_the_veto_unverified():
    readings = {f: "CONSTRAINT REMOVES" for f in aa.FAMILIES}
    readings["granite-2b"] = "VOID (prompt identity MISMATCHED on C vs P)"
    finding = aa.scoped_x1_finding(readings)
    # three REMOVES at most: with a VOID family X1 cannot read HOLDS
    assert finding.startswith("X1 MIXED: CONSTRAINT REMOVES in 3 of 5")
    assert '"granite-2b": "VOID' in finding
    exception = dict(readings) | {"granite-2b": "VOID (x)", "falcon3-1b": "CONSTRAINT HURTS (1 -> 3)"}
    assert "3 read, 1 VOID (granite-2b); phi3-mini not read" in aa.scoped_x1_finding(exception)


def _row(family, x1="CONSTRAINT INERT", x2="PATH REPRODUCES", basis="token ids",
         consistent=True, differing=(), first_coded=0, changed=()):
    return {"family": family, "model": "m", "n": 50, "x1_reading": x1, "x2_reading": x2,
            "x1_invalid_plain": 0, "x1_invalid_constrained": 0, "x2_basis": basis,
            "x2_invalid_plain": 3, "x2_invalid_other_arm": 5, "x2_compared_against": "G0",
            "accounting_consistent": consistent, "x1_differing_documents": list(differing),
            "documents_with_zero_steps_but_differing_arms": [] if consistent else ["cord-001"],
            "documents_with_steps_but_identical_arms": [],
            "x2_divergent_documents_as_first_coded": first_coded, "x2_divergent_documents": 0,
            "x2_documents_changed_by_the_eos_erratum": list(changed),
            "x2_eos_erratum_stripped_ids": {d: 151645 for d in changed}}


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
    x_protocol = " ".join((REPO / "docs" / "research" / "ADDENDUM_X_PROTOCOL.md").read_text().split())
    # the full sentence, word for word, in the reader and in X's own module
    assert " ".join(checks["verbatim"]["X2"].split()) in x_protocol
    import ast
    strings = [n.value for n in ast.walk(ast.parse((REPO / "scripts" / "cord_decoder_isolation.py").read_text()))
               if isinstance(n, ast.Constant) and isinstance(n.value, str) and "Granite because" in n.value]
    assert strings and all(s.removeprefix("X2: ") == aa.X_PREDICTIONS_VERBATIM["X2"] for s in strings)


def test_prediction_checks_mark_a_void_family_void():
    void = {"granite-2b": {"reason": "prompt identity MISMATCHED on C vs P"}}
    checks = aa.prediction_checks(_rows(), void)
    assert checks["x1_inert_holds"]["granite-2b"].startswith("VOID")
    assert checks["x1_inert_where_predicted"]["granite-2b"].startswith("VOID")
    assert checks["x1_accounting_per_document"]["granite-2b"].startswith("VOID")
    assert checks["x1_inert_holds"]["qwen2.5-0.5b"] is True


def test_licensed_sentences_carry_the_caveats_and_void_a_family():
    scope = x._scope()
    void = {"granite-2b": {"reason": "prompt identity MISMATCHED on C vs P"}}
    rows = _rows(**{"granite-2b": {"x1_reading": "CONSTRAINT HURTS (invalid 1 -> 3, regressions 3)"},
                    "qwen2.5-0.5b": {"x2_divergent_documents_as_first_coded": 1,
                                     "x2_documents_changed_by_the_eos_erratum": ["cord-007"]}})
    sentences = aa.licensed_sentences(rows, "X1 MIXED: ... Per-family: {}", None, void, scope)
    text = " ".join(sentences)
    assert "granite-2b: VOID" in text and "granite-2b (X2)" not in text and "granite-2b (X1)" not in text
    assert "HURTS" not in text  # a VOID family's raw HURTS is never named
    qwen = [s for s in sentences if s.startswith("qwen2.5-0.5b (X2)")][0]
    assert "token-for-token" in qwen
    # both counts, inline, with what X's code at b4facac reads
    assert "0 of 50 documents diverge" in qwen and "b4facac reads PATH DIVERGES (1)" in qwen
    assert "cord-007" in qwen
    smol = [s for s in sentences if s.startswith("smollm2-1.7b (X2)")][0]
    assert "WEAKER TEST" in smol and "token-for-token" not in smol
    falcon = [s for s in sentences if s.startswith("falcon3-1b (X2)")][0]
    assert "no environment record" in falcon and "invalid outputs 3" in falcon
    falcon_x1 = [s for s in sentences if s.startswith("falcon3-1b (X1)")][0]
    assert "cannot separate the two" in falcon_x1  # arm C is V's reused cell
    # the combined line names the reused-arm family it credits, with the same caveat
    assert "In falcon3-1b, arm C is V's banked cell" in sentences[0]
    assert "cannot separate the two" in sentences[0]
    assert "phi3" not in " ".join(s for s in sentences if not s.startswith("X1 (combined")).lower()


def test_obligations_fire_from_the_readings_and_never_from_a_void_family():
    owed = aa.obligations(_rows(), None)
    assert [o["file"] for o in owed] == ["tools/README.md"]
    hurts = _rows(**{"falcon3-1b": {"x1_reading": "CONSTRAINT HURTS (invalid 1 -> 3, regressions 2)"}})
    x3 = {"family": "qwen2.5-0.5b", "reading": "V'S ATTRIBUTION IS TOO STRONG: 3 of 5 ..."}
    files = [o["file"] for o in aa.obligations(hurts, x3)]
    assert files.count("tools/README.md") == 2
    assert any(f.startswith("VERDICT.md") for f in files) and any(f.startswith("CORRECTIONS.md") for f in files)
    void = {"falcon3-1b": {"reason": "r"}, "qwen2.5-0.5b": {"reason": "r"}}
    assert [o["file"] for o in aa.obligations(hurts, x3, void)] == ["tools/README.md"]


def test_x3_of_a_void_family_keeps_only_the_check_on_v_s_own_count():
    x3 = {"family": "qwen2.5-0.5b", "reading": "WITHHELD: ...", "v_regression_count_reproduces": False,
          "v_published_regressions": 5, "v_regressions_recomputed_here": 4, "note_if_v_does_not_reproduce": "n",
          "regressions_explained_by_the_defaults": 2, "residual_documents": ["a"],
          "residual_attributed_by_arithmetic": {"a": "x"}, "v_regressions_reexamined": [{"id": "a"}],
          "documents_differing_G_vs_G0": 9}
    void = {"qwen2.5-0.5b": {"reason": "prompt identity MISMATCHED on P vs G"}}
    banked = aa.x3_as_banked(x3, void)
    assert banked["reading"].startswith("VOID")
    assert set(banked) == {"family", "reading", *aa.X3_V_COUNT_KEYS}
    reproducing = dict(x3, v_regression_count_reproduces=True)
    assert set(aa.x3_as_banked(reproducing, void)) == {"family", "reading"}


def test_read_withholds_and_writes_nothing_when_a_precondition_fails(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(aa, "REPO", tmp_path)
    monkeypatch.setattr(aa, "preconditions", lambda scope, out, banked: ["anchor: not attested"])

    def forbidden(*a, **k):
        raise AssertionError("the reading was computed before its preconditions held")

    monkeypatch.setattr(aa.x, "read_families", forbidden)
    assert aa.read(tmp_path / "out.json") == 2
    assert "WITHHELD" in capsys.readouterr().out
    assert not (tmp_path / "out.json").exists()


# -- the reader end to end, on synthetic cells for the four families --------------


@pytest.fixture
def four_families(tmp_path, monkeypatch):
    """Synthetic cells for the four real family names, shaped like the real
    scope: Qwen token basis with X3, SmolLM2 and Falcon3 text basis with V's
    reused arm C, Granite token basis with its own arm C. Granite's C vs P
    prompt digests disagree on one document, so Granite is VOID, and its raw
    X1 is HURTS. Qwen's V count does not reproduce, and one Qwen document
    changes under the EOS erratum. SmolLM2's accounting is wrong. Falcon3
    reads REMOVES with differing arms."""
    root = tmp_path
    cells, v_cells, split = root / "cells", root / "v_cells", root / "split"
    for d in (cells, v_cells, split, root / "experiments"):
        d.mkdir()
    docs = ["d1", "d2", "d3"]
    _gold(split, {d: {"a": 1} for d in docs})
    ok, bad = '{"a": 1}', "not json"
    same = {d: "h" for d in docs}
    scope = {
        "qwen2.5-0.5b": {"arm_C_reused_from_V": True, "arm_G_runs": True, "modifiers": {"temperature": 0.7}},
        "smollm2-1.7b": {"arm_C_reused_from_V": True, "arm_G_runs": False, "modifiers": {}},
        "granite-2b": {"arm_C_reused_from_V": False, "arm_G_runs": True, "modifiers": {},
                       "chat_template_clock_calls": ["strftime_now"]},
        "falcon3-1b": {"arm_C_reused_from_V": True, "arm_G_runs": False, "modifiers": {}},
    }

    def plain(preds, stops, ids=None):
        out = {"predictions": dict(zip(docs, preds)), "per_document_prompt_sha256_16": same,
               "per_document_decode": {d: {"stopped_on": s} for d, s in zip(docs, stops)}}
        if ids:
            out["per_document_token_ids"] = dict(zip(docs, ids))
        return out

    def const(preds, steps, digests=None):
        out = {"predictions": dict(zip(docs, preds)),
               "per_document_decode": {d: {"constrained_steps": s} for d, s in zip(docs, steps)}}
        if digests:
            out["per_document_prompt_sha256_16"] = digests
        return out

    # Qwen: d2 stopped on EOS, generate kept 151645 (the erratum); V's const fails on d3, where
    # G0 is valid (one regression: X1 reads MIXED, and V's published 4 does not reproduce)
    _cells(cells, "qwen2.5-0.5b_plain.json", plain([bad, "nope", ok], ["complete", "eos", "complete"],
                                                  [[1], [2, 3], [4]]))
    _cells(v_cells, "qwen2.5-0.5b_schema_constrained.json", const([ok, ok, bad], [1, 1, 1]))
    _cells(cells, "qwen2.5-0.5b_generate_neutral.json", {
        "predictions": dict(zip(docs, [ok, ok, ok])), "per_document_prompt_sha256_16": same,
        "per_document_token_ids": {"d1": [1, 5], "d2": [2, 3, 151645], "d3": [4]}})
    # SmolLM2: d1 has 0 steps but its arms differ -- the accounting is wrong
    _cells(cells, "smollm2-1.7b_plain.json", plain([ok, ok, ok], ["complete"] * 3))
    _cells(v_cells, "smollm2-1.7b_schema_constrained.json", const(['{"a": 2}', ok, ok], [0, 0, 0]))
    # Granite: three regressions (HURTS, raw), and a prompt digest mismatch on d3 (VOID)
    _cells(cells, "granite-2b_plain.json", plain([ok, ok, ok], ["complete"] * 3, [[1], [2], [3]]))
    _cells(cells, "granite-2b_constrained.json",
           const([bad, bad, bad], [1, 1, 1], {"d1": "h", "d2": "h", "d3": "OTHER"}))
    _cells(cells, "granite-2b_generate_neutral.json", {
        "predictions": dict(zip(docs, [ok, ok, ok])), "per_document_prompt_sha256_16": same,
        "per_document_token_ids": {"d1": [1], "d2": [2], "d3": [3]}})
    # Falcon3: plain invalid on two documents the validator fixed: REMOVES, arms differ
    _cells(cells, "falcon3-1b_plain.json", plain([bad, "nope", ok], ["complete"] * 3))
    _cells(v_cells, "falcon3-1b_schema_constrained.json", const([ok, ok, ok], [1, 1, 0]))
    comparators = {}
    for family in aa.FAMILIES:
        comparators[family] = root / f"{family}_g0.json"
        comparators[family].write_text(json.dumps({"predictions": {d: ok for d in docs}}))
    v_artifact = root / "v.json"
    v_artifact.write_text(json.dumps({"rows": [{"family": "qwen2.5-0.5b", "regressions": 4}]}))
    protocol = root / "ADDENDUM_AA_PROTOCOL.md"
    protocol.write_text("synthetic\n")
    vendored = root / aa.PINNED["FENCECHECK"]
    vendored.parent.mkdir(parents=True)
    vendored.write_bytes((REPO / aa.PINNED["FENCECHECK"]).read_bytes())
    monkeypatch.setattr(x, "REPO", root)
    monkeypatch.setattr(x, "OUT_DIR", cells)
    monkeypatch.setattr(x, "V_CELLS", v_cells)
    monkeypatch.setattr(x, "V_ARTIFACT", v_artifact)
    monkeypatch.setattr(x.cft, "SPLIT_DIR", split)
    monkeypatch.setattr(x, "CELLS", {f: (f"model/{f}", "float32", comparators[f]) for f in aa.FAMILIES})
    monkeypatch.setattr(x, "_scope", lambda: scope)
    monkeypatch.setattr(x, "_environment", lambda: {"synthetic": True})
    monkeypatch.setattr(x, "PLAIN_LOOP_EOS_ID", {"qwen2.5-0.5b": 151645, "granite-2b": 0})
    monkeypatch.setattr(aa, "REPO", root)
    monkeypatch.setattr(aa, "PROTOCOL_PATH", protocol)
    monkeypatch.setattr(aa, "preconditions", lambda scope, out, banked: [])
    monkeypatch.setattr(aa, "anchor_record", lambda: {"ok": True, "stamp": STAMP, "proof_sha256": "p",
                                                      "bitcoin_block_heights": [968300]})
    monkeypatch.setattr(aa, "inputs_match_the_manifest", lambda scope: ([], {"in": "sha"}))
    monkeypatch.setattr(aa, "stop_id_check", lambda scope: {})
    monkeypatch.setattr(aa, "loaded_but_unpinned", lambda: [])
    monkeypatch.setattr(aa, "git_status_of", lambda rels: "")
    monkeypatch.setattr(aa, "git_head", lambda: "abc123")
    monkeypatch.setattr(aa, "x_gates", lambda: {f: {"n": 0 if f == "granite-2b" else 5,
                                                    "mismatches": 0, "passed": True}
                                                for f in aa.FAMILIES})
    return root


def _banked(root):
    (path,) = list((root / "experiments").glob("cord_decoder_isolation_read_*.json"))
    return path, json.loads(path.read_text())


def test_the_reading_end_to_end_voids_names_both_counts_and_withholds_x3(four_families, capsys):
    assert aa.read() == 0
    path, record = _banked(four_families)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # the name carries the date the reading ran, and the record says so
    assert path.name == f"cord_decoder_isolation_read_{today}.json"
    assert record["artifact"] == f"experiments/{path.name}" and record["artifact_date"].startswith(today)
    assert record["read_started_utc"].startswith(today)
    assert record["anchor"]["proof_sha256"] == "p" and record["anchor"]["bitcoin_block_heights"] == [968300]
    assert record["git_status_porcelain_of_pinned_paths"] == ""

    # Granite is VOID in every field; its raw HURTS appears nowhere
    assert list(record["void_families"]) == ["granite-2b"]
    (granite,) = [r for r in record["rows"] if r["family"] == "granite-2b"]
    assert granite["reading"] == "VOID" and "x1_reading" not in granite
    assert record["x1_per_family"]["granite-2b"].startswith("VOID")
    assert record["x2_per_family"]["granite-2b"].startswith("VOID")
    assert "granite-2b" not in record["x2_eos_erratum"]
    assert record["x_predictions_checked"]["x1_inert_holds"]["granite-2b"].startswith("VOID")
    assert "granite-2b" not in record["console_of_read_families"]
    assert "CONSTRAINT HURTS" not in json.dumps(record)
    assert '"granite-2b": "VOID' in record["x1_finding"]
    assert record["published_first"][0].startswith("PROMPT IDENTITY FAILED")

    # SmolLM2's accounting is wrong, published second, with the reused-arm caveat
    assert "smollm2-1.7b" in record["published_first"][1] and "accounting is wrong" in record["published_first"][1]
    assert "cannot separate a wrong accounting" in record["published_first"][1]

    # Qwen: both EOS counts in the reading string, the sentence, the record
    assert record["x2_eos_erratum"]["qwen2.5-0.5b"] == {
        "as_corrected": 0, "as_first_coded": 1, "documents_changed": ["d2"],
        "stripped_id_per_document": {"d2": 151645}, "plain_loop_stop_id": 151645}
    assert "1 as first coded" in record["x2_per_family"]["qwen2.5-0.5b"]
    (qwen_x2,) = [s for s in record["licensed_sentences"] if s.startswith("qwen2.5-0.5b (X2)")]
    assert "b4facac reads PATH DIVERGES (1)" in qwen_x2 and "d2" in qwen_x2

    # Qwen's V count does not reproduce (4 published, 1 recomputed): X3 withheld
    assert record["x3"]["v_regression_count_reproduces"] is False
    for key in aa.X3_WITHHELD_KEYS:
        assert record["x3"][key] is None, key
    assert "does not reproduce" in record["published_first"][2]

    # Falcon3 is credited as REMOVES on a reused arm: the combined line carries the caveat
    assert "In falcon3-1b, arm C is V's banked cell" in record["licensed_sentences"][0]
    assert record["determinism_gate_from_x"]["granite-2b"]["status"] == "NOT APPLICABLE"
    assert "per_document_seconds" in record["disclosure"]
    assert record["how_to_recompute"].startswith("git checkout abc123 && ")

    # and the reading is made once: a second run the same day writes nothing
    before = path.read_bytes()
    assert aa.read() == 2
    assert path.read_bytes() == before


def test_a_recomputation_writes_where_it_is_told_and_says_it_is_not_the_banked_reading(four_families):
    out = four_families / "elsewhere.json"
    assert aa.read(out) == 0
    record = json.loads(out.read_text())
    assert record["banked_reading"] is False and record["artifact_date"] is None
    assert not list((four_families / "experiments").glob("*.json"))


def test_the_protocol_names_its_choices_and_owes_the_disclosure():
    text = aa.PROTOCOL_PATH.read_text()
    flat = " ".join(text.split())
    for needle in ("Nothing here is read blind", "HURTS veto", "NOT APPLICABLE",
                   "moves X2 toward PATH REPRODUCES", "not verifiable",
                   "cord_decoder_isolation_read_<date>.json", "did **not** explain",
                   "except two things", "One later access, dated 2026-10-21", "per_document_seconds",
                   "authored 05:52Z", "since 2026-09-23 (`534e7fd`)", "documents on 2026-09-17",
                   "X's artifact banks Granite's gate as n = 0", "fencecheck_x.py",
                   "ladder_e6_cord_split/gold.jsonl"):
        assert needle in flat, needle
    for gone in ("cannot help us", "strictly harder", "2.53B", "still says",
                 "cord_decoder_isolation_read_2026-09-23.json", "except one rule",
                 "no family is an exception", "covered by the same anchor",
                 "documents today", "since 2026-09-17"):
        assert gone not in flat, gone
    # the clause about Addendum Z is left exactly as it was
    assert "It needs the box that is running Addendum Z for the next 40–60 hours" in flat
