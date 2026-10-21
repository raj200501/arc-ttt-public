#!/usr/bin/env python3
"""Addendum AA — Addendum X's decoder isolation, read on the four families
whose determinism gates did not fail.

Addendum X (2026-09-16) withheld itself: its determinism gate failed on
Phi-3 in both of its runs, and X's frozen protocol made a failed gate
terminal for the whole addendum. Its cells for the other four families
were banked unread: *"raw data for a successor addendum, not results of
this one"* (X's artifact). This is the successor's reader.

It uses X's frozen readings through X's own code (`read_families`, moved
out of X's `read()`), sets aside the gate's terminality for four families,
replaces X's unscoped X1 HOLDS sentence with a scoped one, and never reads
Phi-3. What those choices buy and cost is stated in the protocol; the
reader enforces the parts that code can enforce: Phi-3 counts against the
combined X1 claim, X's unscoped HOLDS sentence is never emitted, a family
whose prompt identity fails is VOID in every field, every file the reading
runs is pinned by the anchor, and a banked reading is never overwritten.

    PYTHONPATH=src python3 scripts/decoder_isolation_read.py            # the reading (once)
    PYTHONPATH=src python3 scripts/decoder_isolation_read.py --check    # preconditions only
    PYTHONPATH=src python3 scripts/decoder_isolation_read.py --out /tmp/aa.json   # a recomputation

`--check` opens no cell: before the anchor holds it stops at the anchor,
and after it it only checks that the cells exist. The banked reading is
written once, to `experiments/cord_decoder_isolation_read_<date>.json`
where `<date>` is the UTC date the reading starts. `--out` recomputes into
a new path outside `experiments/`, only after the banked reading exists,
and under the same anchor, attestation and clean-tree checks.

Protocol: docs/research/ADDENDUM_AA_PROTOCOL.md. The stamp is taken from
its freeze line; the anchored bytes are `snapshots_ADDENDUM_AA_PROTOCOL_
<stamp>.md`, and that snapshot quotes the sha256 of every file in
`PINNED`. One of them is the input manifest, which holds the sha256 of
every input the reading opens (cells, comparators, V's artifact, the scope
record, X's artifact, the gold split); it is written once, before the
freeze, by

    PYTHONPATH=src python3 scripts/decoder_isolation_read.py --write-manifest

which hashes those files' bytes and parses none of them.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import re
import subprocess
import sys
from datetime import datetime, timezone

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import cord_decoder_isolation as x  # noqa: E402

FAMILIES = ("qwen2.5-0.5b", "smollm2-1.7b", "granite-2b", "falcon3-1b")
EXCLUDED = {
    "phi3-mini": ("NOT READ: X's determinism gate failed on this family in both of its "
                  "runs. Addendum Y explained the first run's constrained-arm flake on "
                  "cord-004 (reduction order); the generate-comparator mismatches on "
                  "cord-001 and cord-002, present in both runs, remain unexplained. "
                  "Counted as a family where CONSTRAINT REMOVES did not fire; its HURTS "
                  "veto is unverified."),
}
PROTOCOL_PATH = REPO / "docs" / "research" / "ADDENDUM_AA_PROTOCOL.md"
BANKED_NAME_RE = re.compile(r"cord_decoder_isolation_read_(\d{4}-\d{2}-\d{2})\.json")

# Every code file the reading imports or runs, and the manifest that pins
# its input data, by the name of its placeholder in the protocol's freeze
# line (`<NAME>_SHA_PLACEHOLDER`). The anchored snapshot must quote a sha256
# for exactly these paths, and the reading refuses on any mismatch. POSIX
# paths, relative to the repository.
PINNED = {
    "READER": "scripts/decoder_isolation_read.py",
    "XMODULE": "scripts/cord_decoder_isolation.py",
    "FENCECHECK": "scripts/aa_pinned/fencecheck_x.py",
    "CFT": "scripts/cord_fence_tax.py",
    "VMODULE": "scripts/cord_constrained_families.py",
    "ARCTTT_INIT": "src/arcttt/__init__.py",
    "SCORING": "src/arcttt/scoring.py",
    "TEXTTASK": "src/arcttt/text_task.py",
    "TASKS": "src/arcttt/tasks.py",
    "INPUTS": "scripts/aa_pinned/aa_input_manifest.json",
}
# X's banking commit: X's artifact and every cell X wrote first appear here,
# and the V and S/T cells X compares against were already committed by then.
# The manifest is written only if every input is byte-identical to its blob
# at this commit, so a manifest match at read time means X's own bytes.
X_RUN_COMMIT = "b4facacea71c144654b49eec7e393e79c1e0162f"
# tools/fencecheck.py as it stood at X_RUN_COMMIT (blob e138847), vendored
# byte for byte, so a later change to the shipped tool cannot move a reading.
FENCECHECK_BLOB_AT_X_RUN = "e138847d0c39d128b8e17b1d9a413ef34daebbee"

OTS_MAGIC = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
ATTESTATIONS = {"bitcoin": bytes.fromhex("0588960d73d71901"),
                "pending": bytes.fromhex("83dfe30d2ef90c8e")}
STAMP_RE = re.compile(r"ADDENDUM_AA_PROTOCOL\.md\.(\d{4}-\d{2}-\d{2}T\d{4}Z)\.ots")

# X's frozen predictions (ADDENDUM_X_PROTOCOL.md, "Two predictions, written
# before the run"), in its words, split into what four families can check.
X_PREDICTIONS_VERBATIM = {
    "X1": ("V banked 0 constrained steps on every document of four families and on "
           "43 of Falcon3's 50, with 0 fallbacks everywhere. If that accounting is "
           "correct, then arms C and P must be byte-identical on every document with "
           "0 constrained steps [...] so X1 reads INERT on Qwen2.5-0.5B, SmolLM2, "
           "Granite and Phi-3, and on Falcon3 the arms differ on exactly the seven "
           "documents where the validator fired."),
    "X2": ("PATH DIVERGES on Phi-3 (V saw 32 of 50 bodies differ at bfloat16) and "
           "PATH REPRODUCES on SmolLM2. Qwen, Falcon3 and Granite are not predicted "
           "— Granite because its arms are being compared under a prompt V never used."),
}
X1_INERT_PREDICTED = ("qwen2.5-0.5b", "smollm2-1.7b", "granite-2b")
X2_REPRODUCES_PREDICTED = ("smollm2-1.7b",)
ACCOUNTING_WORDS = ("the decoder's own accounting is wrong, and that is the result — a "
                    "larger and worse one than the contrast this addendum was built to make")
REUSED_C_CAVEAT = ("the difference is the constraint, or V's cell not reproducing across "
                   "processes on that document, and this reading cannot separate the two")
SECONDS_DISCLOSURE = (
    "On 2026-10-21 an agent designing a different addendum (AC) read the "
    "per_document_seconds field of the Addendum S/T comparator cells to estimate decode "
    "cost. It read no prediction text, no token ids and no other field. No reading here "
    "takes decode time as an input: X1 compares arms C and P by their texts and banked "
    "constrained steps, X2 compares arm P with G by token ids or with G0 by decoded text, "
    "and X3 classifies the texts of arms C, G and G0 as valid or invalid JSON. Decode time "
    "tracks roughly how long an output was, and none of the three readings is a function of "
    "output length alone. It is the only access to those cells since X's run that the "
    "operator knows of.")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rel(path: pathlib.Path) -> str:
    return pathlib.Path(path).relative_to(REPO).as_posix()


# -- the proof -------------------------------------------------------------------
# A minimal reader of the OpenTimestamps proof format, enough to name the file
# digest the proof commits to and every attestation it carries, with the block
# height of each Bitcoin one. It walks the operation tree; it does not
# recompute the merkle path (`ots verify` does that against a node).

_UNARY_OPS = {0x02, 0x03, 0x08, 0x67, 0xf2, 0xf3}   # sha1 ripemd160 sha256 keccak256 reverse hexlify
_BINARY_OPS = {0xf0, 0xf1}                          # append prepend


class _Cursor:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise ValueError("truncated proof")
        chunk = self.data[self.pos:self.pos + n]
        self.pos += n
        return chunk

    def varuint(self) -> int:
        value = shift = 0
        while True:
            byte = self.take(1)[0]
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return value
            shift += 7
            if shift > 63:
                raise ValueError("varuint too long")

    def varbytes(self, limit: int) -> bytes:
        n = self.varuint()
        if n > limit:
            raise ValueError("varbytes too long")
        return self.take(n)


def _ots_tag_or_attestation(cur: _Cursor, tag: int, found: list, depth: int) -> None:
    if tag == 0x00:
        kind = cur.take(8)
        payload = _Cursor(cur.varbytes(8192))
        if kind == ATTESTATIONS["bitcoin"]:
            found.append({"kind": "bitcoin", "height": payload.varuint()})
        elif kind == ATTESTATIONS["pending"]:
            found.append({"kind": "pending",
                          "uri": payload.varbytes(1000).decode("ascii", "replace")})
        else:
            found.append({"kind": "unknown", "tag": kind.hex()})
        return
    if tag in _BINARY_OPS:
        cur.varbytes(4096)
    elif tag not in _UNARY_OPS:
        raise ValueError(f"unknown operation {tag:#04x}")
    _ots_timestamp(cur, found, depth + 1)


def _ots_timestamp(cur: _Cursor, found: list, depth: int) -> None:
    if depth > 256:
        raise ValueError("proof nested too deep")
    tag = cur.take(1)[0]
    while tag == 0xFF:
        _ots_tag_or_attestation(cur, cur.take(1)[0], found, depth)
        tag = cur.take(1)[0]
    _ots_tag_or_attestation(cur, tag, found, depth)


def parse_ots(blob: bytes) -> dict | None:
    """{file_sha256, attestations} of a version-1 sha256 file proof, or None
    for anything that does not parse as one, to its last byte."""
    try:
        cur = _Cursor(blob)
        if cur.take(len(OTS_MAGIC)) != OTS_MAGIC or cur.varuint() != 1:
            return None
        if cur.take(1)[0] != 0x08:
            return None
        digest = cur.take(32).hex()
        found: list = []
        _ots_timestamp(cur, found, 0)
        if cur.pos != len(blob):
            return None
        return {"file_sha256": digest, "attestations": found}
    except (ValueError, IndexError):
        return None


def anchored_digest(ots_path: pathlib.Path) -> str | None:
    """The file sha256 an OpenTimestamps proof commits to, or None unless
    it is a version-1 sha256 file proof carrying at least one Bitcoin or
    pending attestation."""
    parsed = parse_ots(pathlib.Path(ots_path).read_bytes())
    if parsed is None or not any(a["kind"] in ATTESTATIONS for a in parsed["attestations"]):
        return None
    return parsed["file_sha256"]


def attestation_kinds(ots_path: pathlib.Path) -> list[str]:
    ots_path = pathlib.Path(ots_path)
    parsed = parse_ots(ots_path.read_bytes()) if ots_path.exists() else None
    if parsed is None:
        return []
    return sorted({a["kind"] for a in parsed["attestations"] if a["kind"] in ATTESTATIONS})


def bitcoin_heights(ots_path: pathlib.Path) -> list[int]:
    parsed = parse_ots(pathlib.Path(ots_path).read_bytes())
    if parsed is None:
        return []
    return sorted({a["height"] for a in parsed["attestations"] if a["kind"] == "bitcoin"})


# -- the anchor ---------------------------------------------------------------


def stamp_of(protocol_text: str) -> str | None:
    """The stamp the freeze line quotes; the single source of it."""
    match = STAMP_RE.search(protocol_text)
    return match.group(1) if match else None


def proof_and_snapshot(stamp: str) -> tuple[pathlib.Path, pathlib.Path]:
    return (PROTOCOL_PATH.with_name(f"{PROTOCOL_PATH.name}.{stamp}.ots"),
            PROTOCOL_PATH.with_name(f"snapshots_ADDENDUM_AA_PROTOCOL_{stamp}.md"))


def pinned_digests(anchored_text: str) -> dict:
    """The digests the anchored protocol quotes, against every pinned file as
    it is now. They match only if the protocol quotes exactly the pinned
    set and every digest agrees."""
    quoted = dict(re.findall(r"`([A-Za-z0-9_./-]+)` sha256 `([0-9a-f]{64})`", anchored_text))
    now = {}
    for rel in PINNED.values():
        path = REPO / rel
        now[rel] = _sha(path.read_bytes()) if path.is_file() else None
    expected = set(PINNED.values())
    mismatched = sorted(r for r in expected if r in quoted and quoted[r] != now[r])
    return {"quoted_in_anchored_protocol": quoted, "now": now,
            "not_quoted": sorted(expected - set(quoted)),
            "quoted_but_not_pinned_by_the_reader": sorted(set(quoted) - expected),
            "mismatched": mismatched,
            "matches": set(quoted) == expected and not mismatched}


def anchor_record() -> dict:
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    stamp = stamp_of(text)
    record = {"stamp": stamp, "protocol_sha256_at_read": _sha(PROTOCOL_PATH.read_bytes()),
              "ok": False}
    if stamp is None:
        record["why"] = "the freeze line quotes no stamp"
        return record
    proof, snapshot = proof_and_snapshot(stamp)
    record.update({"proof": _rel(proof), "snapshot": _rel(snapshot)})
    if not proof.exists() or not snapshot.exists():
        record["why"] = "proof or anchored snapshot missing"
        return record
    blob = proof.read_bytes()
    parsed = parse_ots(blob)
    snap = snapshot.read_bytes()
    record.update({
        "proof_sha256": _sha(blob),
        "attestations": attestation_kinds(proof),
        "bitcoin_block_heights": bitcoin_heights(proof),
        "anchored_sha256": None if parsed is None else parsed["file_sha256"],
        "snapshot_sha256": _sha(snap),
        "protocol_begins_with_the_snapshot": PROTOCOL_PATH.read_bytes().startswith(snap),
    })
    record["snapshot_is_what_the_proof_commits_to"] = record["anchored_sha256"] == _sha(snap)
    pinned = pinned_digests(snap.decode("utf-8"))
    record["pinned"] = pinned
    # A calendar promise is not a timestamp. The reading waits for a Bitcoin
    # attestation, so it cannot precede the block that dates the protocol;
    # the block height read from the proof is banked with the reading.
    record["bitcoin_attested"] = bool(record["bitcoin_block_heights"])
    record["ok"] = bool(record["snapshot_is_what_the_proof_commits_to"]
                        and record["protocol_begins_with_the_snapshot"]
                        and pinned["matches"] and record["bitcoin_attested"])
    if not record["ok"]:
        record["why"] = ("the proof, the snapshot, the protocol or a pinned file's digest "
                         "disagree, or the proof is not yet Bitcoin-attested (run ots upgrade)")
    return record


# -- git ------------------------------------------------------------------------


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=check)


def uncommitted(rels) -> list[str]:
    """The paths among `rels` that are not tracked, or differ from HEAD."""
    out = []
    for rel in rels:
        try:
            tracked = _git("ls-files", "--error-unmatch", "--", rel, check=False).returncode == 0
            dirty = bool(_git("status", "--porcelain", "--", rel).stdout.strip())
        except (OSError, subprocess.CalledProcessError) as error:
            return [f"git cannot report on the tree: {error!r}"]
        if not tracked or dirty:
            out.append(rel)
    return out


def clean_tree_paths() -> list[str]:
    """What must be committed and unchanged for a reading to run: the
    protocol, its snapshot once it exists, and every pinned file, so that
    the git HEAD the reading records holds every byte of code it ran."""
    paths = [_rel(PROTOCOL_PATH)]
    stamp = stamp_of(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if stamp:
        snapshot = proof_and_snapshot(stamp)[1]
        if snapshot.exists():
            paths.append(_rel(snapshot))
    return paths + list(PINNED.values())


def git_status_of(rels) -> str:
    try:
        return _git("status", "--porcelain", "--", *rels).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        return f"git unavailable: {error!r}"


def git_head() -> str:
    try:
        return _git("rev-parse", "HEAD").stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


# -- the banked reading and --out -----------------------------------------------


def banked_path(date: str) -> pathlib.Path:
    return REPO / "experiments" / f"cord_decoder_isolation_read_{date}.json"


def banked_readings() -> list[pathlib.Path]:
    folder = REPO / "experiments"
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("cord_decoder_isolation_read_*.json")
                  if BANKED_NAME_RE.fullmatch(p.name))


def out_path_problems(out) -> list[str]:
    """A recomputation writes a new file outside experiments/, whatever the
    spelling: the path is resolved (symlinks, `..`, relative) first."""
    resolved = pathlib.Path(os.path.realpath(out))
    experiments = pathlib.Path(os.path.realpath(REPO / "experiments"))
    problems = []
    if resolved == experiments or experiments in resolved.parents:
        problems.append(f"--out {out} resolves to {resolved}, inside experiments/, where the "
                        "banked reading and every other banked artifact live; recompute "
                        "into a path outside it")
    if resolved.exists():
        problems.append(f"--out {out} exists; a recomputation writes a new file and never "
                        "replaces one")
    return problems


@contextlib.contextmanager
def _one_reading_at_a_time():
    """An exclusive, non-blocking lock on the protocol file, released when the
    process ends however it ends, so two readings cannot run at once and a
    killed one leaves no lock behind."""
    try:
        import fcntl
    except ImportError:  # not POSIX: the exclusive write below still holds
        yield True
        return
    with open(PROTOCOL_PATH, "rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# -- preconditions --------------------------------------------------------------


def x_gates() -> dict:
    """X's determinism gate as X's banked (and pinned) artifact records it."""
    return json.loads(x.OUT.read_text(encoding="utf-8"))["determinism_gate"]


def gate_status(family: str, scope: dict, gates: dict | None = None) -> dict:
    gates = x_gates() if gates is None else gates
    gate = gates.get(family)
    if gate is None:
        return {"status": "MISSING FROM X'S ARTIFACT"}
    reused = scope[family]["arm_C_reused_from_V"]
    if not reused and gate["n"] == 0 and gate["passed"]:
        status = "NOT APPLICABLE"
    elif reused and gate["n"] == x.DETERMINISM_DOCS and gate["mismatches"] == 0 and gate["passed"]:
        status = "PASSED"
    elif not gate["passed"]:
        status = "FAILED"
    else:
        status = "INCONSISTENT"
    out = {"status": status, "n": gate["n"], "mismatches": gate["mismatches"],
           "x_artifact_banks": {"n": gate["n"], "mismatches": gate["mismatches"],
                                "passed": gate["passed"]}}
    if status == "NOT APPLICABLE":
        out["not_applicable_because"] = {
            "arm_C_reused_from_V": False,
            "chat_template_clock_calls": scope[family]["chat_template_clock_calls"]}
    return out


def cell_paths(scope: dict) -> list[pathlib.Path]:
    paths = []
    for family in FAMILIES:
        paths += [x.OUT_DIR / f"{family}_plain.json", x._arm_c_path(family, scope),
                  x.CELLS[family][2]]
        if scope[family]["arm_G_runs"]:
            paths.append(x.OUT_DIR / f"{family}_generate_neutral.json")
    return paths


def preconditions(scope: dict, out: pathlib.Path, banked_reading: bool) -> list[str]:
    """Every reason the reading must not run; empty means it may. The same
    anchor, attestation and clean-tree checks hold for the banked reading
    and for a recomputation. Nothing about a cell is looked at until the
    anchor holds, and even then only whether it exists."""
    problems = []
    anchor = anchor_record()
    if not anchor["ok"]:
        problems.append(f"anchor: {anchor.get('why')} (stamp {anchor.get('stamp')})")
    existing = banked_readings()
    if banked_reading:
        if existing:
            problems.append(f"{existing[0].name} is already banked; a reading is made once. "
                            "Recompute into another path with --out to compare.")
    else:
        if not existing:
            problems.append("the banked reading does not exist yet; a recomputation with --out "
                            "is only possible after it")
        problems += out_path_problems(out)
    dirty = uncommitted(clean_tree_paths())
    if dirty:
        problems.append("not committed, or changed since the last commit: " + ", ".join(dirty))
    if not anchor["ok"]:
        return problems
    gates = x_gates()
    for family in FAMILIES:
        status = gate_status(family, scope, gates)["status"]
        if status not in ("PASSED", "NOT APPLICABLE"):
            problems.append(f"{family}: X's determinism gate is {status}")
    problems += [f"missing input: {_rel(p)}" for p in input_paths(scope) if not p.exists()]
    return problems


def input_paths(scope: dict) -> list[pathlib.Path]:
    """Every input file the reading opens, besides code: the scope record,
    X's artifact (the gate status), V's artifact (V's published counts), the
    gold split, and the cells."""
    return [x.CONFIGS, x.OUT, x.V_ARTIFACT, x.cft.SPLIT_DIR / "gold.jsonl",
            x.cft.SPLIT_DIR / "train.jsonl", *cell_paths(scope)]


def write_manifest() -> int:
    """Hash the bytes of every input once, before the freeze. Parses none of
    them. Refuses unless each is committed, clean and byte-identical to its
    blob at X's banking commit, and never writes over a manifest."""
    path = REPO / PINNED["INPUTS"]
    if path.exists():
        print(f"REFUSED: {PINNED['INPUTS']} exists; the manifest is written once.")
        return 2
    rels = [_rel(p) for p in input_paths(x._scope())]
    problems, files = [], {}
    for rel in rels:
        if not (REPO / rel).is_file():
            problems.append(f"{rel}: missing")
            continue
        at_x = _git("rev-parse", "--verify", "--quiet", f"{X_RUN_COMMIT}:{rel}",
                    check=False).stdout.strip()
        now = _git("hash-object", "--", rel).stdout.strip()
        if at_x != now:
            problems.append(f"{rel}: not byte-identical to its blob at {X_RUN_COMMIT[:7]}")
        files[rel] = {"sha256": _sha((REPO / rel).read_bytes()), "git_blob": now}
    problems += [f"{rel}: not committed, or changed since the last commit"
                 for rel in uncommitted(rels)]
    if problems:
        print("REFUSED:\n  " + "\n  ".join(problems))
        return 2
    record = {
        "what": ("Addendum AA's input data, pinned by sha256 before the freeze. The bytes of "
                 "each file were hashed and none was parsed. Every file is byte-identical to "
                 "its blob at X's banking commit."),
        "x_run_commit": X_RUN_COMMIT,
        "files": files,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    x.write_new_file(path, json.dumps(record, indent=1) + "\n")
    print(f"written: {PINNED['INPUTS']} ({len(files)} files); commit it before the freeze")
    return 0


def inputs_match_the_manifest(scope: dict) -> tuple[list[str], dict]:
    """At read time: the inputs the reading opens are exactly the manifest's,
    and every one hashes to it."""
    manifest = json.loads((REPO / PINNED["INPUTS"]).read_text(encoding="utf-8"))["files"]
    rels = [_rel(p) for p in input_paths(scope)]
    problems = []
    if set(rels) != set(manifest):
        problems.append("the reading's inputs and the manifest's differ: "
                        f"not in the manifest {sorted(set(rels) - set(manifest))}, "
                        f"not read {sorted(set(manifest) - set(rels))}")
    read = {}
    for rel in rels:
        digest = _sha((REPO / rel).read_bytes())
        read[rel] = digest
        if manifest.get(rel, {}).get("sha256") != digest:
            problems.append(f"{rel}: differs from the manifest")
    return problems, read


def stop_id_check(scope: dict) -> dict:
    """The plain loop's stop id is pinned in X's module; where transformers
    is present it is re-derived from the tokenizer exactly as X's `_load`
    built it, and a disagreement stops the reading."""
    out = {}
    for family in FAMILIES:
        if not scope[family]["arm_G_runs"]:
            continue
        entry = {"pinned": x.PLAIN_LOOP_EOS_ID.get(family)}
        try:
            from transformers import AutoTokenizer
            entry["tokenizer_now"] = AutoTokenizer.from_pretrained(x.CELLS[family][0]).eos_token_id
            entry["agrees"] = entry["tokenizer_now"] == entry["pinned"]
        except Exception as error:  # noqa: BLE001
            entry["not_cross_checked"] = repr(error)
        out[family] = entry
    return out


def pinned_fencecheck():
    """The fencecheck X ran with, from its vendored, pinned copy."""
    spec = importlib.util.spec_from_file_location("aa_pinned_fencecheck_x",
                                                  REPO / PINNED["FENCECHECK"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _live_fencecheck_refused():
    original = x.cft._fc

    def refuse():
        raise SystemExit("Addendum AA reads with the fencecheck X ran with "
                         f"({PINNED['FENCECHECK']}), never the live tools/fencecheck.py")

    x.cft._fc = refuse
    try:
        yield
    finally:
        x.cft._fc = original


def loaded_but_unpinned() -> list[str]:
    """Repository files loaded as modules that the anchor does not pin."""
    root = REPO.resolve()
    found = set()
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if not file:
            continue
        try:
            found.add(pathlib.Path(file).resolve().relative_to(root).as_posix())
        except ValueError:
            continue
    return sorted(found - set(PINNED.values()))


# -- readings layered on X's -----------------------------------------------------


def prompt_identity(family: str, scope: dict, mismatches: list[str]) -> dict:
    """Per comparison: verified, not verifiable (no digest banked), or MISMATCHED."""
    const = json.loads(x._arm_c_path(family, scope).read_text())
    comparisons = {
        "C vs P": ("verified" if "per_document_prompt_sha256_16" in const
                   else "not verifiable (the reused arm C cell banks no prompt digest)"),
    }
    if scope[family]["arm_G_runs"]:
        comparisons["P vs G"] = "verified"
    else:
        comparisons["P vs G0"] = "not verifiable (the G0 comparator cell banks text only)"
    if scope[family]["modifiers"]:
        comparisons["G vs G0 (X3)"] = "not verifiable (the G0 comparator cell banks text only)"
    for label in list(comparisons):
        short = label.split(" (")[0]
        if any(m.startswith(f"{family}/") and f"({short})" in m for m in mismatches):
            comparisons[label] = "MISMATCHED"
    return comparisons


def void_reasons(identity: dict, mismatches: list[str]) -> dict:
    """family -> why it is VOID, for every family with a MISMATCHED pair."""
    out = {}
    for family, comparisons in identity.items():
        bad = [label for label, state in comparisons.items() if state == "MISMATCHED"]
        if bad:
            out[family] = {
                "reason": "prompt identity MISMATCHED on " + ", ".join(bad),
                "mismatched_documents": sorted(m for m in mismatches if m.startswith(f"{family}/"))}
    return out


def void_reading(void: dict, family: str) -> str:
    return f"VOID ({void[family]['reason']})"


def scoped_x1_finding(x1_per_family: dict) -> str:
    """X's combine rule with Phi-3 counted against it, and X's unscoped
    HOLDS sentence made unreachable: leaving Phi-3 unread also takes it out
    of the HURTS veto, so X's unscoped sentence cannot be said. A VOID
    family (its reading string starts with "VOID") is treated like Phi-3:
    counted as a family where REMOVES did not fire, its HURTS veto
    unverified, and its raw reading never consulted."""
    void = sorted(f for f, r in x1_per_family.items() if r.startswith("VOID"))
    read = len(x1_per_family) - len(void)
    scope_words = (f"{read} read" + (f", {len(void)} VOID ({', '.join(void)})" if void else "")
                   + "; phi3-mini not read")
    combined = dict(x1_per_family)
    combined.update(EXCLUDED)
    finding = x.x1_combine(combined)
    finding = finding.replace("families tested", f"families ({scope_words})")
    if finding.startswith("X1 HOLDS"):
        n = sum(1 for r in x1_per_family.values() if r.startswith("CONSTRAINT REMOVES"))
        return (f"X1 HOLDS ON THE FAMILIES READ ONLY: CONSTRAINT REMOVES in {n} of 5 "
                f"families ({scope_words}, so the condition that no family fires HURTS is "
                f"unverified on {1 + len(void)} of 5). X's unscoped HOLDS sentence is NOT "
                f"licensed. Per-family: {json.dumps(combined)}")
    return finding


def prediction_checks(rows: list[dict], void: dict | None = None) -> dict:
    void = void or {}
    by_family = {r["family"]: r for r in rows if r["family"] not in void}

    def per(family, value):
        return void_reading(void, family) if family in void else value(by_family[family])

    return {
        "verbatim": X_PREDICTIONS_VERBATIM,
        "x1_accounting_per_document": {
            f: per(f, lambda r: {
                "holds": r["accounting_consistent"],
                "documents_with_zero_steps_but_differing_arms": r["documents_with_zero_steps_but_differing_arms"],
                "documents_with_steps_but_identical_arms": r["documents_with_steps_but_identical_arms"]})
            for f in [r["family"] for r in rows]},
        "x1_inert_where_predicted": {f: per(f, lambda r: r["x1_reading"]) for f in X1_INERT_PREDICTED},
        "x1_inert_holds": {f: per(f, lambda r: r["x1_reading"].startswith("CONSTRAINT INERT"))
                           for f in X1_INERT_PREDICTED},
        "x1_falcon3_differs_on_exactly_the_documents_with_steps": per(
            "falcon3-1b", lambda r: {"differing_documents": len(r["x1_differing_documents"]),
                                     "holds": r["accounting_consistent"]}),
        "x2_where_predicted": {f: per(f, lambda r: {"reading": r["x2_reading"],
                                                    "basis": r["x2_basis"],
                                                    "holds": r["x2_reading"].startswith("PATH REPRODUCES")})
                               for f in X2_REPRODUCES_PREDICTED},
        "not_checkable": [
            "X1 INERT on Phi-3 (not read): 3 of the 4 families X predicted INERT are checkable",
            "X1 per-document accounting on Phi-3 (not read)",
            "X2 PATH DIVERGES on Phi-3 (not read)",
        ],
    }


def _erratum_counts(r: dict) -> str:
    changed = r["x2_documents_changed_by_the_eos_erratum"]
    first = r["x2_divergent_documents_as_first_coded"]
    as_coded = "PATH REPRODUCES" if first == 0 else f"PATH DIVERGES ({first})"
    return (f" Both counts, because the EOS erratum moves X2 the way that favours us: "
            f"{r['x2_divergent_documents']} of {r['n']} documents diverge under the "
            f"2026-09-23 erratum; X's code as banked at b4facac reads {as_coded}"
            + (f", the difference being {', '.join(changed)}" if changed else "") + ".")


def licensed_sentences(rows: list[dict], x1_finding: str, x3: dict | None,
                       void: dict, scope: dict) -> list[str]:
    readable = [r for r in rows if r["family"] not in void]
    hurts = [f"{r['family']}: {r['x1_reading']}" for r in readable
             if r["x1_reading"].startswith("CONSTRAINT HURTS")]
    credited = [r["family"] for r in readable
                if r["x1_reading"].startswith(("CONSTRAINT REMOVES", "CONSTRAINT HELPS"))
                and scope[r["family"]]["arm_C_reused_from_V"] and r["x1_differing_documents"]]
    combined = ("X1 (combined; denominator 5, phi3-mini not read and counted against): "
                + x1_finding.split(" Per-family")[0])
    if credited:
        combined += (f" In {', '.join(credited)}, arm C is V's banked cell: where the arms "
                     f"differ, {REUSED_C_CAVEAT}.")
    if hurts:
        combined += f" HURTS, named at full size: {'; '.join(hurts)}."
    out = [combined]
    for r in rows:
        family = r["family"]
        if family in void:
            out.append(f"{family}: VOID ({void[family]['reason']}); its readings are recorded as "
                       "VOID and no sentence about this family's decoder isolation is licensed.")
            continue
        caveat = ""
        if scope[family]["arm_C_reused_from_V"] and r["x1_differing_documents"]:
            caveat = f" Where arms C and P differ, arm C is V's banked cell: {REUSED_C_CAVEAT}."
        out.append(f"{family} (X1): {r['x1_reading']}.{caveat}")
        token = r["x2_basis"] == "token ids"
        if r["x2_reading"].startswith("PATH REPRODUCES") and token:
            out.append(f"{family} (X2): our loop reproduces generate token-for-token on these "
                       f"{r['n']} documents, when neither is constrained and neither has "
                       "generation defaults in play." + _erratum_counts(r))
        elif r["x2_reading"].startswith("PATH REPRODUCES"):
            out.append(f"{family} (X2): our loop reproduces generate's decoded text on these "
                       f"{r['n']} documents (WEAKER TEST: decoded text, not token ids).")
        else:
            note = (_erratum_counts(r) if token else
                    " The comparator cell carries no environment record, so path divergence "
                    "and drift in the banked cell are not separated.")
            out.append(f"{family} (X2): {r['x2_reading']} on the {r['x2_basis']} basis, on "
                       f"these {r['n']} documents; invalid outputs {r['x2_invalid_plain']} (our "
                       f"loop) against {r['x2_invalid_other_arm']} ({r['x2_compared_against']}).{note}")
    if x3 is not None and x3["family"] not in void:
        out.append(f"{x3['family']} (X3): {x3['reading']}")
    return out


def obligations(rows: list[dict], x3: dict | None, void: dict | None = None) -> list[dict]:
    """What the protocol says must change the same day, derived from the readings."""
    void = void or {}
    owed = [{"file": "tools/README.md",
             "what": "the jsongreedy.py section carries this addendum's reading beside X's withholding"}]
    hurts = [r["family"] for r in rows
             if r["family"] not in void and r["x1_reading"].startswith("CONSTRAINT HURTS")]
    if hurts:
        owed.append({"file": "tools/README.md",
                     "what": ("the paragraph beginning 'What it does and does not do is measured' "
                              f"names {', '.join(hurts)} as CONSTRAINT HURTS at full size")})
    if (x3 is not None and x3["family"] not in void
            and str(x3.get("reading", "")).startswith("V'S ATTRIBUTION IS TOO STRONG")):
        for target in ("docs/research/ADDENDUM_V_PROTOCOL.md (dated erratum)",
                       "VERDICT.md (Addendum V row)", "CORRECTIONS.md (dated row)"):
            owed.append({"file": target, "what": "V's attribution narrowed to E of R_V, per X3"})
    return owed


X3_WITHHELD_KEYS = ("regressions_explained_by_the_defaults", "residual_documents",
                    "residual_attributed_by_arithmetic", "v_regressions_reexamined")
X3_V_COUNT_KEYS = ("v_published_regressions", "v_regressions_recomputed_here",
                   "v_regression_count_reproduces", "note_if_v_does_not_reproduce")


def x3_as_banked(x3: dict | None, void: dict) -> dict | None:
    """VOID, withheld or as read. A VOID family keeps only the check on V's
    own published count, which uses V's two cells and none of this family's
    compared pairs; a count that does not reproduce withholds every field
    the explained count could be recomputed from."""
    if x3 is None:
        return None
    reproduces = x3.get("v_regression_count_reproduces", True)
    if x3["family"] in void:
        kept = {k: x3[k] for k in X3_V_COUNT_KEYS if k in x3} if not reproduces else {}
        return {"family": x3["family"], "reading": void_reading(void, x3["family"]), **kept}
    x3 = dict(x3)
    if not reproduces:
        for key in X3_WITHHELD_KEYS:
            x3[key] = None  # withheld: the disagreement is the finding
    return x3


def _without_void_lines(console: str, void: dict, x3: dict | None) -> str:
    """X's console prints every family's raw X1 and X2. A VOID family's
    lines, and its X3 line, are not readings, so they are not banked."""
    kept, skipping_x3 = [], False
    for line in console.splitlines(keepends=True):
        head = line.split(" ", 1)[0]
        if head in void:
            skipping_x3 = x3 is not None and x3["family"] == head
            continue
        if line.startswith(" ") and line.strip().startswith("X3 ") and skipping_x3:
            continue
        if head:
            skipping_x3 = False
        kept.append(line)
    return "".join(kept)


def _withheld(problems: list[str]) -> int:
    print("WITHHELD: nothing is read and nothing is written:")
    for p in problems:
        print(f"  {p}")
    return 2


def read(out=None) -> int:
    banked_reading = out is None
    scope = x._scope()
    started_at = datetime.now(timezone.utc)
    started = started_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    target = (banked_path(started_at.strftime("%Y-%m-%d")) if banked_reading
              else pathlib.Path(os.path.realpath(out)))
    problems = preconditions(scope, target, banked_reading)
    if problems:
        return _withheld(problems)
    with _one_reading_at_a_time() as alone:
        if not alone:
            return _withheld(["another reading holds the lock on the protocol file"])
        return _read_locked(scope, target, banked_reading, started)


def _read_locked(scope: dict, target: pathlib.Path, banked_reading: bool, started: str) -> int:
    if banked_reading and banked_readings():
        return _withheld([f"{banked_readings()[0].name} was banked while this reading waited"])
    anchor = anchor_record()
    cell_problems, inputs_read = inputs_match_the_manifest(scope)
    stop_ids = stop_id_check(scope)
    cell_problems += [f"{f}: the pinned plain-loop stop id {e['pinned']} disagrees with the "
                      f"tokenizer's eos id {e['tokenizer_now']}"
                      for f, e in stop_ids.items() if e.get("agrees") is False]
    if cell_problems:
        return _withheld(cell_problems)
    try:
        environment = x._environment()
    except Exception as error:  # noqa: BLE001
        environment = {"unavailable": repr(error)}
    fencecheck = pinned_fencecheck()
    buffered = io.StringIO()
    with _live_fencecheck_refused(), contextlib.redirect_stdout(buffered):
        rows, x1_per_family, x2_per_family, x3, prompt_mismatches = x.read_families(
            FAMILIES, scope, fencecheck=fencecheck)
    if [r["family"] for r in rows] != [f for f in x.CELLS if f in FAMILIES]:
        raise SystemExit(f"read_families returned {[r['family'] for r in rows]}, not {FAMILIES}")
    unpinned = loaded_but_unpinned()
    if unpinned:
        return _withheld([f"loaded but not pinned by the anchor: {', '.join(unpinned)}"])
    status_of_pinned = git_status_of(clean_tree_paths())
    if status_of_pinned.strip():
        return _withheld(["a pinned path changed during the reading:\n" + status_of_pinned])

    identity = {f: prompt_identity(f, scope, prompt_mismatches) for f in FAMILIES}
    void = void_reasons(identity, prompt_mismatches)
    # A VOID family is recorded as VOID in every field. It stays in the
    # combined count as a family where REMOVES did not fire: dropping it
    # would shrink the denominator, the flattering way.
    x1_banked = {f: (void_reading(void, f) if f in void else r) for f, r in x1_per_family.items()}
    x2_banked = {f: (void_reading(void, f) if f in void else r) for f, r in x2_per_family.items()}
    rows_banked = [r if r["family"] not in void else
                   {"family": r["family"], "model": r["model"], "n": r["n"], "reading": "VOID",
                    **void[r["family"]]} for r in rows]
    readable = [r for r in rows if r["family"] not in void]
    console = _without_void_lines(buffered.getvalue(), void, x3)
    x1_finding = scoped_x1_finding(x1_banked)
    x3_banked = x3_as_banked(x3, void)
    inconsistent = sorted(r["family"] for r in readable if not r["accounting_consistent"])
    first = []
    if void:
        first.append("PROMPT IDENTITY FAILED on a compared pair in "
                     + "; ".join(f"{f} ({void[f]['reason']})" for f in sorted(void))
                     + ": those families are VOID, their readings are recorded as VOID, and "
                       "no sentence about their decoder isolation is licensed.")
    if inconsistent:
        reused = [f for f in inconsistent if scope[f]["arm_C_reused_from_V"]]
        first.append("In " + ", ".join(inconsistent) + ": " + ACCOUNTING_WORDS + "."
                     + (f" In {', '.join(reused)} arm C is V's banked cell, and this reading "
                        "cannot separate a wrong accounting from V's cell not reproducing "
                        "across processes on those documents; either one is a defect in a "
                        "published result." if reused else ""))
    if x3_banked is not None and x3_banked.get("v_regression_count_reproduces") is False:
        first.append("V's published regression count does not reproduce from its own "
                     "banked cells; X3's attribution is withheld and this publishes first.")
    head = git_head()
    rel_target = (_rel(target) if banked_reading else str(target))
    record = {
        "what": ("Addendum AA: Addendum X's frozen readings, through X's own code, on the "
                 "four families whose determinism gates did not fail. Set aside: the "
                 "gate's terminality. Changed: X1's combine rule cannot emit X's unscoped "
                 "HOLDS sentence."),
        "protocol": "docs/research/ADDENDUM_AA_PROTOCOL.md",
        "reads": ("docs/research/ADDENDUM_X_PROTOCOL.md (readings committed "
                  "2026-09-16T06:05Z, authored 05:52Z, in 0e5a11c; X's cells first "
                  "committed in b4facac, 2026-09-16T18:27Z)"),
        "banked_reading": banked_reading,
        "artifact": rel_target,
        "artifact_date": (started[:10] + " (the UTC date the reading started)"
                          if banked_reading else None),
        "read_started_utc": started,
        "git_head": head,
        "git_status_porcelain_of_pinned_paths": status_of_pinned,
        "anchor": anchor,
        "x_run_commit": X_RUN_COMMIT,
        "inputs_read_sha256": inputs_read,
        "plain_loop_stop_ids": stop_ids,
        "environment": environment,
        "families_read": list(FAMILIES),
        "families_not_read": EXCLUDED,
        "determinism_gate_from_x": {f: gate_status(f, scope) for f in FAMILIES},
        "published_first": first,
        "prompt_identity_per_comparison": identity,
        "void_families": void,
        "rows": rows_banked,
        "x1_per_family": x1_banked,
        "x1_finding": x1_finding,
        "x2_per_family": x2_banked,
        "x2_eos_erratum": {r["family"]: {"as_corrected": r["x2_divergent_documents"],
                                         "as_first_coded": r["x2_divergent_documents_as_first_coded"],
                                         "documents_changed": r["x2_documents_changed_by_the_eos_erratum"],
                                         "stripped_id_per_document": r["x2_eos_erratum_stripped_ids"],
                                         "plain_loop_stop_id": x.PLAIN_LOOP_EOS_ID.get(r["family"])}
                           for r in readable if r["x2_basis"] == "token ids"},
        "x3": x3_banked,
        "decoder_accounting_inconsistent_in": inconsistent,
        "x_predictions_checked": prediction_checks(rows, void),
        "licensed_sentences": licensed_sentences(rows, x1_finding, x3_banked, void, scope),
        "obligations_the_same_day": obligations(rows, x3_banked, void),
        "disclosure": (
            "Addendum X's artifact records that the operator computed the per-family X1 "
            "quantities for these families by hand during X's run, for scheduling. The "
            "record names X1 only and is silent on X2 and X3, so no reading here claims to "
            "be blind. X's readings were committed before its cells were; X's anchor came "
            "after its run, so a stranger cannot check that order. The decision to read "
            "these cells, and every choice in this addendum, were made after the hand "
            "computation. " + SECONDS_DISCLOSURE),
        "how_to_recompute": (f"git checkout {head or '<git_head>'} && PYTHONPATH=src python3 "
                             "scripts/decoder_isolation_read.py --out <a new path outside "
                             "experiments/>"),
        "console_of_read_families": console,
    }
    record["read_finished_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        x.write_new_file(target, json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    except FileExistsError:
        return _withheld([f"{target} exists; a reading is made once and never replaces a file"])
    for line in first:
        print(line)
    print(console, end="")
    print(f"\n{x1_finding}")
    if x3_banked is not None:
        print(f"\nX3 {x3_banked['family']}: {x3_banked['reading']}")
    print(f"\nbanked: {target}")
    if banked_reading:
        print(f"now anchor it: ots stamp {rel_target}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report the preconditions; open no cell")
    ap.add_argument("--out", default=None,
                    help="recompute into this new path, outside experiments/, after the banked reading")
    ap.add_argument("--write-manifest", action="store_true",
                    help="before the freeze only: hash the input files' bytes into the manifest")
    args = ap.parse_args(argv)
    if args.write_manifest:
        return write_manifest()
    if args.check:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out = pathlib.Path(args.out) if args.out else banked_path(today)
        problems = preconditions(x._scope(), out, args.out is None)
        print("\n".join(problems) if problems else "preconditions hold; the reading may run")
        return 2 if problems else 0
    return read(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
