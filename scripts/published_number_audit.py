#!/usr/bin/env python3
"""Addendum AD: does the fence move a published number?

LiveBench's `tablejoin` scorer scored a correct answer in a ```json fence
as 0.0 from its first public commit (2024-06-12) until the owner fixed it
on 2025-04-18 (commit e647d72), and an untagged fence too until
2024-10-22T01:43:34Z (f774d19). Its 2024 answers and its published tables
are public. This re-scores the stored answers with each historical scorer
version, as shipped (O) and with exactly one change (F: the shipped
`fencecheck.strip_fence`, applied only to an answer that opens with a
fence), and asks whether any published number moves up by a question's
worth.

Three stages, each made once, each refusing until the protocol's
OpenTimestamps proof is committed, parses as a proof of the committed
snapshot, carries a Bitcoin attestation with a block height, and every
file the reading runs matches the digest the snapshot quotes:

    python3 scripts/published_number_audit.py check   # preconditions only; touches nothing
    python3 scripts/published_number_audit.py fetch   # download exactly the manifest, verify (G0)
    python3 scripts/published_number_audit.py score   # G1, G2, every answer under every version, G3
    python3 scripts/published_number_audit.py read    # AD1 to AD3 and the words, by arithmetic

`restore` puts back a banked data file that went missing (data/ is not in
git) by downloading it again; it accepts only the exact bytes the banked
fetch received, so nothing it restores can differ from what was banked.

A stranger's recomputation, from a clone that holds the banked artifacts
but not data/ (each recompute needs --out, and names its inputs):

    python3 scripts/published_number_audit.py fetch --out /tmp/ad/fetch.json --data-dir /tmp/ad/data
    python3 scripts/published_number_audit.py score --out /tmp/ad/score.json --fetch /tmp/ad/fetch.json
    python3 scripts/published_number_audit.py read --out /tmp/ad/reading.json \
        --fetch /tmp/ad/fetch.json --score /tmp/ad/score.json

The freeze, with every file the reading runs (`required_files`; a test
keeps this command equal to it):

    python3 scripts/freeze_protocol.py prepare docs/research/ADDENDUM_AD_PROTOCOL.md \
        --code scripts/published_number_audit.py \
        --code tools/fencecheck.py \
        --code experiments/ad_manifest_2026-10-21.json \
        --code experiments/ad_synthetic_scorer_check_2026-10-21.json \
        --code scripts/ad_pinned/tablejoin_utils_905b71f.py \
        --code scripts/ad_pinned/tablejoin_utils_f774d19.py \
        --code scripts/ad_pinned/tablejoin_utils_19c7757.py \
        --code scripts/ad_pinned/tablejoin_utils_c5c293c.py \
        --code scripts/ad_pinned/tablejoin_utils_e647d72.py \
        --code scripts/ad_pinned/util_19c7757.py \
        --code scripts/ad_pinned/util_f3501c7.py \
        --code scripts/ad_pinned/gen_ground_truth_judgment_905b71f.py \
        --code scripts/ad_pinned/gen_ground_truth_judgment_f774d19.py \
        --code scripts/ad_pinned/gen_ground_truth_judgment_19c7757.py \
        --code scripts/ad_pinned/gen_ground_truth_judgment_fa6a3c6.py \
        --code scripts/ad_pinned/gen_ground_truth_judgment_f3501c7.py \
        --code scripts/ad_pinned/gen_ground_truth_judgment_e647d72.py
    # add the ANCHORS.md row and the VERDICT row, quoting the stamp
    git commit ...                                   # commit the prepared bytes
    python3 scripts/freeze_protocol.py stamp docs/research/ADDENDUM_AD_PROTOCOL.md
    git commit ...                                   # commit the proof
    # once a block is mined: ots upgrade, commit the upgraded proof, and
    # (outside this runner) ots verify against a node; only then may fetch run

Protocol: docs/research/ADDENDUM_AD_PROTOCOL.md. Manifest (every URL,
size and digest): experiments/ad_manifest_2026-10-21.json. LiveBench's
code is vendored byte for byte under scripts/ad_pinned/ (Apache-2.0,
LICENSE beside it) and executed, never retyped. `score` needs numpy
(LiveBench calls np.round) and pyarrow (the answers are one parquet);
`fetch`, `read` and `check` are stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap
import types
import urllib.request
import warnings
from collections import Counter
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

REPO = pathlib.Path(__file__).resolve().parent.parent
PROTOCOL_PATH = REPO / "docs" / "research" / "ADDENDUM_AD_PROTOCOL.md"
MANIFEST_PATH = REPO / "experiments" / "ad_manifest_2026-10-21.json"
SYNTHETIC_PATH = REPO / "experiments" / "ad_synthetic_scorer_check_2026-10-21.json"
FENCECHECK_PATH = REPO / "tools" / "fencecheck.py"
RUNNER_PATH = pathlib.Path(__file__).resolve()
FETCH_OUT = REPO / "experiments" / "ad_fetch_2026-10-21.json"
SCORE_OUT = REPO / "experiments" / "ad_score_2026-10-21.json"
READ_OUT = REPO / "experiments" / "ad_reading_2026-10-21.json"
ATTEMPTS_LOG = REPO / "experiments" / "ad_attempts_2026-10-21.jsonl"
DEFAULT_DATA_DIR = REPO / "data" / "ad"  # data/ is gitignored: 57 MB never enters git

OTS_MAGIC = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
ATTESTATIONS = {"bitcoin": bytes.fromhex("0588960d73d71901"),
                "pending": bytes.fromhex("83dfe30d2ef90c8e")}
STAMP_RE = re.compile(r"ADDENDUM_AD_PROTOCOL\.md\.(\d{4}-\d{2}-\d{2}T\d{4}Z)\.ots")
QUOTED_DIGEST_RE = re.compile(r"`([A-Za-z0-9_./-]+)` sha256 `([0-9a-f]{64})`")

TASK = "tablejoin"
MOVED_POINTS = 2.0          # one question's worth: one answer from 0 to 1 over 50
FLOAT_SLACK = 1e-9
REPRODUCED = "REPRODUCED"
NOT_REPRODUCED = "NOT REPRODUCED"
POSTDATES = "ANSWERS POSTDATE TABLE"
NO_ANSWERS = "NO ANSWERS"
FENCE_CLASSES = ("python", "json", "empty", "other", "one_line", "none")

# The five synthetic answers of the scoping run (survey section 4), to one
# synthetic ground truth passed as a JSON string, as the dataset stores it.
SYNTHETIC_GT = {"colA": "x1", "colB": "y2"}
SYNTHETIC_CASES = {
    "bare_dict": "{'colA': 'x1', 'colB': 'y2'}",
    "fence_python": "```python\n{'colA': 'x1', 'colB': 'y2'}\n```",
    "fence_json": '```json\n{"colA": "x1", "colB": "y2"}\n```',
    "fence_bare": '```\n{"colA": "x1", "colB": "y2"}\n```',
    "prose_then_fence_json": 'Here is the mapping:\n```json\n{"colA": "x1", "colB": "y2"}\n```',
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(data: bytes) -> str:
    """The id git gives these bytes, which a host's tree listing publishes."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_head(repo: pathlib.Path = REPO) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def same_path(a, b) -> bool:
    """Two paths name the same file, however they are spelled."""
    return pathlib.Path(a).resolve() == pathlib.Path(b).resolve()


def as_recorded(path: pathlib.Path) -> str:
    """How an artifact records a data path: relative to the repo when it
    is inside it, so a clone resolves it against its own checkout."""
    resolved = pathlib.Path(path).resolve()
    try:
        return resolved.relative_to(REPO).as_posix()
    except ValueError:
        return str(resolved)


def data_file(recorded: str) -> pathlib.Path:
    """The file a recorded data path names, in this checkout."""
    path = pathlib.Path(recorded)
    return path if path.is_absolute() else REPO / path


def _fsync_dir(directory: pathlib.Path) -> None:
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


# -- made once ------------------------------------------------------------------


def write_once(path: pathlib.Path, record: dict) -> None:
    """Exclusive and atomic: the bytes go to a fresh private temp file
    (a unique name, so a stale temp file left by a crashed process never
    blocks this one), are synced, and are hard-linked into place, which
    fails if the target exists. A reader never sees half a file, and
    nothing is overwritten."""
    data = (json.dumps(record, indent=1, ensure_ascii=False, sort_keys=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = pathlib.Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o644)
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise SystemExit(f"{path.name} is already banked; a stage is made once. "
                             "Recompute into another path with --out to compare.") from None
    finally:
        tmp.unlink(missing_ok=True)
    _fsync_dir(path.parent)


def append_attempt(stage: str, log: pathlib.Path | None = None, **extra) -> dict:
    """Bank the start clock before any work, append-only, so a stage that
    crashes still leaves its attempt on record and the next banked
    artifact lists it."""
    log = log or ATTEMPTS_LOG
    entry = {"stage": stage, "started_utc": utc_now(), "pid": os.getpid(), **extra}
    log.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return entry


def attempts(stage: str, log: pathlib.Path | None = None) -> list[dict]:
    log = log or ATTEMPTS_LOG
    if not log.exists():
        return []
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("stage") == stage]


# -- the anchor -----------------------------------------------------------------


# A minimal reader of the OpenTimestamps proof format, the same walk as
# Addendum AA's reader (scripts/decoder_isolation_read.py; a test checks
# the two agree): it names the file digest the proof commits to and every
# attestation it carries, with the block height of each Bitcoin one, and
# rejects anything that does not parse as a proof to its last byte. An
# attestation tag counts only where the format puts an attestation, never
# because its eight bytes occur somewhere in the file. It walks the
# operation tree; it does not recompute the merkle path or check a block
# header (`ots verify` does that against a node, outside this runner).

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


def anchored_digest(blob: bytes) -> str | None:
    """The file sha256 an OpenTimestamps proof commits to, or None unless
    it parses as a version-1 sha256 file proof carrying at least one
    Bitcoin or pending attestation where the format puts one."""
    parsed = parse_ots(blob)
    if parsed is None or not any(a["kind"] in ATTESTATIONS for a in parsed["attestations"]):
        return None
    return parsed["file_sha256"]


def attestation_kinds(blob: bytes) -> list[str]:
    parsed = parse_ots(blob)
    if parsed is None:
        return []
    return sorted({a["kind"] for a in parsed["attestations"] if a["kind"] in ATTESTATIONS})


def bitcoin_heights(blob: bytes) -> list[int]:
    parsed = parse_ots(blob)
    if parsed is None:
        return []
    return sorted({a["height"] for a in parsed["attestations"] if a["kind"] == "bitcoin"})


def stamp_of(protocol_text: str) -> str | None:
    match = STAMP_RE.search(protocol_text)
    return match.group(1) if match else None


def required_files(manifest: dict) -> list[str]:
    """Every code and data file the reading runs, repo-relative. Each must
    be quoted with its sha256 in the anchored snapshot."""
    files = {str(RUNNER_PATH.relative_to(REPO)), str(FENCECHECK_PATH.relative_to(REPO)),
             str(MANIFEST_PATH.relative_to(REPO)), str(SYNTHETIC_PATH.relative_to(REPO))}
    files.update(manifest.get("pinned_files", {}))
    return sorted(files)


def committed_unchanged(path: pathlib.Path, repo: pathlib.Path = REPO) -> bool:
    try:
        rel = str(path.resolve().relative_to(repo.resolve()))
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", rel], cwd=repo,
                                 capture_output=True, text=True).returncode == 0
        status = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=repo,
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return False
    return tracked and not status


def anchor_record(protocol_path: pathlib.Path = PROTOCOL_PATH, repo: pathlib.Path = REPO,
                  required: list[str] | None = None) -> dict:
    """Whether every stage may run: the proof is committed and parses as a
    proof of the committed snapshot, the protocol begins with the
    snapshot, every required file's digest is quoted in it and matches,
    and the parsed proof carries a Bitcoin attestation with a block
    height (an attestation cannot exist before its block). The heights
    are banked; the merkle path is not recomputed here (`ots verify`)."""
    record = {"ok": False}
    if not protocol_path.exists():
        record["why"] = "the protocol does not exist"
        return record
    text = protocol_path.read_text(encoding="utf-8")
    record["protocol_sha256_now"] = sha256_bytes(protocol_path.read_bytes())
    stamp = stamp_of(text)
    record["stamp"] = stamp
    if stamp is None:
        record["why"] = "the freeze line quotes no stamp: the protocol is not frozen"
        return record
    proof = protocol_path.with_name(f"{protocol_path.name}.{stamp}.ots")
    snapshot = protocol_path.with_name(f"snapshots_{protocol_path.stem}_{stamp}.md")
    record.update({"proof": proof.name, "snapshot": snapshot.name})
    if not proof.exists() or not snapshot.exists():
        record["why"] = "the proof or the anchored snapshot is missing"
        return record
    blob = proof.read_bytes()
    snap = snapshot.read_bytes()
    record.update({
        "proof_sha256": sha256_bytes(blob),
        "proof_parses": parse_ots(blob) is not None,
        "attestations": attestation_kinds(blob),
        "bitcoin_block_heights": bitcoin_heights(blob),
        "anchored_sha256": anchored_digest(blob),
        "snapshot_sha256": sha256_bytes(snap),
        "snapshot_committed_unchanged": committed_unchanged(snapshot, repo),
        "proof_committed_unchanged": committed_unchanged(proof, repo),
        "protocol_begins_with_the_snapshot": protocol_path.read_bytes().startswith(snap),
    })
    record["proof_commits_to_the_snapshot"] = record["anchored_sha256"] == record["snapshot_sha256"]
    quoted = dict(QUOTED_DIGEST_RE.findall(snap.decode("utf-8")))
    needed = required if required is not None else []
    now = {}
    for rel in needed:
        path = repo / rel
        now[rel] = sha256_file(path) if path.exists() else None
    missing = sorted(rel for rel in needed if rel not in quoted)
    differ = sorted(rel for rel in needed if rel in quoted and quoted[rel] != now[rel])
    record["code"] = {"required": needed, "quoted_in_snapshot": quoted, "now": now,
                      "not_quoted": missing, "differ": differ,
                      "matches": bool(needed) and not missing and not differ}
    # A calendar promise is not a timestamp: only a parsed Bitcoin
    # attestation with a block height counts.
    record["bitcoin_attested"] = bool(record["bitcoin_block_heights"])
    record["ok"] = bool(record["proof_commits_to_the_snapshot"]
                        and record["snapshot_committed_unchanged"]
                        and record["proof_committed_unchanged"]
                        and record["protocol_begins_with_the_snapshot"]
                        and record["code"]["matches"] and record["bitcoin_attested"])
    if not record["ok"]:
        reasons = []
        if not record["proof_parses"]:
            reasons.append("the proof does not parse as an OpenTimestamps file proof")
        elif not record["proof_commits_to_the_snapshot"]:
            reasons.append("the proof does not commit to the snapshot")
        if not record["snapshot_committed_unchanged"]:
            reasons.append("the snapshot is not committed, or changed since")
        if not record["proof_committed_unchanged"]:
            reasons.append("the proof is not committed, or changed since (commit it after ots upgrade)")
        if not record["protocol_begins_with_the_snapshot"]:
            reasons.append("the protocol does not begin with the snapshot")
        if not record["code"]["matches"]:
            reasons.append(f"code digests: not quoted {missing}, differ {differ}")
        if not record["bitcoin_attested"]:
            reasons.append("the proof carries no Bitcoin attestation yet (run ots upgrade)")
        record["why"] = "; ".join(reasons)
    return record


def load_manifest(path: pathlib.Path = MANIFEST_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# -- fetch (G0) -------------------------------------------------------------------


class NotReceived(Exception):
    """A transfer that did not complete: retry, never a digest that differs."""


def download(url: str, dest: pathlib.Path, cap: int, expect_sha256: str | None = None) -> dict:
    """Stream url to dest through a private temp file; sha256 while
    streaming; refuse more than cap bytes. A body shorter than the
    Content-Length the host announced is a cut transfer and raises
    NotReceived. The file is synced before it replaces dest. With
    expect_sha256, dest is replaced only by exactly those bytes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
    tmp = pathlib.Path(tmp_name)
    request = urllib.request.Request(url, headers={"User-Agent": "arc-ttt-addendum-ad/1"})
    h = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(request, timeout=120) as response:
            status = getattr(response, "status", None)
            headers = getattr(response, "headers", None)
            announced = headers.get("Content-Length") if headers is not None else None
            for chunk in iter(lambda: response.read(1 << 20), b""):
                size += len(chunk)
                if size > cap:
                    raise ValueError(f"more than {cap} bytes")
                h.update(chunk)
                out.write(chunk)
            if announced is not None and announced.strip().isdigit() and int(announced) != size:
                raise NotReceived(f"transfer cut: {size} of the {int(announced)} bytes announced")
            out.flush()
            os.fsync(out.fileno())
        if expect_sha256 is not None and h.hexdigest() != expect_sha256:
            raise ValueError(f"the bytes received ({h.hexdigest()[:12]}) are not the banked "
                             f"bytes ({expect_sha256[:12]}); nothing replaced")
        os.replace(tmp, dest)
        _fsync_dir(dest.parent)
    finally:
        tmp.unlink(missing_ok=True)
    return {"status": status, "bytes": size, "sha256": h.hexdigest(), "fetched_utc": utc_now()}


def local_name(entry: dict) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", entry["id"])


def verify_entry(entry: dict, data: bytes) -> dict:
    got = {"bytes": len(data), "sha256": sha256_bytes(data), "git_blob_sha1": git_blob_sha1(data)}
    checks = {}
    if entry.get("bytes") is not None:
        checks["bytes"] = got["bytes"] == entry["bytes"]
    if entry.get("sha256"):
        checks["sha256"] = got["sha256"] == entry["sha256"]
    if entry.get("git_blob_sha1"):
        checks["git_blob_sha1"] = got["git_blob_sha1"] == entry["git_blob_sha1"]
    return {"got": got, "checks": checks, "matches": bool(checks) and all(checks.values())}


def fetch_all(manifest: dict, data_dir: pathlib.Path, root: pathlib.Path = REPO,
              downloader=download) -> dict:
    """Download exactly the remote files the manifest names and verify
    each. A file that was not received in full is `unreceived`: nothing
    is banked and fetch may run again. That covers any transfer that
    raised, for every entry, and a `refuse` entry whose body is not the
    size the manifest pins (every refuse entry is an immutable file at a
    pinned commit or revision, so a size that differs is a transfer
    fault). A `refuse` entry received at its pinned size whose digest
    differs fails G0. A `record` entry (the live tables, the scoping
    page) received in full is banked as found: a difference publishes
    first and stops nothing."""
    results = []
    for entry in manifest["remote"]:
        dest = data_dir / local_name(entry)
        cap = int((entry.get("bytes") or 0) * 2 + 4_000_000)
        row = {"id": entry["id"], "url": entry["url"], "on_mismatch": entry["on_mismatch"],
               "role": entry.get("role"), "path": as_recorded(dest)}
        try:
            row["received"] = downloader(entry["url"], dest, cap)
            row.update(verify_entry(entry, dest.read_bytes()))
            if (entry["on_mismatch"] == "refuse" and entry.get("bytes") is not None
                    and row["got"]["bytes"] != entry["bytes"]):
                row["not_received"] = (f"{row['got']['bytes']} bytes received, the manifest pins "
                                       f"{entry['bytes']}: a transfer fault, retry")
        except Exception as error:  # noqa: BLE001
            row.update({"error": f"{type(error).__name__}: {error}", "matches": False})
        expect = entry.get("expect_equal_to")
        if expect is not None:
            # a live copy is judged by its bytes against its pinned twin,
            # never by its size alone
            twin = next((r for r in results if r["id"] == expect), None)
            row["matches"] = bool(twin and twin.get("got") and row.get("got")
                                  and twin["got"]["sha256"] == row["got"]["sha256"])
            row["equal_to_pinned_twin"] = row["matches"]
        results.append(row)
    # The vendored files are the remote files, byte for byte.
    by_commit_path = {(e.get("commit"), e.get("path")): e["id"] for e in manifest["remote"]
                      if e.get("role") == "scorer_source"}
    by_id = {r["id"]: r for r in results}
    vendored = []
    for rel, digest in sorted(manifest.get("pinned_files", {}).items()):
        local = root / rel
        source = manifest.get("pinned_sources", {}).get(rel)
        remote_row = by_id.get(by_commit_path.get((source["commit"], source["path"]))) if source else None
        remote_sha = remote_row["got"]["sha256"] if remote_row and remote_row.get("got") else None
        now = sha256_file(local) if local.exists() else None
        vendored.append({"file": rel, "manifest_sha256": digest, "local_sha256": now,
                         "remote_sha256": remote_sha, "source": source,
                         "matches": now == digest and (source is None or remote_sha == digest)})
    unreceived = [r["id"] for r in results if "error" in r or "not_received" in r]
    refused = [r["id"] for r in results if r["on_mismatch"] == "refuse" and not r.get("matches")
               and r["id"] not in unreceived]
    recorded = [r["id"] for r in results if r["on_mismatch"] == "record" and not r.get("matches")
                and r["id"] not in unreceived]
    vendored_bad = [v["file"] for v in vendored if not v["matches"]]
    states = verify_first_parent_states(manifest, results)
    consistency = version_consistency(manifest)
    g0 = {"passed": not refused and not vendored_bad and states["matches"] and not consistency,
          "refused_entries": refused, "vendored_files_that_differ": vendored_bad,
          "first_parent_states": states, "version_list_problems": consistency,
          "recorded_differences_published_first": recorded,
          "recorded_roles": {r["id"]: r.get("role") for r in results if r["id"] in recorded}}
    return {"remote": results, "vendored": vendored, "g0": g0, "unreceived": unreceived}


def version_consistency(manifest: dict) -> list[str]:
    """The version list is the first-parent states de-duplicated, in order,
    by the tuple (tablejoin utils, util, extraction) digests, and each
    version's vendored files are that state's files."""
    lb = manifest["livebench"]
    pinned = manifest["pinned_files"]
    problems = []
    expected, last = [], None
    for state in lb["first_parent_states"]:
        key = (state["tablejoin_utils_sha256"], state.get("util_sha256"), state["extraction_sha256"])
        if key != last:
            expected.append((state["commit"], key))
            last = key
    got = [v["commit"] for v in lb["versions"]]
    if got != [c for c, _ in expected]:
        problems.append(f"versions {[c[:7] for c in got]} are not the de-duplicated states "
                        f"{[c[:7] for c, _ in expected]}")
    states = {s["commit"]: s for s in lb["first_parent_states"]}
    for v in lb["versions"]:
        s = states.get(v["commit"])
        if s is None:
            problems.append(f"{v['id']}: not a first-parent state")
            continue
        if pinned.get(v["tablejoin_utils"]) != s["tablejoin_utils_sha256"]:
            problems.append(f"{v['id']}: tablejoin utils file is not the state's")
        if (pinned.get(v["util"]) if v.get("util") else None) != s.get("util_sha256"):
            problems.append(f"{v['id']}: util file is not the state's")
        if pinned.get(v["gen_ground_truth_judgment"]) != s["gen_ground_truth_judgment_sha256"]:
            problems.append(f"{v['id']}: gen_ground_truth_judgment file is not the state's")
        if v["extraction_sha256"] != s["extraction_sha256"]:
            problems.append(f"{v['id']}: extraction digest is not the state's")
        if parse_utc(v["in_force_from_utc"]) != parse_utc(s["committer_time"]):
            problems.append(f"{v['id']}: in force from a time other than its landing on main")
    return problems


def verify_first_parent_states(manifest: dict, results: list[dict]) -> dict:
    """Each state of LiveBench's main in the window, re-derived from the
    downloaded files, has the digest tuple the manifest says, so the
    version list (states de-duplicated by that tuple) is what was frozen."""
    by_id = {r["id"]: r for r in results}
    problems = []
    for state in manifest["livebench"]["first_parent_states"]:
        for key, entry_id in state.get("remote_ids", {}).items():
            row = by_id.get(entry_id)
            if row is None or not row.get("got"):
                problems.append(f"{state['commit'][:7]} {key}: not fetched")
                continue
            if row["got"]["sha256"] != state[key]:
                problems.append(f"{state['commit'][:7]} {key}: differs")
            if key == "gen_ground_truth_judgment_sha256":
                try:
                    data = data_file(row["path"]).read_text(encoding="utf-8")
                    if extraction_sha256(data) != state["extraction_sha256"]:
                        problems.append(f"{state['commit'][:7]} extraction differs")
                except Exception as error:  # noqa: BLE001
                    problems.append(f"{state['commit'][:7]} extraction unreadable: {error!r}")
    return {"checked": len(manifest["livebench"]["first_parent_states"]), "problems": problems,
            "matches": not problems}


# -- LiveBench's code, executed as shipped ------------------------------------------


def _assigns(stmt: ast.stmt, name: str) -> bool:
    return (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == name)


def extraction_source(gg_source: str) -> str:
    """The exact statements of `play_a_match_gt` that turn a stored answer
    into the text the scorer sees: the run of consecutive assignments to
    `llm_answer`, as written in that version of the file."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # LiveBench's f"<think>.*?<\/think>" escape
        tree = ast.parse(gg_source)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "play_a_match_gt")
    start = next(i for i, s in enumerate(fn.body) if _assigns(s, "llm_answer"))
    end = start
    while end < len(fn.body) and _assigns(fn.body[end], "llm_answer"):
        end += 1
    return "\n".join(ast.get_source_segment(gg_source, s) for s in fn.body[start:end])


def extraction_sha256(gg_source: str) -> str:
    return sha256_bytes(extraction_source(gg_source).encode("utf-8"))


def compile_extraction(gg_source: str, name: str = "extraction"):
    body = extraction_source(gg_source)
    src = "def _extract(answer):\n" + textwrap.indent(body, "    ") + "\n    return llm_answer\n"
    namespace: dict = {"re": re}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # LiveBench's f"<think>.*?<\/think>" escape
        exec(compile(src, f"<{name}>", "exec"), namespace)  # noqa: S102 -- their statements, verbatim
    return namespace["_extract"]


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Version:
    """One state of LiveBench's scorer: its tablejoin utils, the util module
    of the same commit, and the answer extraction of the same commit."""

    def __init__(self, spec: dict, root: pathlib.Path = REPO):
        self.id = spec["id"]
        self.spec = spec
        self.start = parse_utc(spec["in_force_from_utc"])
        saved = {k: sys.modules.get(k) for k in
                 ("livebench", "livebench.process_results", "livebench.process_results.util")}
        try:
            if spec.get("util"):
                util = _load_module(f"ad_pinned_util_{self.id}", root / spec["util"])
                sys.modules["livebench"] = types.ModuleType("livebench")
                sys.modules["livebench.process_results"] = types.ModuleType("livebench.process_results")
                sys.modules["livebench.process_results.util"] = util
            module = _load_module(f"ad_pinned_tablejoin_{self.id}", root / spec["tablejoin_utils"])
        finally:
            for key, value in saved.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value
        self.joinmap = module.joinmap_process_results
        gg = (root / spec["gen_ground_truth_judgment"]).read_text(encoding="utf-8")
        self.extraction_source = extraction_source(gg)
        self.extract = compile_extraction(gg, f"extraction {self.id}")

    def score(self, question_text, ground_truth, text):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return float(self.joinmap(question_text, ground_truth, text))
        except Exception as error:  # noqa: BLE001 -- recorded, never hidden
            return f"error: {type(error).__name__}"


def load_versions(manifest: dict, root: pathlib.Path = REPO) -> list[Version]:
    versions = [Version(spec, root) for spec in manifest["livebench"]["versions"]]
    if [v.start for v in versions] != sorted(v.start for v in versions):
        raise SystemExit("manifest versions are not in time order")
    return versions


def load_fencecheck(path: pathlib.Path = FENCECHECK_PATH):
    return _load_module("fencecheck_for_ad", path)


def fence_class(text: str) -> tuple[str, str | None]:
    """Where a leading fence opens and what its info string is. Exact case:
    LiveBench's regexes are case-sensitive, so ```Python is `other`."""
    s = text.strip()
    if not s.startswith("```"):
        return "none", None
    if "\n" not in s:
        return "one_line", None
    info = s.split("\n", 1)[0][3:]
    tag = info.strip()
    if tag == "python":
        return "python", info[:40]
    if tag == "json":
        return "json", info[:40]
    if tag == "":
        return "empty", info[:40]
    return "other", info[:40]


def score_one(version: Version, question: dict, answer: dict, fencecheck,
              wide_cache: dict | None = None) -> dict:
    """O: the version's extraction, then its scorer, as shipped. F: the same
    with exactly one change -- an answer that opens with a fence is passed
    through strip_fence first; any other answer is scored byte for byte as
    O was, so F equals O by construction. A (AD-S2, counts only): the
    opt-in wider scope in place of the leading strip. wide_cache only
    saves recomputing the wider scope's search on a text already seen."""
    try:
        text = version.extract(answer)
    except Exception as error:  # noqa: BLE001
        bad = f"error: extraction {type(error).__name__}"
        return {"O": bad, "F": bad, "A": bad, "fence": "none"}
    if not isinstance(text, str):
        bad = "error: extraction gave no text"
        return {"O": bad, "F": bad, "A": bad, "fence": "none"}
    q_text, gt = question["turns"][0], question["ground_truth"]
    o = version.score(q_text, gt, text)
    body, was_fenced = fencecheck.strip_fence(text)
    f = version.score(q_text, gt, body) if was_fenced else o
    if wide_cache is not None and text in wide_cache:
        wide, kind = wide_cache[text]
    else:
        wide, kind = fencecheck.extract_json(text, scope="any")
        if wide_cache is not None:
            wide_cache[text] = (wide, kind)
    a = version.score(q_text, gt, wide) if kind != "none" else o
    klass, info = fence_class(text)
    row = {"O": o, "F": f, "A": a, "fence": klass}
    if info is not None and klass == "other":
        row["info"] = info
    return row


def synthetic_check(versions: list[Version], expected: dict, fencecheck) -> dict:
    """G1: each version scores the five synthetic answers exactly as the
    scoping run did, for O and for F."""
    gt = json.dumps(SYNTHETIC_GT)
    results, mismatches = {}, []
    for v in versions:
        key = v.spec["synthetic_expectation_key"]
        row = {}
        for name, text in SYNTHETIC_CASES.items():
            o = v.score(None, gt, text)
            body, was_fenced = fencecheck.strip_fence(text)
            f = v.score(None, gt, body) if was_fenced else o
            row[name] = {"O": o, "F": f}
            want = expected.get(key, {}).get(name)
            if want != row[name]:
                mismatches.append({"version": v.id, "expectation_key": key, "case": name,
                                   "got": row[name], "expected": want})
        results[v.id] = row
    return {"passed": not mismatches, "results": results, "mismatches": mismatches}


# -- the published tables ------------------------------------------------------------


def parse_table(text: str) -> tuple[list[str], list[dict]]:
    import csv
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    rows = [row for row in reader if any((v or "").strip() for v in row.values())]
    return list(reader.fieldnames or []), rows


_JS_FLOAT = re.compile(r"^[\t\n\v\f\r  ﻿]*([+-]?(?:Infinity|\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?))")


def js_parse_float(value) -> float:
    """JavaScript's parseFloat, which the site applies to every cell."""
    if isinstance(value, (int, float)):
        return float(value)
    match = _JS_FLOAT.match(value or "")
    if not match:
        return math.nan
    token = match.group(1)
    if token.lstrip("+-") == "Infinity":
        return -math.inf if token.startswith("-") else math.inf
    return float(token)


def to_fixed(value: float, digits: int = 2) -> str:
    """JavaScript's Number.prototype.toFixed: the exact binary value, ties
    away from zero."""
    quantum = Decimal(1).scaleb(-digits)
    return str(Decimal(value).quantize(quantum, rounding=ROUND_HALF_UP))


def calculate_average(row: dict, columns: list[str] | None):
    """Averaging.js calculateAverage: mean of the parseable cells, '-' if none."""
    if not columns:
        return "-"
    values = [js_parse_float(row.get(c)) for c in columns]
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else "-"


def global_average(row: dict, categories: dict):
    """Averaging.js getGlobalAverage with every category checked (the
    site's default): mean of the category averages, toFixed(2); '-' when
    any category has no value (JavaScript's string concatenation makes the
    sum NaN). Two rows are hard-coded by the site."""
    if row.get("model") == "grok-3-thinking":
        return 72.0
    if row.get("model") == "grok-3":
        return 58.0
    averages = [calculate_average(row, cols) for cols in categories.values()]
    if not averages:
        return 0.0
    if any(isinstance(a, str) for a in averages):
        return "-"
    avg = sum(averages) / len(averages)
    return float(to_fixed(avg, 2))


def competition_rank(scores: dict) -> dict:
    """Position by displayed Global Average, descending; ties share the
    higher position (1, 2, 2, 4); rows without a number are unranked."""
    numeric = {k: v for k, v in scores.items() if isinstance(v, float)}
    ordered = sorted(numeric.values(), reverse=True)
    first = {}
    for i, value in enumerate(ordered, start=1):
        first.setdefault(value, i)
    return {k: (first[v] if k in numeric else None) for k, v in scores.items()}


def decimals(cell: str) -> int | None:
    try:
        exponent = Decimal(cell.strip()).as_tuple().exponent
    except (InvalidOperation, AttributeError):
        return None
    if not isinstance(exponent, int):
        return None
    return max(0, -exponent)


def column_tolerance(cells: list[str]) -> tuple[int | None, float | None]:
    """A column printed to d decimals reproduces within half a unit of the
    d-th decimal. d is the most decimals any cell of the column carries
    (a writer drops trailing zeros), capped at three: LiveBench's results
    script rounds to three (show_livebench_result.py at 905b71f)."""
    ds = [d for d in (decimals(c) for c in cells) if d is not None]
    if not ds:
        return None, None
    d = min(max(ds), 3)
    return d, 0.5 * 10 ** (-d)


def match_models(names: list[str], model_ids: list[str]) -> dict:
    """A row is in frame when its name equals an answer-set model_id
    exactly, or case-insensitively when exactly one id matches. Ids with a
    '/' (local paths) are out, as LiveBench's download_leaderboard.py
    skips them. Aliases are never guessed."""
    usable = [m for m in model_ids if "/" not in m]
    exact = set(usable)
    lowered: dict[str, list[str]] = {}
    for m in usable:
        lowered.setdefault(m.lower(), []).append(m)
    out = {}
    for name in names:
        if name in exact:
            out[name] = {"model_id": name, "match": "exact"}
        elif len(lowered.get(name.lower(), [])) == 1:
            out[name] = {"model_id": lowered[name.lower()][0], "match": "case-insensitive"}
        elif len(lowered.get(name.lower(), [])) > 1:
            out[name] = {"model_id": None, "match": "ambiguous",
                         "candidates": sorted(lowered[name.lower()])}
        else:
            out[name] = {"model_id": None, "match": "unmatched"}
    return out


def in_force(versions: list[dict], lower: datetime, upper: datetime,
             window_closes: datetime | None = None) -> list[str]:
    """Versions whose time on main, [its landing, the next version's
    landing), overlaps [lower, upper]: lower is the newest of the model's
    stored answers (a cell cannot be scored before its answers exist),
    upper is the committer time of the last commit that touched the
    table. Landing is a commit's committer time on both sides (push time
    is not public). The last version's time ends where the pinned window
    closes (the next change to any of the three files on main); nothing
    later is pinned."""
    if lower > upper:
        return []  # the answers are newer than the table: no version scored them
    out = []
    for i, v in enumerate(versions):
        start = parse_utc(v["in_force_from_utc"])
        end = parse_utc(versions[i + 1]["in_force_from_utc"]) if i + 1 < len(versions) else window_closes
        if start <= upper and (end is None or end > lower):
            out.append(v["id"])
    return out


def cell_value(rows: list[dict], vid: str, arm: str) -> tuple[float | None, int]:
    """Mean over the model's answers of score x 100 (LiveBench's units).
    O is undefined if any answer raised (the shipped run would have
    stopped). For F and A, an answer that raised keeps its O; the count is
    banked beside the cell."""
    vals, errors = [], 0
    for r in rows:
        o = r["v"][vid]["O"]
        x = r["v"][vid][arm]
        if arm == "O":
            if not isinstance(x, float):
                return None, 1
            vals.append(x * 100)
            continue
        if not isinstance(x, float):
            errors += 1
            if not isinstance(o, float):
                return None, errors
            x = o
        vals.append(x * 100)
    if not vals:
        return None, errors
    return sum(vals) / len(vals), errors


def build_cells(tables: dict, per_answer: list[dict], versions: list[dict],
                model_ids: list[str], window_closes: datetime | None = None) -> dict:
    """Every published tablejoin cell whose row names a model of the answer
    set: its in-force versions, O, F and A under each, and whether some
    in-force version reproduces it from the original scorer. A cell that
    no in-force version reproduces is NOT REPRODUCED: banked, listed, and
    never counted toward a bar. A reproduced cell banks F - O under every
    version that reproduces it, the smallest (`delta`, the claim's
    reading) and the largest (`delta_max`, the predictions' reading)."""
    by_model: dict[str, list[dict]] = {}
    for r in per_answer:
        by_model.setdefault(r["model_id"], []).append(r)
    order = [v["id"] for v in versions]
    cells, frame = [], {}
    for date, table in tables.items():
        header, rows = parse_table(table["text"])
        info = {"rows": len(rows), "has_model_column": "model" in header,
                "has_tablejoin_column": TASK in header}
        frame[date] = info
        if not info["has_model_column"] or not info["has_tablejoin_column"]:
            info["unreadable"] = True
            continue
        d_col, tol = column_tolerance([r.get(TASK) or "" for r in rows])
        info.update({"decimals": d_col, "tolerance_points": tol})
        matches = match_models([r["model"] for r in rows], model_ids)
        info["unmatched"] = sorted(n for n, m in matches.items() if m["match"] == "unmatched")
        info["ambiguous"] = sorted(n for n, m in matches.items() if m["match"] == "ambiguous")
        info["no_published_value"] = []
        info["postdates_cells_the_draft_lower_bound_kept"] = 0
        upper = parse_utc(table["upper_utc"])
        # the scoping draft's lower bound was the table's first commit; the
        # cells its rule would have kept and this rule does not are counted
        first = parse_utc(table["first_utc"]) if table.get("first_utc") else None
        for index, row in enumerate(rows):
            m = matches[row["model"]]
            if m["model_id"] is None:
                continue
            published = (row.get(TASK) or "").strip()
            value = js_parse_float(published)
            if published == "" or math.isnan(value):
                info["no_published_value"].append(row["model"])
                continue
            answers = by_model.get(m["model_id"], [])
            cell = {"table": date, "row": index, "model": row["model"], "model_id": m["model_id"],
                    "match": m["match"], "published": published, "published_value": value,
                    "tolerance_points": tol, "n_answers": len(answers), "per_version": {}}
            if not answers:
                cell.update({"status": NO_ANSWERS, "in_force": []})
                cells.append(cell)
                continue
            stamps = [a["tstamp"] for a in answers if isinstance(a.get("tstamp"), (int, float))]
            # no timestamp: no lower bound, so more versions are in force, never fewer
            lower = (datetime.fromtimestamp(max(stamps), timezone.utc) if stamps
                     else datetime(2000, 1, 1, tzinfo=timezone.utc))
            cell["answers_without_tstamp"] = len(answers) - len(stamps)
            cell["lower_utc"], cell["upper_utc"] = iso(lower), iso(upper)
            cell["in_force"] = in_force(versions, lower, upper, window_closes)
            if first is not None:
                cell["in_force_from_the_tables_first_commit"] = in_force(versions, first, upper,
                                                                         window_closes)
                if not cell["in_force"] and cell["in_force_from_the_tables_first_commit"]:
                    info["postdates_cells_the_draft_lower_bound_kept"] += 1
            for vid in cell["in_force"]:
                o, o_err = cell_value(answers, vid, "O")
                f, f_err = cell_value(answers, vid, "F")
                a, a_err = cell_value(answers, vid, "A")
                diff = None if o is None else abs(value - o)
                cell["per_version"][vid] = {
                    "O": o, "F": f, "A": a, "o_errors": o_err, "f_errors": f_err,
                    "a_errors": a_err, "abs_published_minus_O": diff,
                    "reproduces": diff is not None and f is not None and diff <= tol + FLOAT_SLACK}
            repro = [vid for vid in cell["in_force"] if cell["per_version"][vid]["reproduces"]]
            if not cell["in_force"]:
                cell["status"] = POSTDATES
            elif not repro:
                cell["status"] = NOT_REPRODUCED
            else:
                deltas = {vid: cell["per_version"][vid]["F"] - cell["per_version"][vid]["O"]
                          for vid in repro}
                # Several versions reproduce and disagree. Only an upward
                # F - O can be the fence failure's (a downward one is the
                # strip's own error), so the smallest F - O is the reading
                # against a positive: a cell moves up only if it moves up
                # under every version that reproduces it, and a downward
                # strip error is counted wherever one version shows it.
                # The largest is banked beside it; the predictions use it.
                chosen = min(repro, key=lambda vid: (deltas[vid], order.index(vid)))
                top = max(repro, key=lambda vid: (deltas[vid], -order.index(vid)))
                up = down = 0
                for a in answers:
                    o_a, f_a = a["v"][chosen]["O"], a["v"][chosen]["F"]
                    if isinstance(o_a, float) and isinstance(f_a, float):
                        up += f_a > o_a
                        down += f_a < o_a
                cell.update({"status": REPRODUCED, "reproduced_by": repro, "deltas": deltas,
                             "chosen_version": chosen, "delta": deltas[chosen],
                             "delta_max_version": top, "delta_max": deltas[top],
                             "answers_up": up, "answers_down": down,
                             "O": cell["per_version"][chosen]["O"],
                             "F": cell["per_version"][chosen]["F"]})
            cells.append(cell)
    return {"cells": cells, "frame": frame}


# -- score --------------------------------------------------------------------------


def load_parquet_rows(path: pathlib.Path, columns: list[str], task: str = TASK) -> list[dict]:
    import pyarrow.parquet as pq
    rows = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=2048, columns=columns):
        block = batch.to_pydict()
        for i, value in enumerate(block["task"]):
            if value == task:
                rows.append({c: block[c][i] for c in columns})
    return rows


def score_answers(questions: dict, answers: list[dict], versions: list[Version], fencecheck) -> list[dict]:
    out = []
    for answer in answers:
        question = questions[answer["question_id"]]
        wide_cache: dict = {}  # per answer: the versions' extractions give at most two texts
        out.append({"model_id": answer["model_id"], "question_id": answer["question_id"],
                    "tstamp": answer["tstamp"],
                    "v": {v.id: score_one(v, question, answer, fencecheck, wide_cache)
                          for v in versions}})
    return out


def dedupe_answers(rows: list[dict], questions: dict) -> tuple[list[dict], dict]:
    """LiveBench's loader keeps the last answer per (model, question) id;
    so does this, in parquet order. Answers to unknown questions are out."""
    kept: dict[tuple, dict] = {}
    duplicates, unknown = Counter(), Counter()
    for row in rows:
        if row["question_id"] not in questions:
            unknown[row["model_id"]] += 1
            continue
        key = (row["model_id"], row["question_id"])
        if key in kept:
            duplicates[row["model_id"]] += 1
        kept[key] = row
    return list(kept.values()), {"duplicates_by_model": dict(duplicates),
                                 "unknown_question_by_model": dict(unknown)}


def feasibility_check(page_bytes: bytes | None, page_matches_scoping: bool,
                      parquet_answers: list[dict], disclosed: dict) -> dict:
    """G2: the scoping page is the parquet. Never stops the run; a
    difference publishes first."""
    model = disclosed["model_id"]
    mine = {a["question_id"]: a for a in parquet_answers if a["model_id"] == model}
    tags = Counter()
    for a in mine.values():
        s = a["choices"][0]["turns"][0].strip()
        if s.startswith("```"):
            tag = s[3:].split("\n", 1)[0].strip().lower()   # the scoping script's rule
            tags[tag or "(bare)"] += 1
        else:
            tags["(no leading fence)"] += 1
    record = {"model_id": model, "parquet_answers": len(mine),
              "leading_fence_info_strings_scoping_rule": dict(tags),
              "tags_match_disclosure": dict(tags) == disclosed["leading_fence_info_strings"]
              and len(mine) == disclosed["answers"],
              "page_matches_scoping_digest": page_matches_scoping}
    if page_bytes is None:
        record["texts_equal"] = None
        record["why"] = "the page could not be fetched"
        return record
    try:
        page = json.loads(page_bytes)
        differ, truncated, compared = [], 0, 0
        for item in page.get("rows", []):
            row = item["row"]
            if "choices" in (item.get("truncated_cells") or []):
                truncated += 1
                continue
            qid = row["question_id"]
            compared += 1
            if qid not in mine or mine[qid]["choices"][0]["turns"] != row["choices"][0]["turns"]:
                differ.append(qid)
        record.update({"compared": compared, "truncated": truncated, "differ": differ,
                       "texts_equal": compared == len(mine) and not differ and truncated == 0})
    except Exception as error:  # noqa: BLE001
        record.update({"texts_equal": None, "why": f"page unreadable: {error!r}"})
    return record


def environment() -> dict:
    import resource
    env = {"python": sys.version.split()[0], "platform": sys.platform,
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    try:
        env["nice"] = os.nice(0)
    except OSError:
        pass
    for name in ("numpy", "pyarrow"):
        try:
            env[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            env[name] = None
    return env


def lower_priority() -> None:
    try:
        current = os.nice(0)
        if current < 19:
            os.nice(19 - current)
    except OSError:
        pass


# -- the readings, by arithmetic --------------------------------------------------------


def verdict_words(n_models: int) -> str:
    if n_models == 0:
        return "NULL"
    if n_models <= 2:
        return "MOVES A NUMBER"
    return "MOVES NUMBERS"


def version_on_main_at(versions: list[dict], when: datetime | None) -> str | None:
    """The pinned version that was on main at `when`; the first version
    when `when` is unknown or precedes them all."""
    if not versions:
        return None
    chosen = versions[0]["id"]
    if when is None:
        return chosen
    for v in versions:
        if parse_utc(v["in_force_from_utc"]) <= when:
            chosen = v["id"]
    return chosen


def _direction(o, f) -> str | None:
    if not isinstance(o, float) or not isinstance(f, float):
        return None
    return "up" if f > o else "down" if f < o else "equal"


def ad1(score: dict) -> dict:
    """Answers whose score moves: every stored answer of every model with
    an in-frame cell, once, under the reading version of the earliest
    table that shows the model: the version that reproduced that cell
    (the smallest F - O); else the earliest version in force for it;
    else (no version is in force for it: the answers postdate the table)
    the version on main at the model's newest stored answer. Beside it,
    the harder count the predictions use: an answer counts as up (down)
    when F > O (F < O) under any version in force for one of its model's
    cells, or under the reading version when none is."""
    order = {d: i for i, d in enumerate(score["table_order"])}
    versions = score.get("versions") or []
    earliest: dict[str, tuple] = {}
    in_force_any: dict[str, set] = {}
    for cell in score["cells"]:
        m = cell["model_id"]
        key = (order[cell["table"]], cell["row"])
        if m not in earliest or key < earliest[m][0]:
            earliest[m] = (key, cell)
        in_force_any.setdefault(m, set()).update(cell.get("in_force") or [])
    newest: dict[str, float] = {}
    with_answers = set()
    for r in score["per_answer"]:
        with_answers.add(r["model_id"])
        t = r.get("tstamp")
        if isinstance(t, (int, float)):
            newest[r["model_id"]] = max(newest.get(r["model_id"], t), t)
    reading_version: dict[str, dict] = {}
    for m, (_, cell) in earliest.items():
        if cell.get("chosen_version"):
            vid, rule = cell["chosen_version"], "the version that reproduced the earliest table's cell"
        elif cell.get("in_force"):
            vid, rule = cell["in_force"][0], "the earliest version in force for the earliest table's cell"
        else:
            when = datetime.fromtimestamp(newest[m], timezone.utc) if m in newest else None
            vid = version_on_main_at(versions, when)
            rule = "no version in force for the earliest table's cell: the version on main at the newest answer"
        reading_version[m] = {"version": vid, "table": cell["table"], "rule": rule}
    split = {k: Counter() for k in FENCE_CLASSES}
    totals = Counter()
    harder = Counter()
    other_tags = Counter()
    for r in score["per_answer"]:
        rv = reading_version.get(r["model_id"])
        if rv is None or rv["version"] not in r["v"]:
            continue  # a model with no in-frame cell
        row = r["v"][rv["version"]]
        o, f = row["O"], row["F"]
        klass = row["fence"]
        totals["answers"] += 1
        split[klass]["answers"] += 1
        if klass != "none":
            totals["open_with_a_fence"] += 1
        if klass == "other":
            other_tags[row.get("info", "")] += 1
        considered = sorted(in_force_any.get(r["model_id"]) or {rv["version"]})
        moves = {_direction(r["v"][v]["O"], r["v"][v]["F"]) for v in considered if v in r["v"]}
        harder["answers"] += 1
        harder["up"] += "up" in moves
        harder["down"] += "down" in moves
        if not isinstance(o, float):
            totals["o_raised"] += 1
            split[klass]["o_raised"] += 1
            continue
        if not isinstance(f, float):
            totals["f_raised"] += 1
            split[klass]["f_raised"] += 1
            continue
        direction = _direction(o, f)
        totals[direction] += 1
        split[klass][direction] += 1
        if o == 0.0 and f == 1.0:
            totals["full_flips"] += 1
            split[klass]["full_flips"] += 1
    no_version = sorted(m for m in reading_version if m in with_answers and not in_force_any.get(m))
    return {"totals": dict(totals), "by_opening_fence": {k: dict(v) for k, v in split.items()},
            "other_info_strings": dict(other_tags),
            "any_version_in_force": dict(harder),
            "reading_version_by_model": dict(sorted(reading_version.items())),
            "models_with_no_version_in_force": no_version}


def row_after(table_rows: list[dict], replace: dict[int, float]) -> list[dict]:
    out = []
    for i, row in enumerate(table_rows):
        row = dict(row)
        if i in replace:
            row[TASK] = replace[i]
        out.append(row)
    return out


def ad2_ad3(score: dict, tables: dict, reading: str = "lower") -> dict:
    """AD2 and AD3 under one reading of each reproduced cell's F - O:
    `lower`, the smallest over the versions that reproduce it (the claim
    and its sentence), or `upper`, the largest (the predictions). A cell
    MOVED UP when F - O >= 2.0, one question's worth: only an upward move
    can be the fence failure's. A cell with F - O <= -2.0 is the strip's
    own error: it is counted and published beside the words and never
    counts toward them. AD3 adds each reproduced cell's upward F - O (0
    when it is not upward) to its published value.

    `as_drafted` is the rule the predictions were written under (the
    smallest F - O; a move in either direction counts; AD3 adds the whole
    F - O). It licenses no sentence. It exists so that the change of rule
    cannot make a prediction easier to hold: AD2 NULL and AD3 NO RANK
    CHANGED are checked under `upper` and under `as_drafted`."""
    key = {"lower": "delta", "upper": "delta_max", "as_drafted": "delta"}[reading]
    vkey = {"lower": "chosen_version", "upper": "delta_max_version",
            "as_drafted": "chosen_version"}[reading]
    both_ways = reading == "as_drafted"
    threshold = MOVED_POINTS - FLOAT_SLACK
    reproduced = [c for c in score["cells"] if c["status"] == REPRODUCED]
    d = {id(c): c[key] for c in reproduced}
    added = {id(c): (d[id(c)] if both_ways else max(d[id(c)], 0.0)) for c in reproduced}
    up = [c for c in reproduced if d[id(c)] >= threshold]
    down = [c for c in reproduced if d[id(c)] <= -threshold]
    counted = up + down if both_ways else up
    small_up = [c for c in reproduced if FLOAT_SLACK < d[id(c)] < threshold]
    small_down = [c for c in reproduced if -threshold < d[id(c)] < -FLOAT_SLACK]
    every_zero = all(abs(x) <= FLOAT_SLACK for c in reproduced
                     for x in (c.get("deltas") or {"chosen": c["delta"]}).values())
    flagged = {id(c) for c in up + down}
    per_cell, ad3 = [], {}
    for date, table in tables.items():
        header, rows = parse_table(table["text"])
        if "model" not in header or TASK not in header:
            continue
        categories = table["categories"]
        da_cols = next((cols for cols in categories.values() if TASK in cols), None)
        mine = [c for c in reproduced if c["table"] == date]
        replace = {c["row"]: c["published_value"] + added[id(c)] for c in mine}
        after = row_after(rows, replace)
        before_ga = {i: global_average(r, categories) for i, r in enumerate(rows)}
        after_ga = {i: global_average(r, categories) for i, r in enumerate(after)}
        before_rank = competition_rank(before_ga)
        after_rank = competition_rank(after_ga)
        changed = {i: (before_rank[i], after_rank[i]) for i in before_rank
                   if before_rank[i] != after_rank[i]}
        moves = [abs(a - b) for b, a in changed.values() if a is not None and b is not None]
        top10 = any((b is not None and b <= 10) or (a is not None and a <= 10)
                    for b, a in changed.values())
        ad3[date] = {"rows": len(rows), "cells_replaced": len(replace),
                     "rank_changed": [{"model": rows[i]["model"], "before": b, "after": a}
                                      for i, (b, a) in sorted(changed.items())],
                     "models_whose_rank_changed": len(changed),
                     "largest_move": max(moves) if moves else 0,
                     "top10_position_changed": top10}
        for c in mine:
            if id(c) in flagged:
                per_cell.append({
                    "table": date, "model": c["model"], "published": c["published"],
                    "direction": "up" if d[id(c)] > 0 else "down (the strip's own error)",
                    "version": c[vkey], "delta": d[id(c)],
                    "data_analysis_before": calculate_average(rows[c["row"]], da_cols),
                    "data_analysis_after": calculate_average(after[c["row"]], da_cols),
                    "global_average_before": before_ga[c["row"]],
                    "global_average_after": after_ga[c["row"]],
                    "rank_before": before_rank[c["row"]], "rank_after": after_rank[c["row"]]})
    any_rank = any(v["models_whose_rank_changed"] for v in ad3.values())
    return {
        "ad2": {"reading": reading, "delta_used": key,
                "reproduced_cells": len(reproduced),
                "reproduced_models": len({c["model_id"] for c in reproduced}),
                "cells": [{"table": c["table"], "model": c["model"], "published": c["published"],
                           "version": c[vkey], "O": c["per_version"][c[vkey]]["O"],
                           "F": c["per_version"][c[vkey]]["F"], "delta": d[id(c)],
                           "delta_smallest": c["delta"], "delta_largest": c["delta_max"],
                           "deltas_by_reproducing_version": c.get("deltas"),
                           "answers_up_under_the_smallest": c.get("answers_up"),
                           "answers_down_under_the_smallest": c.get("answers_down")}
                          for c in reproduced],
                "moved_up_cells": len(up), "models_moved_up": sorted({c["model_id"] for c in up}),
                "moved_down_cells": len(down),
                "models_moved_down": sorted({c["model_id"] for c in down}),
                "moved_up_by_less_than_a_question": len(small_up),
                "moved_down_by_less_than_a_question": len(small_down),
                "largest_upward_points": max((d[id(c)] for c in up + small_up), default=0.0),
                "largest_downward_points": max((-d[id(c)] for c in down + small_down), default=0.0),
                "every_delta_zero_under_every_reproducing_version": every_zero,
                "moved_cell_detail": per_cell,
                "words": verdict_words(len({c["model_id"] for c in counted}))},
        "ad3": {"reading": reading, "per_table": ad3,
                "words": "RANK CHANGED" if any_rank else "NO RANK CHANGED"},
    }


def secondary(score: dict, ad1_record: dict, fix_version: str) -> dict:
    """Context, never used for a bar."""
    s1_cells = []
    for c in score["cells"]:
        if c["status"] != REPRODUCED:
            continue
        rows = [r for r in score["per_answer"] if r["model_id"] == c["model_id"]]
        o_fix, _ = cell_value(rows, fix_version, "O") if rows and fix_version in rows[0]["v"] else (None, 0)
        if o_fix is not None:
            s1_cells.append({"table": c["table"], "model": c["model"],
                             "fix_minus_O": o_fix - c["O"]})
    s2 = Counter()
    for r in score["per_answer"]:
        rv = ad1_record["reading_version_by_model"].get(r["model_id"])
        if rv is None or rv["version"] not in r["v"]:
            continue
        row = r["v"][rv["version"]]
        if isinstance(row["O"], float) and isinstance(row["A"], float):
            s2["up" if row["A"] > row["O"] else "down" if row["A"] < row["O"] else "equal"] += 1
    return {
        "ad_s1_owners_fix": {"version": fix_version, "reproduced_cells_compared": len(s1_cells),
                             "cells_where_the_fix_moves_two_points_or_more":
                                 sum(abs(x["fix_minus_O"]) >= MOVED_POINTS - FLOAT_SLACK for x in s1_cells),
                             "cells": s1_cells},
        "ad_s2_wider_scope_counts": dict(s2),
        "ad_s3_table_edit": score.get("ad_s3"),
    }


def predictions(score: dict, ad1_record: dict, lower: dict | None, upper: dict | None,
                drafted: dict | None = None) -> dict:
    """The protocol's predictions, checked as written, each on the reading
    that makes it harder to hold: up and down flips under any version in
    force for the answer's model; AD2 and AD3 under the largest F - O
    among the versions that reproduce each cell AND under the rule the
    predictions were drafted under (a move either way counts), holding
    only if both hold."""
    out = {}
    per_table = Counter()
    repro_table = Counter()
    for c in score["cells"]:
        per_table[c["table"]] += 1
        repro_table[c["table"]] += c["status"] == REPRODUCED
    p1 = {d: (repro_table[d] / per_table[d] if per_table[d] else None)
          for d in ("2024_06_24", "2024_07_26")}
    out["G3_at_least_90pct_in_each_of_2024_06_24_and_2024_07_26"] = {
        "fractions": p1, "held": all(v is not None and v >= 0.9 for v in p1.values())}
    t = ad1_record["totals"]
    harder = ad1_record["any_version_in_force"]
    n = t.get("answers", 0)
    fenced = t.get("open_with_a_fence", 0)
    python = ad1_record["by_opening_fence"]["python"].get("answers", 0)
    out["AD1_most_open_with_a_fence"] = {"fenced": fenced, "answers": n, "held": n > 0 and fenced * 2 > n}
    out["AD1_most_fences_tagged_python"] = {"python": python, "fenced": fenced,
                                            "held": fenced > 0 and python * 2 > fenced}
    up_any, down_any = harder.get("up", 0), harder.get("down", 0)
    out["AD1_upward_under_2pct"] = {"up_under_any_version_in_force": up_any,
                                    "up_under_the_reading_version": t.get("up", 0), "answers": n,
                                    "held": n > 0 and up_any < 0.02 * n}
    out["AD1_downward_at_most_5"] = {"down_under_any_version_in_force": down_any,
                                     "down_under_the_reading_version": t.get("down", 0),
                                     "held": down_any <= 5}
    if lower is None or upper is None:
        out["AD2_NULL"] = {"held": None, "why": "AD2 withheld"}
        out["AD3_NO_RANK_CHANGED"] = {"held": None, "why": "AD3 withheld"}
    else:
        drafted = drafted or lower
        out["AD2_NULL"] = {"words_smallest": lower["ad2"]["words"],
                           "words_largest": upper["ad2"]["words"],
                           "words_as_drafted": drafted["ad2"]["words"],
                           "held": upper["ad2"]["words"] == "NULL" and drafted["ad2"]["words"] == "NULL"}
        out["AD3_NO_RANK_CHANGED"] = {
            "words_smallest": lower["ad3"]["words"], "words_largest": upper["ad3"]["words"],
            "words_as_drafted": drafted["ad3"]["words"],
            "held": (upper["ad3"]["words"] == "NO RANK CHANGED"
                     and drafted["ad3"]["words"] == "NO RANK CHANGED")}
    return out


def names(models: list[str]) -> str:
    return " and ".join(f"`{m}`" for m in models)


def _points(value: float) -> str:
    return f"{value:.2f}"


def _rank_detail(a3: dict) -> str:
    changed = {d: v["models_whose_rank_changed"] for d, v in a3["per_table"].items()
               if v["models_whose_rank_changed"]}
    return (", ".join(f"{d}: {k} model(s)" for d, k in changed.items())
            + "; largest move " + str(max(v["largest_move"] for v in a3["per_table"].values()))
            + "; a top-10 position changed: "
            + ("yes" if any(v["top10_position_changed"] for v in a3["per_table"].values()) else "no"))


def licensed_sentences(lower: dict | None, upper: dict | None, scope: dict | None,
                       withheld_why: str | None) -> dict:
    """The only sentences the reading licenses, and the exact text that
    replaces the open question in tools/README.md. Each states its reach:
    how many in-frame cells reproduced, how many table rows were never in
    frame, every move smaller than a question, the strip's own errors,
    what another reproducing version would show, and the ranks."""
    if lower is None or upper is None:
        s2 = ("We could not reproduce LiveBench's published `tablejoin` numbers from its "
              "stored answers, so nothing is said about them." if withheld_why == "G3" else withheld_why)
        return {"ad2": s2, "ad3": None, "directions": None, "readme_replacement": None}
    a2, a3, u2, u3 = lower["ad2"], lower["ad3"], upper["ad2"], upper["ad3"]
    n, m = a2["reproduced_cells"], a2["reproduced_models"]
    reach = (f"{n} of the {scope['in_frame_cells']} in-frame cells reproduced, and only those are "
             f"read; {scope['table_rows'] - scope['in_frame_cells']} of the {scope['table_rows']} "
             "table rows were not in frame.")
    strip = ""
    lowered = a2["moved_down_cells"] + a2["moved_down_by_less_than_a_question"]
    if lowered:
        strip = (f" The strip itself lowered {lowered} reproduced cell(s), {a2['moved_down_cells']} "
                 f"of them by a question's worth or more (largest {_points(a2['largest_downward_points'])} "
                 "points): that is fencecheck's own error, not LiveBench's.")
    if a3["words"] == "NO RANK CHANGED":
        s3 = ("NO RANK CHANGED: adding each reproduced cell's upward F - O to its published value "
              "changes no position.")
        if u3["words"] == "RANK CHANGED":
            s3 += (" Under the scorer version with the largest F - O among those that also "
                   "reproduce each cell, a position would change: " + _rank_detail(u3) + ".")
    else:
        s3 = "RANK CHANGED: " + _rank_detail(a3) + "."
    words = a2["words"]
    if words == "NULL":
        parts = ["Re-scoring LiveBench's stored 2024 `tablejoin` answers with one fence strip moved "
                 f"no published number up by a question's worth (2.0 points), in {n} reproduced "
                 f"cells over {m} models.", reach]
        if a2["moved_up_by_less_than_a_question"]:
            parts.append(f"{a2['moved_up_by_less_than_a_question']} reproduced cell(s) moved up by "
                         f"less than that (largest {_points(a2['largest_upward_points'])} points).")
        else:
            parts.append("No reproduced cell moved up at all.")
        if u2["moved_up_cells"]:
            parts.append(f"Under another scorer version that also reproduces them, "
                         f"{u2['moved_up_cells']} cell(s) over {len(u2['models_moved_up'])} model(s) "
                         "would move up by a question's worth or more.")
        if a3["words"] == "RANK CHANGED":
            parts.append("Ranks still changed: " + _rank_detail(a3) + ".")
        elif u3["words"] == "RANK CHANGED":
            parts.append("No rank changed, but under another scorer version that also reproduces "
                         "the cells one would: " + _rank_detail(u3) + ".")
        else:
            parts.append("No rank changed.")
        s2 = " ".join(parts) + strip + " The fence failure was real in the scorer"
        if a2["every_delta_zero_under_every_reproducing_version"] and u3["words"] == "NO RANK CHANGED":
            s2 += f"; in these {n} cells it changed no published number and no rank."
        else:
            s2 += "."
        readme = s2
    else:
        if words == "MOVES A NUMBER":
            s2 = ("The fence failure moved a published evaluation number on LiveBench's 2024 "
                  f"`tablejoin` task, for {names(a2['models_moved_up'])} ({a2['moved_up_cells']} "
                  "cell(s) up by a question's worth or more under every scorer version that "
                  "reproduces them, listed).")
        else:
            s2 = ("The fence failure moved published evaluation numbers: on LiveBench's 2024 "
                  f"`tablejoin` task, {len(a2['models_moved_up'])} models, cells listed "
                  f"({a2['moved_up_cells']} cell(s) up by a question's worth or more under every "
                  "scorer version that reproduces them).")
        s2 = s2 + " " + reach + strip
        readme = s2 + (" " + s3 if a3["words"] == "RANK CHANGED" else "")
    return {"ad2": s2, "ad3": s3,
            "directions": (f"{a2['moved_up_cells']} cell(s) moved up by a question's worth or more; "
                           f"{a2['moved_down_cells']} moved down by as much, the strip's own error."),
            "readme_replacement": readme}


def _published_first_line(entry: str, role: str | None) -> str:
    if role == "feasibility":
        return (f"{entry}: the scoping page served at fetch is not the page seen at scoping (its "
                "sha256 differs); G2 compares the texts, and this difference publishes first.")
    return (f"{entry}: the live file served at fetch differs from the pinned one; the served file "
            "is the published number, and this difference publishes first.")


def reading_from(fetch: dict, score: dict, tables: dict, manifest: dict) -> dict:
    """Everything the protocol reads, from the banked artifacts alone."""
    first: list[str] = []
    g0 = fetch["g0"]
    roles = g0.get("recorded_roles", {})
    for entry in g0["recorded_differences_published_first"]:
        first.append(_published_first_line(entry, roles.get(entry)))
    if not score["g1"]["passed"]:
        return {"g1": score["g1"], "published_first": first,
                "words": "G1 FAILED",
                "licensed_sentences": {"ad2": (
                    "LiveBench's historical tablejoin scorer did not repeat its scoping behaviour "
                    "on synthetic input; nothing was scored, and that mismatch is the result."),
                    "readme_replacement": None}}
    g2 = score["g2"]
    if not (g2.get("texts_equal") and g2.get("tags_match_disclosure")
            and g2.get("page_matches_scoping_digest")):
        first.append("G2: the scoping page and the parquet differ, or the fence tags differ from "
                     "the disclosure; the difference publishes first (see g2).")
    scoping = manifest["frame"]["scoping_counts"]
    in_frame = len(score["cells"])
    if in_frame != scoping["in_frame_cells"]:
        first.append(f"the run counts {in_frame} in-frame cells; scoping counted "
                     f"{scoping['in_frame_cells']} from names alone.")
    statuses = Counter(c["status"] for c in score["cells"])
    reproduced = statuses.get(REPRODUCED, 0)
    g3 = {"in_frame_cells": in_frame, "by_status": dict(statuses),
          "reproduced_fraction": reproduced / in_frame if in_frame else None,
          "not_reproduced_listed": [{"table": c["table"], "model": c["model"], "status": c["status"]}
                                    for c in score["cells"] if c["status"] != REPRODUCED],
          "postdates_cells_the_draft_lower_bound_kept": {
              d: info.get("postdates_cells_the_draft_lower_bound_kept", 0)
              for d, info in score.get("frame", {}).items()}}
    g3["withholds"] = in_frame == 0 or reproduced * 2 < in_frame
    scope = {"in_frame_cells": in_frame,
             "table_rows": sum(info.get("rows", 0) for info in score.get("frame", {}).values())}
    ad1_record = ad1(score)
    lower = None if g3["withholds"] else ad2_ad3(score, tables, "lower")
    upper = None if g3["withholds"] else ad2_ad3(score, tables, "upper")
    drafted = None if g3["withholds"] else ad2_ad3(score, tables, "as_drafted")
    fix_version = manifest["livebench"]["versions"][-1]["id"]
    withheld = {"words": "WITHHELD (NOT REPRODUCED)"}
    sentences = licensed_sentences(lower, upper, scope, "G3" if g3["withholds"] else None)
    record = {
        "g0": g0, "g1": {"passed": True}, "g2": g2, "g3": g3,
        "published_first": first,
        "ad1": ad1_record,
        "ad2": lower["ad2"] if lower else withheld,
        "ad3": lower["ad3"] if lower else withheld,
        "ad2_largest_delta": upper["ad2"] if upper else withheld,
        "ad3_largest_delta": upper["ad3"] if upper else withheld,
        "ad2_ad3_as_drafted_for_the_predictions_only": (
            {"ad2_words": drafted["ad2"]["words"], "ad3_words": drafted["ad3"]["words"],
             "moved_cells_either_way": drafted["ad2"]["moved_up_cells"] + drafted["ad2"]["moved_down_cells"]}
            if drafted else withheld),
        "secondary": secondary(score, ad1_record, fix_version),
        "predictions": predictions(score, ad1_record, lower, upper, drafted),
        "licensed_sentences": sentences,
    }
    record["words"] = record["ad2"]["words"]
    record["words_largest_delta"] = record["ad2_largest_delta"]["words"]
    record["obligations_the_same_day"] = [
        "tools/README.md: replace 'Whether that bias shows up in published work is a separate "
        "question that this tool does not answer.' with licensed_sentences.readme_replacement, "
        "word for word, at the same place and size"
        + ("; AD2 is WITHHELD, so the sentence stays, with a pointer to the VERDICT row."
           if lower is None else "."),
        "VERDICT.md: an AD row with the words above, both readings, and every prediction "
        "checked (held, failed, or not checkable).",
        "Nothing is sent to LiveBench or anyone else by this script; whether to tell the "
        "maintainers is the founder's call.",
    ]
    return record


# -- stages --------------------------------------------------------------------------------


def banked_path(stage: str) -> pathlib.Path:
    return {"fetch": FETCH_OUT, "score": SCORE_OUT, "read": READ_OUT}[stage]


def _load(path: pathlib.Path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def append_outcome(stage: str, log: pathlib.Path | None = None, **extra) -> None:
    """How an attempt ended without banking anything, appended to the
    same log, so an attempt that banked nothing still says why."""
    log = log or ATTEMPTS_LOG
    entry = {"stage": stage, "outcome_utc": utc_now(), "pid": os.getpid(), **extra}
    log.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def files_not_as_fetched(fetch: dict) -> list[str]:
    """The fetched files that are missing on disk, or whose bytes are not
    the bytes fetch received."""
    bad = []
    for r in fetch.get("remote", []):
        if not r.get("got"):
            continue
        path = data_file(r["path"])
        if not path.is_file() or sha256_file(path) != r["got"]["sha256"]:
            bad.append(r["id"])
    return bad


def preconditions(stage: str, out: pathlib.Path | None = None,
                  fetch_path: pathlib.Path | None = None,
                  score_path: pathlib.Path | None = None,
                  data_dir: pathlib.Path | None = None) -> list[str]:
    """Every reason `stage` may not run, found before it does any work.
    Without --out a stage makes the banked artifact, and only from the
    banked inputs. With --out it recomputes into a new file, only after
    the banked artifact exists, and never over a banked file or into the
    banked data directory."""
    banked_out = banked_path(stage)
    out = pathlib.Path(out) if out is not None else banked_out
    banked = same_path(out, banked_out)
    problems = []
    manifest = load_manifest()
    anchor = anchor_record(required=required_files(manifest))
    if not anchor["ok"]:
        problems.append(f"anchor: {anchor.get('why')}")
    if banked:
        if out.exists():
            problems.append(f"{out.name} is already banked; a stage is made once")
        if fetch_path is not None and not same_path(fetch_path, FETCH_OUT):
            problems.append("the banked artifact is made only from the banked fetch artifact; "
                            "--fetch is for a recompute, with --out")
        if score_path is not None and not same_path(score_path, SCORE_OUT):
            problems.append("the banked reading is made only from the banked score artifact; "
                            "--score is for a recompute, with --out")
    else:
        if any(same_path(out, p) for p in (FETCH_OUT, SCORE_OUT, READ_OUT, ATTEMPTS_LOG)):
            problems.append(f"--out {out} names a banked file")
        if not banked_out.exists():
            problems.append(f"the banked {banked_out.name} does not exist yet; --out recomputes after it")
        if out.exists():
            problems.append(f"{out} exists; a recompute writes a new file")
    if stage == "fetch" and not banked and data_dir is not None:
        reserved = [DEFAULT_DATA_DIR]
        if FETCH_OUT.exists():
            recorded = _load(FETCH_OUT).get("data_dir")
            if recorded:
                reserved.append(data_file(recorded))
        if any(same_path(data_dir, p) for p in reserved):
            problems.append(f"--data-dir {data_dir} is the banked fetch's data directory; "
                            "a recompute never writes there")
    fetch_path = pathlib.Path(fetch_path) if fetch_path is not None else FETCH_OUT
    score_path = pathlib.Path(score_path) if score_path is not None else SCORE_OUT
    if stage in ("score", "read"):
        if not fetch_path.exists():
            problems.append("fetch has not been banked" if same_path(fetch_path, FETCH_OUT)
                            else f"{fetch_path} does not exist")
        else:
            fetch = _load(fetch_path)
            if not fetch.get("complete") or not fetch.get("g0", {}).get("passed"):
                problems.append("G0 did not pass: nothing is scored until a dated erratum amends "
                                "the protocol")
            elif stage == "score":
                bad = files_not_as_fetched(fetch)
                if bad:
                    fix = ("run `published_number_audit.py restore`, which accepts only the banked "
                           "bytes" if same_path(fetch_path, FETCH_OUT)
                           else "run `fetch --out` again into a new --data-dir")
                    problems.append(f"{len(bad)} file(s) of the fetch artifact are missing or are not "
                                    f"the bytes fetch received ({', '.join(bad[:6])}"
                                    f"{', ...' if len(bad) > 6 else ''}): {fix}")
    if stage == "score":
        for name in ("numpy", "pyarrow"):
            if importlib.util.find_spec(name) is None:
                problems.append(f"{name} is not installed")
    if stage == "read":
        if not score_path.exists():
            problems.append("score has not been banked" if same_path(score_path, SCORE_OUT)
                            else f"{score_path} does not exist")
        else:
            score = _load(score_path)
            if not score.get("complete"):
                problems.append("the score artifact is incomplete")
            if fetch_path.exists() and score.get("fetch_sha256") != sha256_file(fetch_path):
                problems.append("the score artifact was not made from this fetch artifact")
    return problems


def _withheld(what: str, problems: list[str]) -> int:
    print(f"WITHHELD: nothing is {what} and nothing is written:")
    print("\n".join(f"  {p}" for p in problems))
    return 2


def _tables_from(fetch: dict, manifest: dict) -> dict:
    """The published tables as served (the live file is the published
    number when it differs from the pinned one), with their categories
    and time bounds: the upper bound is the committer time of the table's
    last commit, and the scoping draft's lower bound (the table's first
    commit) is carried so the cells it would have kept are counted."""
    rows = {r["id"]: r for r in fetch["remote"]}
    tables = {}
    for date, meta in manifest["site"]["tables"].items():
        live, pinned = rows.get(meta["live_id"]), rows.get(meta["pinned_id"])
        source = live if live and live.get("got") else pinned
        text = data_file(source["path"]).read_bytes().decode("utf-8")
        categories = json.loads(data_file(rows[meta["categories_id"]]["path"]).read_text(encoding="utf-8"))
        upper, why = meta["last_commit"]["committer_time_utc"], "the pinned table's last commit"
        if source is live and not live.get("matches"):
            # served bytes no commit we pinned explains: any version up to
            # the fetch may have produced them
            upper, why = live["received"]["fetched_utc"], "the live table differs; its fetch time"
        tables[date] = {"text": text, "categories": categories, "source": source["id"],
                        "upper_utc": upper, "upper_bound_from": why,
                        "first_utc": meta.get("first_commit", {}).get("author_time_utc")}
    return tables


def stage_fetch(data_dir: pathlib.Path, out: pathlib.Path | None = None) -> int:
    out = pathlib.Path(out) if out is not None else FETCH_OUT
    problems = preconditions("fetch", out, data_dir=data_dir)
    if problems:
        return _withheld("fetched", problems)
    lower_priority()
    attempt = append_attempt("fetch", out=as_recorded(out), data_dir=as_recorded(data_dir))
    manifest = load_manifest()
    anchor = anchor_record(required=required_files(manifest))
    result = fetch_all(manifest, data_dir)
    if result["unreceived"]:
        # a transfer that did not complete is not a digest that differs:
        # nothing is banked, the attempt and its outcome stay in the log,
        # and fetch may run again
        why = {r["id"]: r.get("error") or r.get("not_received") for r in result["remote"]
               if r["id"] in result["unreceived"]}
        append_outcome("fetch", attempt_started_utc=attempt["started_utc"],
                       outcome="not received in full; nothing banked", unreceived=why)
        print("INCOMPLETE: not received in full, nothing banked; run fetch again:")
        for eid, text in why.items():
            print(f"  {eid}: {text}")
        return 3
    record = {"what": "Addendum AD fetch: exactly the manifest, verified (G0)",
              "protocol": str(PROTOCOL_PATH.relative_to(REPO)),
              "fetch_started_utc": attempt["started_utc"], "git_head": git_head(),
              "anchor": anchor, "manifest_sha256": sha256_file(MANIFEST_PATH),
              "data_dir": as_recorded(data_dir), "attempts": attempts("fetch"), **result,
              "environment": environment(), "complete": True}
    record["fetch_finished_utc"] = utc_now()
    write_once(out, record)
    print(f"G0 {'PASSED' if result['g0']['passed'] else 'FAILED'}; banked {out}")
    for entry in result["g0"]["recorded_differences_published_first"]:
        print(f"  publishes first: {entry} differs from the pinned file")
    return 0 if result["g0"]["passed"] else 1


def stage_restore() -> int:
    """Put back the banked fetch's data files that are missing or changed
    on disk, accepting only the exact bytes the banked fetch received (a
    file the host now serves differently cannot be restored; only a dated
    erratum moves the run on). Writes no artifact."""
    problems = []
    manifest = load_manifest()
    anchor = anchor_record(required=required_files(manifest))
    if not anchor["ok"]:
        problems.append(f"anchor: {anchor.get('why')}")
    if not FETCH_OUT.exists():
        problems.append("fetch has not been banked; there is nothing to restore")
    if problems:
        return _withheld("restored", problems)
    fetch = _load(FETCH_OUT)
    bad = files_not_as_fetched(fetch)
    if not bad:
        print("nothing to restore: every banked file is on disk as fetch received it")
        return 0
    lower_priority()
    attempt = append_attempt("restore", ids=bad)
    rows = {r["id"]: r for r in fetch["remote"]}
    failed = {}
    for eid in bad:
        r = rows[eid]
        try:
            download(r["url"], data_file(r["path"]), int(r["got"]["bytes"] * 2 + 4_000_000),
                     expect_sha256=r["got"]["sha256"])
        except Exception as error:  # noqa: BLE001
            failed[eid] = f"{type(error).__name__}: {error}"
    append_outcome("restore", attempt_started_utc=attempt["started_utc"],
                   restored=[e for e in bad if e not in failed], failed=failed)
    if failed:
        print("NOT RESTORED (the banked bytes were not received):")
        for eid, text in failed.items():
            print(f"  {eid}: {text}")
        return 3
    print(f"restored {len(bad)} file(s), each byte for byte what the banked fetch received")
    return 0


def stage_score(out: pathlib.Path | None = None, fetch_path: pathlib.Path | None = None) -> int:
    out = pathlib.Path(out) if out is not None else SCORE_OUT
    problems = preconditions("score", out, fetch_path)
    if problems:
        return _withheld("scored", problems)
    fetch_path = pathlib.Path(fetch_path) if fetch_path is not None else FETCH_OUT
    lower_priority()
    attempt = append_attempt("score", out=as_recorded(out), fetch=as_recorded(fetch_path))
    manifest = load_manifest()
    anchor = anchor_record(required=required_files(manifest))
    fetch = _load(fetch_path)
    rows = {r["id"]: r for r in fetch["remote"]}
    # the files on disk are still the files fetch received (checked again
    # after the attempt is logged, in case they changed in between)
    bad = files_not_as_fetched(fetch)
    if bad:
        append_outcome("score", attempt_started_utc=attempt["started_utc"],
                       outcome="files not as fetched; nothing scored", files=bad)
        raise SystemExit(f"{', '.join(bad)} changed on disk since fetch; nothing is scored "
                         "(restore puts back the banked bytes)")
    fencecheck = load_fencecheck()
    versions = load_versions(manifest)
    expected = json.loads(SYNTHETIC_PATH.read_text(encoding="utf-8"))
    g1 = synthetic_check(versions, expected, fencecheck)
    record = {"what": "Addendum AD score: every tablejoin answer under every pinned version, O, F "
                      "and A; every in-frame cell reproduced from O before it counts",
              "protocol": str(PROTOCOL_PATH.relative_to(REPO)),
              "score_started_utc": attempt["started_utc"], "git_head": git_head(),
              "anchor": anchor, "fetch": as_recorded(fetch_path),
              "fetch_sha256": sha256_file(fetch_path),
              "attempts": attempts("score"),
              "versions": [{"id": v.id, "in_force_from_utc": v.spec["in_force_from_utc"],
                            "extraction_source": v.extraction_source} for v in versions],
              "g1": g1}
    if not g1["passed"]:
        record.update({"complete": True, "environment": environment(), "score_finished_utc": utc_now()})
        write_once(out, record)
        print(f"G1 FAILED: nothing scored; the mismatch is the result. banked {out}")
        return 1
    q_rows = load_parquet_rows(data_file(rows["questions"]["path"]),
                               ["question_id", "task", "turns", "ground_truth"])
    questions = {q["question_id"]: q for q in q_rows}
    a_rows = load_parquet_rows(data_file(rows["answers"]["path"]),
                               ["question_id", "model_id", "task", "choices", "tstamp"])
    answers, integrity = dedupe_answers(a_rows, questions)
    page = rows.get("feasibility_page")
    page_bytes = data_file(page["path"]).read_bytes() if page and page.get("got") else None
    g2 = feasibility_check(page_bytes, bool(page and page.get("matches")), answers,
                           manifest["feasibility_disclosed"])
    per_answer = score_answers(questions, answers, versions, fencecheck)
    tables = _tables_from(fetch, manifest)
    model_ids = sorted({a["model_id"] for a in answers})
    built = build_cells(tables, per_answer, manifest["livebench"]["versions"], model_ids,
                        parse_utc(manifest["livebench"]["window_closes_utc"]))
    ad_s3 = table_edit(rows, manifest, model_ids)
    record.update({
        "questions": len(questions), "answers_read": len(a_rows), "answers_scored": len(answers),
        "answer_integrity": integrity, "models": len(model_ids),
        "model_ids_excluded_for_slash": sorted(m for m in model_ids if "/" in m),
        "g2": g2, "table_order": list(manifest["site"]["tables"]),
        "tables": tables,  # as read: the reading is arithmetic on this artifact alone
        "frame": built["frame"], "cells": built["cells"], "per_answer": per_answer,
        "ad_s3": ad_s3, "environment": environment(), "complete": True,
    })
    record["score_finished_utc"] = utc_now()
    write_once(out, record)
    statuses = Counter(c["status"] for c in built["cells"])
    print(f"G1 passed; {len(answers)} answers x {len(versions)} versions scored; cells {dict(statuses)}")
    print(f"banked {out}")
    return 0


def table_edit(rows: dict, manifest: dict, model_ids: list[str]) -> dict | None:
    """AD-S3: did the edit made 5 h 25 min after the owner's fix change any
    in-frame tablejoin cell of table_2025_04_02? Read at run time."""
    meta = manifest["site"].get("table_edit_after_fix")
    if not meta:
        return None
    before, after = rows.get(meta["before_id"]), rows.get(meta["after_id"])
    if not (before and before.get("got") and after and after.get("got")):
        return {"readable": False}
    _, b_rows = parse_table(data_file(before["path"]).read_text(encoding="utf-8"))
    _, a_rows = parse_table(data_file(after["path"]).read_text(encoding="utf-8"))
    in_frame = {n for n, m in match_models([r["model"] for r in a_rows], model_ids).items()
                if m["model_id"]}
    b_map = {r["model"]: (r.get(TASK) or "").strip() for r in b_rows}
    changed = sorted(r["model"] for r in a_rows if r["model"] in in_frame and r["model"] in b_map
                     and b_map[r["model"]] != (r.get(TASK) or "").strip())
    added = sorted(r["model"] for r in a_rows if r["model"] in in_frame and r["model"] not in b_map)
    return {"readable": True, "in_frame_rows_with_a_changed_tablejoin_cell": changed,
            "in_frame_rows_added": added}


def stage_read(out: pathlib.Path | None = None, fetch_path: pathlib.Path | None = None,
               score_path: pathlib.Path | None = None) -> int:
    out = pathlib.Path(out) if out is not None else READ_OUT
    problems = preconditions("read", out, fetch_path, score_path)
    if problems:
        return _withheld("read", problems)
    fetch_path = pathlib.Path(fetch_path) if fetch_path is not None else FETCH_OUT
    score_path = pathlib.Path(score_path) if score_path is not None else SCORE_OUT
    attempt = append_attempt("read", out=as_recorded(out), fetch=as_recorded(fetch_path),
                             score=as_recorded(score_path))
    manifest = load_manifest()
    anchor = anchor_record(required=required_files(manifest))
    fetch = _load(fetch_path)
    score = _load(score_path)
    reading = reading_from(fetch, score, score.get("tables", {}), manifest)
    record = {"what": "Addendum AD reading: the frozen readings and words, by arithmetic on the "
                      "banked score artifact",
              "protocol": str(PROTOCOL_PATH.relative_to(REPO)),
              "read_started_utc": attempt["started_utc"], "git_head": git_head(),
              "anchor": anchor, "fetch": as_recorded(fetch_path), "score": as_recorded(score_path),
              "fetch_sha256": sha256_file(fetch_path),
              "score_sha256": sha256_file(score_path), "attempts": attempts("read"),
              **reading, "complete": True,
              "how_to_recompute": ("fetch --out F --data-dir D; score --out S --fetch F; "
                                   "read --out R --fetch F --score S (scripts/published_number_audit.py)")}
    record["read_finished_utc"] = utc_now()
    write_once(out, record)
    for line in record.get("published_first", []):
        print(line)
    for key in ("ad2", "ad3", "directions"):
        sentence = record["licensed_sentences"].get(key)
        if sentence:
            print(sentence)
    print(f"banked {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="report every stage's preconditions; touches nothing")
    f = sub.add_parser("fetch")
    f.add_argument("--data-dir", default=None,
                   help="default: data/ad for the banked fetch, a new data/ad_recompute_<UTC> with --out")
    f.add_argument("--out", default=None)
    sub.add_parser("restore", help="re-download banked data files that are missing; only their banked bytes")
    s = sub.add_parser("score")
    s.add_argument("--out", default=None)
    s.add_argument("--fetch", default=None, help="with --out only: the fetch artifact to score from")
    r = sub.add_parser("read")
    r.add_argument("--out", default=None)
    r.add_argument("--fetch", default=None, help="with --out only")
    r.add_argument("--score", default=None, help="with --out only")
    args = ap.parse_args(argv)
    if args.cmd == "check":
        worst = 0
        for stage in ("fetch", "score", "read"):
            problems = preconditions(stage)
            print(f"{stage}: " + ("may run" if not problems else "; ".join(problems)))
            worst = max(worst, 2 if problems else 0)
        return worst
    path = lambda value: pathlib.Path(value).resolve() if value else None  # noqa: E731
    if args.cmd == "fetch":
        out = path(args.out)
        if args.data_dir:
            data_dir = path(args.data_dir)
        elif out is None:
            data_dir = DEFAULT_DATA_DIR
        else:
            # a recomputation never writes where the banked fetch's files are
            data_dir = DEFAULT_DATA_DIR.with_name(
                f"ad_recompute_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
        return stage_fetch(data_dir, out)
    if args.cmd == "restore":
        return stage_restore()
    if args.cmd == "score":
        return stage_score(path(args.out), path(args.fetch))
    return stage_read(path(args.out), path(args.fetch), path(args.score))


if __name__ == "__main__":
    raise SystemExit(main())
