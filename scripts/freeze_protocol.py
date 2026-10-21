#!/usr/bin/env python3
"""Freeze a preregistration the way docs/research/ANCHORS.md says it must be frozen.

The order is the whole point, and doing it by hand got it nearly wrong
twice (placeholders in two shapes; a draft header that still quoted the
placeholder). So it is a tool. For Addendum AA, with every pinned file
(the list is `PINNED` in scripts/decoder_isolation_read.py, and a test
keeps this command equal to it):

    python3 scripts/freeze_protocol.py prepare docs/research/ADDENDUM_AA_PROTOCOL.md \
        --code scripts/decoder_isolation_read.py \
        --code scripts/cord_decoder_isolation.py \
        --code scripts/aa_pinned/fencecheck_x.py \
        --code scripts/cord_fence_tax.py \
        --code scripts/cord_constrained_families.py \
        --code src/arcttt/__init__.py \
        --code src/arcttt/scoring.py \
        --code src/arcttt/text_task.py \
        --code src/arcttt/tasks.py \
        --code scripts/aa_pinned/aa_input_manifest.json
    # add the protocol's ANCHORS.md row and VERDICT row, quoting the stamp
    git commit ...                                   # commit the prepared bytes
    python3 scripts/freeze_protocol.py stamp docs/research/ADDENDUM_AA_PROTOCOL.md
    git commit ...                                   # commit the proof
    # later, once a block has been mined:
    ots upgrade docs/research/ADDENDUM_AA_PROTOCOL.md.<stamp>.ots
    git commit ...                                   # commit the upgraded proof

`prepare` refuses unless every `--code` file is tracked and unchanged
since the last commit, so each quoted digest is of bytes some commit
holds, and unless the file carries the placeholders it fills:
STAMP_PLACEHOLDER (the stamp, everywhere it appears) and, for each
`--code` file, a `<NAME>_SHA_PLACEHOLDER` token on the same line as that
file's path in backticks. It removes a draft header delimited by the lines
`<!-- DRAFT-HEADER-START -->` and `<!-- DRAFT-HEADER-END -->`, fills the
placeholders, and writes the anchored snapshot
`docs/research/snapshots_<STEM>_<stamp>.md`, byte-identical to the
protocol. `stamp` refuses unless the protocol and snapshot are committed
and unchanged since (so the anchored bytes are the committed bytes), and
unless every file the protocol quotes a digest for is committed, unchanged
and still hashes to that digest; then it runs `ots stamp` and names the
proof `<file>.<stamp>.ots`.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

REPO = pathlib.Path(__file__).resolve().parent.parent
DRAFT_START = "<!-- DRAFT-HEADER-START -->"
DRAFT_END = "<!-- DRAFT-HEADER-END -->"
STAMP_TOKEN = "STAMP_PLACEHOLDER"


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_draft_header(text: str) -> str:
    if DRAFT_START not in text:
        return text
    if DRAFT_END not in text:
        raise SystemExit("draft header has a start marker and no end marker")
    before, rest = text.split(DRAFT_START, 1)
    _, after = rest.split(DRAFT_END, 1)
    return before + after.lstrip("\n")


def fill(text: str, stamp: str, code: dict[str, str]) -> str:
    """Substitute the stamp and each code digest; refuse on anything left over."""
    if STAMP_TOKEN not in text:
        raise SystemExit(f"no {STAMP_TOKEN} in the protocol: nothing to freeze, or already frozen")
    text = text.replace(STAMP_TOKEN, stamp)
    for rel, digest in code.items():
        pattern = re.compile(r"(`" + re.escape(rel) + r"`[^\n]*?`)([A-Z_]+_SHA_PLACEHOLDER)(`)")
        if not pattern.search(text):
            raise SystemExit(f"no digest placeholder on the line naming `{rel}`")
        text = pattern.sub(lambda m: m.group(1) + digest + m.group(3), text, count=1)
    leftover = re.findall(r"[A-Z_]*PLACEHOLDER", text)
    if leftover:
        raise SystemExit(f"placeholders left after filling: {sorted(set(leftover))}")
    return text


def snapshot_path(protocol: pathlib.Path, stamp: str) -> pathlib.Path:
    return protocol.with_name(f"snapshots_{protocol.stem}_{stamp}.md")


def _posix(rel: str) -> str:
    """A `--code` argument as the protocol quotes it: relative, with / separators."""
    path = pathlib.PurePath(rel)
    if path.is_absolute():
        path = pathlib.Path(rel).resolve().relative_to(REPO.resolve())
    return path.as_posix()


def prepare(protocol: pathlib.Path, code_files: list[str], stamp: str | None = None) -> str:
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{4}Z", stamp):
        raise SystemExit(f"bad stamp {stamp!r}")
    code_files = [_posix(rel) for rel in code_files]
    unclean = [rel for rel in code_files if not _committed_unchanged(REPO / rel)]
    if unclean:
        raise SystemExit("not committed, or changed since the last commit: "
                         + ", ".join(unclean) + ". A digest of bytes no commit holds "
                         "cannot be checked by anyone else; commit first.")
    code = {rel: _sha(REPO / rel) for rel in code_files}
    text = fill(strip_draft_header(protocol.read_text(encoding="utf-8")), stamp, code)
    protocol.write_text(text, encoding="utf-8")
    snap = snapshot_path(protocol, stamp)
    shutil.copyfile(protocol, snap)
    assert snap.read_bytes() == protocol.read_bytes()
    print(f"prepared {protocol.name} with stamp {stamp}; snapshot {snap.name}")
    for rel, digest in code.items():
        print(f"  {rel} sha256 {digest}")
    print("now commit both files, then run: freeze_protocol.py stamp", protocol)
    return stamp


def _committed_unchanged(path: pathlib.Path) -> bool:
    if not path.is_file():
        return False
    rel = path.relative_to(REPO).as_posix()
    status = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=REPO,
                            capture_output=True, text=True, check=True).stdout.strip()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO,
                             capture_output=True, text=True).returncode == 0
    return tracked and not status


def stamp(protocol: pathlib.Path) -> pathlib.Path:
    text = protocol.read_text(encoding="utf-8")
    match = re.search(re.escape(protocol.name) + r"\.(\d{4}-\d{2}-\d{2}T\d{4}Z)\.ots", text)
    if not match:
        raise SystemExit("the freeze line quotes no stamp; run prepare first")
    stamp_str = match.group(1)
    snap = snapshot_path(protocol, stamp_str)
    if not snap.exists() or snap.read_bytes() != protocol.read_bytes():
        raise SystemExit("the snapshot is missing or differs from the protocol")
    for path in (protocol, snap):
        if not _committed_unchanged(path):
            raise SystemExit(f"{path.name} is not committed, or changed since: commit first")
    for rel, digest in re.findall(r"`([A-Za-z0-9_./-]+)` sha256 `([0-9a-f]{64})`", text):
        path = REPO / rel
        if not _committed_unchanged(path):
            raise SystemExit(f"{rel} is quoted with a digest and is not committed, or changed "
                             "since: commit first")
        if _sha(path) != digest:
            raise SystemExit(f"{rel} no longer hashes to the digest the protocol quotes")
    proof = protocol.with_name(f"{protocol.name}.{stamp_str}.ots")
    if proof.exists():
        raise SystemExit(f"{proof.name} exists already")
    subprocess.run(["ots", "stamp", str(protocol)], check=True)
    protocol.with_name(protocol.name + ".ots").rename(proof)
    print(f"stamped: {proof.name} (sha256 {_sha(protocol)}); commit it, then ots upgrade later")
    return proof


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("protocol")
    p.add_argument("--code", action="append", default=[])
    p.add_argument("--stamp", default=None)
    s = sub.add_parser("stamp")
    s.add_argument("protocol")
    args = ap.parse_args(argv)
    path = (REPO / args.protocol).resolve() if not pathlib.Path(args.protocol).is_absolute() \
        else pathlib.Path(args.protocol)
    if args.cmd == "prepare":
        prepare(path, args.code, args.stamp)
    else:
        stamp(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
