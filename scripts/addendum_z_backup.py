#!/usr/bin/env python3
"""Addendum Z's run state, backed up encrypted so it can sit in a repository branch.

The run (scripts/novel_schema_rerun.py) checkpoints to work/z/: per arm an
adapter (.pt), a meta file (restart count, start clock, adapter digest) and
a journal (.jsonl, one row per decoded document, holding the model's outputs
and their scores). work/z/ is not committed. The container that holds it is
reclaimed when a session ends, and the run outlasts a session. A lost
journal means documents decoded again, which is a redraw the run log flags.

The only durable store is a repository branch, treated as if it could become
public (the one used on 2026-10-21 is private), and the protocol
forbids anyone reading a score before the reading. So:

- `backup` writes the journals, metas and operator logs as one encrypted tar
  (`addendum_z_state.tar.enc`), and each adapter as its own encrypted file.
  An adapter is written once and reused by later backups while its bytes
  are unchanged. So a ~35 MB file is not recommitted at every backup, and no
  single file nears GitHub's per-file limit as adapters accumulate. Next to
  them goes a plaintext manifest. It holds per-file sha256 and sizes, the
  time, the run log's per-arm fields (prefix hash, row counts, restart
  count, start clock, adapter digest), every ciphertext's sha256, and an
  HMAC. No score and no text.
- `restore` checks the HMAC, every ciphertext hash, every decrypted file's
  hash, and the per-arm fields. It refuses to overwrite a work dir that is
  newer than the backup or diverges from it, unless --force. Any existing
  file it replaces is moved aside, never deleted. It refuses always while a
  runner is alive on that work dir.
- `keygen` writes a random 256-bit key. The key never goes in a repository.
  The owner keeps it privately, off this container. Without it the backup
  cannot be read and the run starts again from scratch.

Cipher: `openssl enc -aes-256-cbc -pbkdf2 -iter 100000 -md sha256 -salt`,
the passphrase given on stdin, never on the command line. `openssl enc`
refuses AEAD modes such as GCM ("AEAD ciphers not supported"), so integrity
is encrypt-then-MAC: the manifest commits to each ciphertext's sha256 and is
HMAC-SHA256'd under a key derived from the same secret. No compression, so
a ciphertext's length shows only the sizes the manifest already lists.

    python3 scripts/addendum_z_backup.py keygen  --key-file ~/.z_backup.key
    python3 scripts/addendum_z_backup.py backup  --work work/z --out <dir> --key-file ~/.z_backup.key
    python3 scripts/addendum_z_backup.py restore --in <dir> --key-file ~/.z_backup.key --work work/z

If this script is ever gone, the key file is the openssl passphrase. This
skips every check, so use it only as a last resort:

    openssl enc -d -aes-256-cbc -pbkdf2 -iter 100000 -md sha256 \\
        -pass file:KEY -in addendum_z_state.tar.enc | tar -x -C work/z
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import pathlib
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import UTC, datetime

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

# stdlib only at import time; novel_schema_rerun imports torch inside functions
import addendum_z_run_log as runlog

z = runlog.z

FORMAT = "addendum-z-backup/1"
STATE_NAME = "addendum_z_state.tar.enc"
MANIFEST_NAME = "addendum_z_state.manifest.json"
RUN_LOG = REPO / "experiments" / "addendum_z_run_log.json"
ITER = 100000
OPENSSL_ENC = ["enc", "-aes-256-cbc", "-pbkdf2", "-iter", str(ITER), "-md", "sha256"]
MAC_LABEL = b"addendum-z-backup/1 manifest hmac"
CIPHER = {
    "tool": "openssl enc (it has no AEAD mode, so encrypt-then-MAC)",
    "cipher": "aes-256-cbc",
    "kdf": f"pbkdf2-hmac-sha256, {ITER} iterations, random 8-byte salt in the ciphertext header",
    "passphrase": "the key file's 64 hex characters",
    "mac": "hmac-sha256 over this manifest minus hmac_sha256 as canonical JSON, "
           "under HMAC-SHA256(key, 'addendum-z-backup/1 manifest hmac')",
    "compression": "none",
}
GITHUB_FILE_WARN = 50 * 1024 * 1024  # GitHub warns above 50 MB per file
GITHUB_FILE_LIMIT = 100 * 1024 * 1024  # and rejects a push above 100 MB
STEMS = {z.ckpt_stem(seed, arm): (seed, arm) for seed, arm in z.ARM_ORDER}
LOG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.(log|md|txt)$")
BLOB_RE = re.compile(r"^novel_ckpt_z_[A-Za-z0-9._-]+_adapter\.pt\.[0-9a-f]{16}\.enc$")


class BackupError(Exception):
    """A refusal. Printed, exit status 2."""


# -- small helpers -----------------------------------------------------------


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _n(value) -> int:
    return -1 if value is None else int(value)


def _git_root(path: pathlib.Path) -> pathlib.Path | None:
    for parent in (path, *path.parents):
        if (parent / ".git").exists():
            return parent
    return None


def _put(src: pathlib.Path, dst: pathlib.Path) -> pathlib.Path:
    """Copy and fsync to a hidden partial name next to dst; the caller renames."""
    partial = dst.with_name(f".{dst.name}.partial")
    shutil.copyfile(src, partial)
    with open(partial, "rb") as handle:
        os.fsync(handle.fileno())
    return partial


def wanted(name: str) -> str | None:
    """'adapter', 'state', or None for anything not backed up (runner.pid, *.tmp, ...)."""
    for stem in STEMS:
        if name == f"{stem}_adapter.pt":
            return "adapter"
        if name in (f"{stem}_meta.json", f"{stem}_docs.jsonl"):
            return "state"
    return "state" if LOG_RE.match(name) else None


# -- the key -----------------------------------------------------------------


def _check_key_location(path: pathlib.Path) -> None:
    root = _git_root(path.resolve().parent)
    if root is not None:
        raise BackupError(f"key file {path} is inside the git work tree {root}; "
                          "keep the key outside every repository")


def keygen(path: pathlib.Path) -> None:
    path = path.expanduser()
    _check_key_location(path)
    if path.exists():
        raise BackupError(f"{path} exists; refusing to overwrite a key "
                          "(every backup made with it would become unreadable)")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(secrets.token_bytes(32).hex() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_key(path: pathlib.Path) -> bytes:
    path = path.expanduser()
    _check_key_location(path)
    try:
        text = path.read_text()
    except FileNotFoundError:
        raise BackupError(f"no key file at {path}") from None
    # strict, so `-pass file:KEY` in the manual recovery reads the same passphrase
    if not re.fullmatch(r"[0-9a-f]{64}\n?", text):
        raise BackupError(f"{path} is not a key from `keygen` (64 lowercase hex characters)")
    if path.stat().st_mode & 0o077:
        print(f"warning: {path} is readable by other users; chmod 600 it", file=sys.stderr)
    return bytes.fromhex(text.strip())


def _mac(manifest: dict, key: bytes) -> str:
    body = {k: v for k, v in manifest.items() if k != "hmac_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    mac_key = hmac.new(key, MAC_LABEL, hashlib.sha256).digest()
    return hmac.new(mac_key, blob, hashlib.sha256).hexdigest()


def _openssl(decrypt: bool, key: bytes, src: pathlib.Path, dst: pathlib.Path) -> None:
    exe = shutil.which("openssl")
    if exe is None:
        raise BackupError("the openssl CLI is not installed; it does the encryption")
    cmd = [exe, *OPENSSL_ENC, *(["-d"] if decrypt else ["-e", "-salt"]),
           "-pass", "stdin", "-in", str(src), "-out", str(dst)]
    done = subprocess.run(cmd, input=(key.hex() + "\n").encode(), capture_output=True, check=False)
    if done.returncode != 0:
        what = "decrypt" if decrypt else "encrypt"
        raise BackupError(f"openssl could not {what} {src.name}: "
                          f"{done.stderr.decode(errors='replace').strip()[:200]}")


# -- the run state -------------------------------------------------------------


def _journal_rows(path: pathlib.Path) -> list[dict]:
    return sorted(runlog._read_journal_readonly(path), key=lambda r: r["index"])


def arm_states(work: pathlib.Path) -> tuple[list[dict], dict[str, list[dict]]]:
    """Per arm with any file in `work`: the run log's fields (`arm_state`) minus
    `banked`, which would look in experiments/. Hashes and counts only. Also
    returns each arm's journal rows, for the continuation checks."""
    states, rows_by_arm = [], {}
    for stem, (seed, arm) in STEMS.items():
        if not any((work / f"{stem}{s}").exists() for s in ("_adapter.pt", "_meta.json", "_docs.jsonl")):
            continue
        meta_path = work / f"{stem}_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        rows = _journal_rows(work / f"{stem}_docs.jsonl")
        prefix = []
        for want, row in enumerate(rows):
            if row["index"] != want:
                break
            prefix.append(row)
        state = {
            "arm": z.artifact_name(seed, arm),
            "started_utc": meta.get("started_utc"),
            "restarts": meta.get("restarts"),
            "adapter_sha256": meta.get("adapter_sha256"),
            "documents_decoded": len(rows),
            "prefix_documents": len(prefix),
            "prefix_sha256": runlog.row_digest(prefix) if prefix else None,
        }
        # the manifest is public: a meta field of an unexpected type is refused, not copied
        for field, types in (("started_utc", str), ("restarts", int), ("adapter_sha256", str)):
            if state[field] is not None and not isinstance(state[field], types):
                raise BackupError(f"{meta_path.name}: {field} has an unexpected type")
        if state["adapter_sha256"] is not None and not re.fullmatch(r"[0-9a-f]{64}", state["adapter_sha256"]):
            raise BackupError(f"{meta_path.name}: adapter_sha256 is not a sha256")
        states.append(state)
        rows_by_arm[state["arm"]] = rows
    return states, rows_by_arm


def not_a_continuation(old: list[dict], new: list[dict], new_rows: dict[str, list[dict]],
                       old_label: str, new_label: str) -> list[str]:
    """Why `new` does not extend `old`, per arm: a missing arm, a different run
    start, fewer restarts or documents, a changed document, a changed adapter."""
    now = {s["arm"]: s for s in new}
    problems = []
    for before in old:
        name, after = before["arm"], now.get(before["arm"])
        if after is None:
            problems.append(f"{name}: {old_label}, absent {new_label}")
            continue
        if before["started_utc"] and after["started_utc"] != before["started_utc"]:
            problems.append(f"{name}: a different run start ({before['started_utc']} {old_label}, "
                            f"{after['started_utc']} {new_label})")
        if _n(before["restarts"]) > _n(after["restarts"]):
            problems.append(f"{name}: {before['restarts']} restarts {old_label}, "
                            f"{after['restarts']} {new_label}")
        if before["documents_decoded"] > after["documents_decoded"]:
            problems.append(f"{name}: {before['documents_decoded']} documents {old_label}, "
                            f"{after['documents_decoded']} {new_label}")
        n = before["prefix_documents"]
        if n and (after["prefix_documents"] < n
                  or runlog.row_digest(new_rows[name][:n]) != before["prefix_sha256"]):
            problems.append(f"{name}: the first {n} documents {old_label} are not the first {n} "
                            f"{new_label} (a redraw, or another lineage)")
        if before["adapter_sha256"] and after["adapter_sha256"] != before["adapter_sha256"]:
            problems.append(f"{name}: the adapter digest {old_label} differs {new_label}")
    return problems


def _stage(work: pathlib.Path, stage: pathlib.Path) -> tuple[list[str], list[str]]:
    """Copy the run state out of `work` once, so hashes, manifest and tar all
    describe the same bytes while the runner keeps appending."""
    kept, skipped = [], []
    for entry in sorted(os.scandir(work), key=lambda e: e.name):
        kind = wanted(entry.name) if entry.is_file(follow_symlinks=False) else None
        if kind is None:
            skipped.append(entry.name)
            continue
        if entry.name.endswith(".jsonl"):
            # the runner appends and fsyncs one row at a time: keep whole lines only
            data = pathlib.Path(entry.path).read_bytes()
            (stage / entry.name).write_bytes(data[: data.rfind(b"\n") + 1])
        else:
            shutil.copyfile(entry.path, stage / entry.name)
        kept.append(entry.name)
    return kept, skipped


def _tar(stage: pathlib.Path, names: list[str], dest: pathlib.Path) -> None:
    with tarfile.open(dest, "w", format=tarfile.PAX_FORMAT) as tar:
        for name in names:
            info = tar.gettarinfo(str(stage / name), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o600
            with open(stage / name, "rb") as handle:
                tar.addfile(info, handle)


def _untar(archive: pathlib.Path, stage: pathlib.Path, names: set[str]) -> None:
    seen: set[str] = set()
    try:
        with tarfile.open(archive, "r:") as tar:
            for member in tar:
                if not member.isreg() or member.name not in names or member.name in seen:
                    raise BackupError(f"unexpected member in the tar: {member.name!r}")
                seen.add(member.name)
                with tar.extractfile(member) as src, open(stage / member.name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    except tarfile.TarError as error:
        raise BackupError(f"the decrypted tar is unreadable: {error}") from None
    if seen != names:
        raise BackupError(f"missing from the tar: {sorted(names - seen)}")


def _private_tmp() -> tempfile.TemporaryDirectory:
    tmp = tempfile.TemporaryDirectory(prefix="addendum-z-backup-")  # mode 0700
    if _git_root(pathlib.Path(tmp.name)) is not None:
        tmp.cleanup()
        raise BackupError(f"the temp dir {tmp.name} is inside a git work tree; "
                          "plaintext is staged there, so set TMPDIR elsewhere")
    return tmp


def _ignored_by_git(out: pathlib.Path, names: list[str]) -> list[str]:
    if _git_root(out) is None:
        return []
    return [name for name in names
            if subprocess.run(["git", "-C", str(out), "check-ignore", "-q", "--", name],
                              capture_output=True, check=False).returncode == 0]


# -- backup ------------------------------------------------------------------


def backup(work: pathlib.Path, out: pathlib.Path, key_path: pathlib.Path, force: bool = False) -> dict:
    key = read_key(key_path)
    work, out = work.resolve(), out.resolve()
    if not work.is_dir():
        raise BackupError(f"no work dir at {work}")
    if out == work or work in out.parents:
        raise BackupError("--out must not be inside --work")
    if out in key_path.expanduser().resolve().parents:
        raise BackupError("the key file must not sit inside --out")
    out.mkdir(parents=True, exist_ok=True)
    prev = None
    if (out / MANIFEST_NAME).exists():
        try:
            prev = load_manifest(out, key)
        except BackupError as error:
            if not force:
                raise BackupError(f"{error}. Refusing to replace that backup (--force to do it anyway)") from None

    with _private_tmp() as tmp_name:
        tmp = pathlib.Path(tmp_name)
        stage = tmp / "stage"
        stage.mkdir()
        kept, skipped = _stage(work, stage)
        if not kept:
            raise BackupError(f"nothing to back up in {work}")
        states, rows = arm_states(stage)
        if prev is not None and not force:
            problems = not_a_continuation(prev["arms"], states, rows, "in the backup in --out", "in the work dir")
            if problems:
                raise BackupError("the work dir does not extend the backup already in --out, so this "
                                  "backup would replace a newer or different state:\n  " + "\n  ".join(problems)
                                  + "\nRestore first, or pass --force once the loss is booked.")

        old_blobs = {b["file"]: b for b in prev["blobs"]} if prev else {}
        old_files = {(f["name"], f["sha256"], f["in"]) for f in prev["files"]} if prev else set()
        files, blobs, tar_names, fresh = [], [], [], []
        for name in kept:
            digest, size = _sha_file(stage / name), (stage / name).stat().st_size
            if wanted(name) == "adapter":
                blob = f"{name}.{digest[:16]}.enc"
                old = old_blobs.get(blob)
                if (old and (name, digest, blob) in old_files and (out / blob).exists()
                        and _sha_file(out / blob) == old["ciphertext_sha256"]):
                    blobs.append(old)  # same adapter, already encrypted and committed: reuse
                else:
                    _openssl(False, key, stage / name, tmp / blob)
                    blobs.append({"file": blob, "ciphertext_sha256": _sha_file(tmp / blob),
                                  "ciphertext_bytes": (tmp / blob).stat().st_size})
                    fresh.append(blob)
            else:
                tar_names.append(name)
                blob = STATE_NAME
            files.append({"name": name, "sha256": digest, "bytes": size, "in": blob})
        _tar(stage, tar_names, tmp / "state.tar")
        _openssl(False, key, tmp / "state.tar", tmp / STATE_NAME)
        fresh.append(STATE_NAME)
        manifest = {
            "format": FORMAT,
            "what": ("Encrypted backup of Addendum Z's work/z run state. This manifest is public "
                     "and holds hashes, sizes, row counts and restart counts only."),
            "utc": _utc(),
            "cipher": CIPHER,
            "state": {"file": STATE_NAME, "ciphertext_sha256": _sha_file(tmp / STATE_NAME),
                      "ciphertext_bytes": (tmp / STATE_NAME).stat().st_size},
            "blobs": blobs,
            "files": files,
            "arms": states,
        }
        manifest["hmac_sha256"] = _mac(manifest, key)
        (tmp / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        fresh.append(MANIFEST_NAME)

        for name in fresh:
            size = (tmp / name).stat().st_size
            if size >= GITHUB_FILE_LIMIT:
                raise BackupError(f"{name} is {size} bytes; GitHub rejects files of 100 MB or more")
            if size > GITHUB_FILE_WARN:
                print(f"warning: {name} is {size} bytes; GitHub warns above 50 MB", file=sys.stderr)
        # every file fully written beside its final name first, then renamed;
        # the manifest goes last, so a reader never sees it name absent bytes
        partials = [(_put(tmp / name, out / name), out / name) for name in fresh]
        for partial, final in partials:
            os.replace(partial, final)
    keep = {b["file"] for b in blobs}
    for entry in os.scandir(out):
        if BLOB_RE.match(entry.name) and entry.name not in keep:
            os.unlink(entry.path)  # an adapter no longer in the run (still in git history)
    for name in _ignored_by_git(out, [STATE_NAME, MANIFEST_NAME, *sorted(keep)]):
        print(f"warning: git ignores {out / name}; a commit would leave it out", file=sys.stderr)
    if skipped:
        print(f"not backed up: {', '.join(skipped)}", file=sys.stderr)
    return manifest


# -- restore -------------------------------------------------------------------


def load_manifest(directory: pathlib.Path, key: bytes) -> dict:
    path = directory / MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError:
        raise BackupError(f"no {MANIFEST_NAME} in {directory}") from None
    except json.JSONDecodeError:
        raise BackupError(f"{path} is not JSON") from None
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise BackupError(f"{path} is not a {FORMAT} manifest")
    if not hmac.compare_digest(str(manifest.get("hmac_sha256", "")), _mac(manifest, key)):
        raise BackupError(f"{path} does not authenticate under this key: "
                          "it was edited, or made with another key")
    return manifest


def runner_alive(work: pathlib.Path, proc: pathlib.Path = pathlib.Path("/proc")) -> bool:
    """True if a `novel_schema_rerun.py --run` process is using `work`.
    Reads /proc only and never signals anything."""
    work = pathlib.Path(os.path.normpath(work.resolve()))
    for entry in proc.glob("[0-9]*"):
        try:
            args = [a.decode(errors="replace") for a in (entry / "cmdline").read_bytes().split(b"\0") if a]
            cwd = pathlib.Path(os.readlink(entry / "cwd"))
        except OSError:
            continue
        script = next((a for a in args if a.endswith("novel_schema_rerun.py")), None)
        if script is None or "--run" not in args:
            continue
        theirs = None
        for i, arg in enumerate(args):
            if arg == "--work" and i + 1 < len(args):
                theirs = args[i + 1]
            elif arg.startswith("--work="):
                theirs = arg.split("=", 1)[1]
        their_work = cwd / theirs if theirs else (cwd / script).parent.parent / "work" / "z"
        if pathlib.Path(os.path.normpath(their_work)) == work:
            return True
    return False


def run_log_gaps(states: list[dict], rows: dict[str, list[dict]], log_path: pathlib.Path) -> list[str]:
    """Documents the public run log has banked that this backup cannot give back.
    Each will be decoded again, and `addendum_z_run_log.py --verify` will flag it."""
    try:
        log = json.loads(log_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    most: dict[str, tuple[str, dict]] = {}  # per arm, the snapshot with the longest prefix
    restarts: dict[str, int] = {}  # per arm, the most restarts any snapshot saw
    for snap in log.get("snapshots", []):
        for arm in snap.get("arms", []):
            name = arm["arm"]
            restarts[name] = max(restarts.get(name, -1), _n(arm.get("restarts")))
            if (arm.get("prefix_documents") or 0) > (most[name][1]["prefix_documents"] if name in most else 0):
                most[name] = (snap["utc"], arm)
    have = {s["arm"]: s for s in states}
    notes = []
    for name, (utc, logged) in most.items():
        n, got = logged["prefix_documents"], have.get(name)
        m = got["prefix_documents"] if got else 0
        if m < n:
            notes.append(f"{name}: the run log banked {n} documents ({utc}), this backup holds {m}; "
                         f"documents {m}..{n - 1} will be decoded again and --verify will flag them. Book it.")
        elif runlog.row_digest(rows[name][:n]) != logged["prefix_sha256"]:
            notes.append(f"{name}: the first {n} documents here do not match the run log's prefix ({utc})")
    for name, seen in restarts.items():
        mine = _n(have[name]["restarts"]) if name in have else -1
        if seen > mine:
            notes.append(f"{name}: the run log saw {seen} restarts, this backup's meta has "
                         f"{mine if mine >= 0 else 'none'}; the runner counts on from the backup, "
                         "so its restart count will be low. Book the difference.")
    return notes


def restore(src: pathlib.Path, key_path: pathlib.Path, work: pathlib.Path,
            force: bool = False, run_log: pathlib.Path = RUN_LOG) -> dict:
    key = read_key(key_path)
    src, work = src.resolve(), work.resolve()
    manifest = load_manifest(src, key)
    containers = {STATE_NAME: manifest["state"], **{b["file"]: b for b in manifest["blobs"]}}
    for name, entry in containers.items():
        if name != STATE_NAME and not BLOB_RE.match(name):
            raise BackupError(f"the manifest names an unexpected ciphertext {name!r}")
        if not (src / name).exists():
            raise BackupError(f"{name} is missing from {src}")
        if _sha_file(src / name) != entry["ciphertext_sha256"]:
            raise BackupError(f"{name}: ciphertext sha256 does not match the manifest; not decrypting")
    if runner_alive(work):
        raise BackupError(f"a runner is alive on {work}; restoring under it would interleave its "
                          "journal. Not restoring, --force or not.")
    expected = {f["name"]: f for f in manifest["files"]}
    if len(expected) != len(manifest["files"]) or any(wanted(name) is None for name in expected):
        raise BackupError("the manifest lists a duplicate or unexpected file name")

    with _private_tmp() as tmp_name:
        tmp, stage = pathlib.Path(tmp_name), pathlib.Path(tmp_name) / "stage"
        stage.mkdir()
        _openssl(True, key, src / STATE_NAME, tmp / "state.tar")
        _untar(tmp / "state.tar", stage, {n for n, f in expected.items() if f["in"] == STATE_NAME})
        for name, entry in expected.items():
            if entry["in"] != STATE_NAME:
                if entry["in"] not in containers:
                    raise BackupError(f"{name} names a ciphertext the manifest does not list")
                _openssl(True, key, src / entry["in"], stage / name)
        for name, entry in expected.items():
            if (stage / name).stat().st_size != entry["bytes"] or _sha_file(stage / name) != entry["sha256"]:
                raise BackupError(f"{name}: the decrypted bytes do not match the manifest")
        states, rows = arm_states(stage)
        if states != manifest["arms"]:
            raise BackupError("the decrypted journals and metas do not reproduce the manifest's per-arm fields")
        notes = run_log_gaps(states, rows, run_log)

        work.mkdir(parents=True, exist_ok=True)
        existing, _ = arm_states(work)
        problems = not_a_continuation(existing, states, rows, "in the work dir", "in the backup")
        keep_existing = set()  # logs the work dir already holds and extends
        for name in sorted(expected):
            target = work / name
            if not target.exists() or _sha_file(target) == expected[name]["sha256"]:
                continue
            if wanted(name) == "adapter":
                problems.append(f"{name}: different adapter bytes in the work dir")
            elif LOG_RE.match(name):
                mine, theirs = target.read_bytes(), (stage / name).read_bytes()
                if mine.startswith(theirs):
                    keep_existing.add(name)
                elif not theirs.startswith(mine):
                    problems.append(f"{name}: the work dir's copy and the backup's have each lines the other lacks")
        if problems and not force:
            raise BackupError("the work dir is newer than this backup or diverges from it; restoring "
                              "would overwrite it:\n  " + "\n  ".join(problems)
                              + "\nNothing was changed. Pass --force to restore anyway "
                              "(replaced files are moved aside, not deleted).")

        aside = work / f"superseded_{_utc().replace(':', '')}"
        written, moved = [], []
        for name in sorted(expected):
            target = work / name
            if target.exists() and (name in keep_existing
                                    or _sha_file(target) == expected[name]["sha256"]):
                continue
            # the new bytes are complete on disk before the old file moves, so a
            # crash in between never leaves a meta or a journal missing
            partial = _put(stage / name, target)
            if target.exists():
                aside.mkdir(exist_ok=True)
                os.replace(target, aside / name)
                moved.append(name)
            os.replace(partial, target)
            if _sha_file(target) != expected[name]["sha256"]:
                raise BackupError(f"{name}: the written file does not match the manifest")
            written.append(name)

    manifest_sha = _sha_file(src / MANIFEST_NAME)
    with open(work / "run.log", "a") as handle:  # the operator log records the restore itself
        handle.write(f"{_utc()} addendum_z_backup: restored the backup made {manifest['utc']} "
                     f"(manifest sha256 {manifest_sha}): {len(written)} files written, "
                     f"{len(moved)} moved to {aside.name if moved else '-'}"
                     f"{', forced past: ' + '; '.join(problems) if problems else ''}\n")
        handle.writelines(f"{_utc()} addendum_z_backup: {note}\n" for note in notes)
    return {"written": written, "moved": moved, "aside": str(aside) if moved else None,
            "notes": notes, "forced_past": problems}


# -- command line ----------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    k = sub.add_parser("keygen", help="write a random 256-bit key (hex) to a new 0600 file")
    k.add_argument("--key-file", type=pathlib.Path, required=True)
    b = sub.add_parser("backup", help="encrypt work/z into --out, with a public manifest")
    b.add_argument("--work", type=pathlib.Path, default=z.DEFAULT_WORK)
    b.add_argument("--out", type=pathlib.Path, required=True)
    b.add_argument("--key-file", type=pathlib.Path, required=True)
    b.add_argument("--force", action="store_true", help="replace a backup the work dir does not extend")
    r = sub.add_parser("restore", help="verify and decrypt a backup into --work")
    r.add_argument("--in", dest="src", type=pathlib.Path, required=True)
    r.add_argument("--key-file", type=pathlib.Path, required=True)
    r.add_argument("--work", type=pathlib.Path, default=z.DEFAULT_WORK)
    r.add_argument("--force", action="store_true", help="overwrite a newer or diverging work dir")
    r.add_argument("--run-log", type=pathlib.Path, default=RUN_LOG)
    args = ap.parse_args(argv)
    try:
        if args.command == "keygen":
            keygen(args.key_file)
            print(f"wrote a 256-bit key to {args.key_file} (mode 0600). Keep a copy privately, off this "
                  "container: it is reclaimed when the session ends, and without the key no backup "
                  "made with it can be read. Never commit it.")
        elif args.command == "backup":
            manifest = backup(args.work, args.out, args.key_file, args.force)
            print(f"backed up {len(manifest['files'])} files to {args.out.resolve()}: "
                  f"{STATE_NAME} {manifest['state']['ciphertext_bytes']} bytes, "
                  f"{len(manifest['blobs'])} adapter file(s)")
            for state in manifest["arms"]:
                print(f"{state['arm']}: restarts {state['restarts']}, decoded {state['documents_decoded']}, "
                      f"adapter {'yes' if state['adapter_sha256'] else 'no'}")
        else:
            result = restore(args.src, args.key_file, args.work, args.force, args.run_log)
            print(f"restored {len(result['written'])} files into {args.work.resolve()}"
                  + (f"; moved {len(result['moved'])} to {result['aside']}" if result["moved"] else ""))
            for note in result["notes"]:
                print(f"note: {note}")
    except BackupError as error:
        print(f"addendum_z_backup: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
