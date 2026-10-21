"""Addendum Z's backup: the run state outlives a reclaimed container, in a
public branch, with no score and no output text readable there. Pinned on
synthetic work dirs only; never touches work/z, a model, or the real /proc."""
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bk = _load("addendum_z_backup")
z = bk.z
REAL_RUNNER_ALIVE = bk.runner_alive
needs_openssl = pytest.mark.skipif(
    shutil.which("openssl") is None,
    reason="the openssl CLI is not installed; the backup encrypts with `openssl enc`")

A1, K1 = z.ckpt_stem(1, "adapted"), z.ckpt_stem(1, "kshot")
SENTINEL_TEXT = "SENTINEL-PREDICTION-7f3a"
SENTINEL_SCORE = 0.8123456789
FORBIDDEN_KEYS = {"prediction", "completions", "distinct_log_probabilities", "candidates", "text",
                  "valid_json", "exact_match", "micro_f1_raw", "score", "scores", "mean_micro_f1",
                  "gold", "seconds", "adapt_seconds", "prompt_tokens", "prompt_sha256"}
ALLOWED_KEYS = {"format", "what", "utc", "cipher", "tool", "kdf", "passphrase", "mac", "compression",
                "state", "file", "ciphertext_sha256", "ciphertext_bytes", "blobs", "files", "name",
                "sha256", "bytes", "in", "arms", "arm", "started_utc", "restarts", "adapter_sha256",
                "documents_decoded", "prefix_documents", "prefix_sha256", "hmac_sha256"}


@pytest.fixture(autouse=True)
def _no_live_runner(monkeypatch):
    # the real check scans /proc; these tests never do
    monkeypatch.setattr(bk, "runner_alive", lambda work, proc=None: False)


@pytest.fixture
def key(tmp_path):
    path = tmp_path / "keys" / "z.key"
    bk.keygen(path)
    return path


def _row(i, tweak=""):
    return {"index": i, "prompt_tokens": 100 + i, "prompt_sha256": "p" * 64,
            "completions": [f"{SENTINEL_TEXT} {i}{tweak}", "{}"],
            "distinct_log_probabilities": [[f"{SENTINEL_TEXT} {i}{tweak}", -0.5]],
            "candidates": [{"text": f"{SENTINEL_TEXT} {i}{tweak}", "count": 1}],
            "prediction": f"{SENTINEL_TEXT} {i}{tweak}", "valid_json": True, "exact_match": False,
            "micro_f1_raw": SENTINEL_SCORE, "seconds": 1.5}


def _work(root, rows_a=3, restarts=1, tweak=None, adapter=b"weights-v1" * 400):
    work = root / "work_z"
    work.mkdir(parents=True)
    (work / f"{A1}_adapter.pt").write_bytes(adapter)
    (work / f"{A1}_meta.json").write_text(json.dumps(
        {"restarts": restarts, "started_utc": "2026-10-01T00:00:00Z", "adapter_sha256": "a" * 64,
         "adapt_seconds": 1234.5}))
    (work / f"{A1}_docs.jsonl").write_text(
        "".join(json.dumps(_row(i, "x" if i == tweak else "")) + "\n" for i in range(rows_a)))
    (work / f"{K1}_meta.json").write_text(json.dumps({"restarts": 0, "started_utc": "2026-10-02T00:00:00Z"}))
    # a torn last line, mid-append: whole lines only go into the backup
    (work / f"{K1}_docs.jsonl").write_text(
        "".join(json.dumps(_row(i)) + "\n" for i in range(2)) + '{"index": 2, "predic')
    (work / "run.log").write_text('{"seed": 1, "arm": "adapted", "index": 0, "pool": 5}\n')
    (work / "runner.pid").write_text("999999\n")
    (work / f"{A1}_meta.json.tmp").write_text("{}")
    return work


def _snapshot(directory):
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir()) if p.is_file()}


def _keys(node):
    if isinstance(node, dict):
        return set(node) | set().union(*(_keys(v) for v in node.values()))
    if isinstance(node, list):
        return set().union(*(_keys(v) for v in node)) if node else set()
    return set()


def test_keygen_writes_a_private_key_once_and_never_inside_a_repository(tmp_path):
    path = tmp_path / "keys" / "z.key"
    bk.keygen(path)
    text = path.read_text()
    assert re.fullmatch(r"[0-9a-f]{64}\n", text)
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(bk.BackupError, match="refusing to overwrite"):
        bk.keygen(path)
    assert path.read_text() == text
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    with pytest.raises(bk.BackupError, match="inside the git work tree"):
        bk.keygen(repo / "sub" / "z.key")
    shutil.copy(path, repo / "z.key")
    with pytest.raises(bk.BackupError, match="inside the git work tree"):
        bk.read_key(repo / "z.key")


def test_the_arm_fields_are_the_run_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(bk.runlog, "REPO", tmp_path)  # `banked` then looks in tmp_path only
    work = _work(tmp_path)
    states, _ = bk.arm_states(work)
    expected = []
    for seed, arm in z.ARM_ORDER:
        if (work / f"{z.ckpt_stem(seed, arm)}_meta.json").exists():
            state = bk.runlog.arm_state(seed, arm, work)
            state.pop("banked")
            expected.append(state)
    assert states == expected and len(states) == 2


@needs_openssl
def test_roundtrip_restores_every_byte_and_the_documented_manual_recovery_works(tmp_path, key):
    work = _work(tmp_path / "a")
    original = _snapshot(work)
    out = tmp_path / "branch"
    manifest = bk.backup(work, out, key)
    adapter_sha = manifest["files"][0]["sha256"]
    assert sorted(os.listdir(out)) == sorted([bk.STATE_NAME, bk.MANIFEST_NAME,
                                              f"{A1}_adapter.pt.{adapter_sha[:16]}.enc"])
    assert {f["name"] for f in manifest["files"]} == {
        f"{A1}_adapter.pt", f"{A1}_meta.json", f"{A1}_docs.jsonl", f"{K1}_meta.json", f"{K1}_docs.jsonl",
        "run.log"}
    assert _snapshot(work) == original  # backup only reads

    fresh = tmp_path / "b" / "work_z"
    result = bk.restore(out, key, fresh, run_log=tmp_path / "no_run_log.json")
    assert result["moved"] == [] and result["notes"] == []
    restored = _snapshot(fresh)
    for name in (f"{A1}_adapter.pt", f"{A1}_meta.json", f"{A1}_docs.jsonl", f"{K1}_meta.json"):
        assert restored[name] == original[name], name
    torn = original[f"{K1}_docs.jsonl"]
    assert restored[f"{K1}_docs.jsonl"] == torn[: torn.rfind(b"\n") + 1]
    assert restored["run.log"].startswith(original["run.log"])
    assert b"restored the backup made" in restored["run.log"]
    assert "runner.pid" not in restored and f"{A1}_meta.json.tmp" not in restored

    # the module docstring's last-resort command: the key file is the passphrase
    plain = tmp_path / "manual.tar"
    subprocess.run(["openssl", *bk.OPENSSL_ENC, "-d", "-pass", f"file:{key}",
                    "-in", str(out / bk.STATE_NAME), "-out", str(plain)], check=True)
    with tarfile.open(plain) as tar:
        assert sorted(tar.getnames()) == sorted(f["name"] for f in manifest["files"]
                                                if f["in"] == bk.STATE_NAME)


@needs_openssl
def test_tampering_is_refused_before_anything_is_written(tmp_path, key):
    out = tmp_path / "branch"
    manifest = bk.backup(_work(tmp_path / "a"), out, key)
    blob = out / manifest["blobs"][0]["file"]
    fresh = tmp_path / "b" / "work_z"
    no_log = tmp_path / "no_run_log.json"

    def refused(match, key_path=key):
        with pytest.raises(bk.BackupError, match=match):
            bk.restore(out, key_path, fresh, run_log=no_log)
        assert not fresh.exists() or not any(fresh.iterdir())

    for path in (out / bk.STATE_NAME, blob):  # one flipped ciphertext byte
        good = path.read_bytes()
        bad = bytearray(good)
        bad[len(bad) // 2] ^= 0x01
        path.write_bytes(bytes(bad))
        refused("ciphertext sha256 does not match")
        path.write_bytes(good)

    text = (out / bk.MANIFEST_NAME).read_text()
    edited = json.loads(text)
    edited["arms"][0]["restarts"] = 0  # hide a restart
    (out / bk.MANIFEST_NAME).write_text(json.dumps(edited))
    refused("does not authenticate")
    edited = json.loads(text)
    edited["state"]["ciphertext_sha256"] = "0" * 64
    (out / bk.MANIFEST_NAME).write_text(json.dumps(edited))
    refused("does not authenticate")
    (out / bk.MANIFEST_NAME).write_text(text)

    other = tmp_path / "keys" / "other.key"
    bk.keygen(other)
    refused("does not authenticate", other)
    bk.restore(out, key, fresh, run_log=no_log)  # the untouched backup still restores


@needs_openssl
def test_restore_refuses_to_overwrite_a_newer_or_diverging_work_dir_unless_forced(tmp_path, key):
    out = tmp_path / "branch"
    bk.backup(_work(tmp_path / "a", rows_a=3), out, key)
    no_log = tmp_path / "no_run_log.json"

    for label, kwargs, match in (
        ("more_rows", {"rows_a": 5}, "5 documents in the work dir, 3 in the backup"),
        ("more_restarts", {"restarts": 4}, "4 restarts in the work dir, 1 in the backup"),
        ("redrawn", {"tweak": 1}, "first 3 documents in the work dir are not"),
        ("other_adapter", {"adapter": b"weights-v2" * 400}, "different adapter bytes"),
    ):
        newer = _work(tmp_path / label, **kwargs)
        before = _snapshot(newer)
        with pytest.raises(bk.BackupError, match=match):
            bk.restore(out, key, newer, run_log=no_log)
        assert _snapshot(newer) == before, label

    newer = _work(tmp_path / "forced", rows_a=5)
    before = _snapshot(newer)
    result = bk.restore(out, key, newer, force=True, run_log=no_log)
    assert f"{A1}_docs.jsonl" in result["moved"]
    aside = pathlib.Path(result["aside"])
    assert (aside / f"{A1}_docs.jsonl").read_bytes() == before[f"{A1}_docs.jsonl"]  # moved, not deleted
    assert len((newer / f"{A1}_docs.jsonl").read_text().splitlines()) == 3
    assert b"forced past" in (newer / "run.log").read_bytes()

    older = _work(tmp_path / "older", rows_a=1, restarts=0)  # the backup extends it: no force needed
    bk.restore(out, key, older, run_log=no_log)
    assert len((older / f"{A1}_docs.jsonl").read_text().splitlines()) == 3
    bk.restore(out, key, older, run_log=no_log)  # and again: idempotent, the appended log line is kept


@needs_openssl
def test_backup_refuses_to_replace_a_newer_backup_and_reuses_an_unchanged_adapter(tmp_path, key):
    out = tmp_path / "branch"
    first = bk.backup(_work(tmp_path / "a", rows_a=5), out, key)
    blob = out / first["blobs"][0]["file"]
    blob_bytes, published = blob.read_bytes(), _snapshot(out)

    # a new session that started the runner before restoring, and one that redrew a document
    for label, kwargs, match in (("fresh", {"rows_a": 2}, "5 documents in the backup"),
                                 ("redraw", {"rows_a": 5, "tweak": 3}, "a redraw")):
        with pytest.raises(bk.BackupError, match=match):
            bk.backup(_work(tmp_path / label, **kwargs), out, key)
        assert _snapshot(out) == published, label

    second = bk.backup(_work(tmp_path / "continued", rows_a=7, restarts=2), out, key)
    assert second["blobs"] == first["blobs"] and blob.read_bytes() == blob_bytes  # not re-encrypted
    assert second["arms"][0]["documents_decoded"] == 7

    bk.backup(_work(tmp_path / "forced", rows_a=2), out, key, force=True)
    assert json.loads((out / bk.MANIFEST_NAME).read_text())["arms"][0]["documents_decoded"] == 2

    replaced = bk.backup(_work(tmp_path / "new_adapter", rows_a=2, adapter=b"weights-v3" * 400),
                         out, key, force=True)
    assert not blob.exists() and (out / replaced["blobs"][0]["file"]).exists()  # stale adapter removed


@needs_openssl
def test_the_public_manifest_and_outputs_carry_no_score_and_no_text(tmp_path, key, capsys):
    work, out = _work(tmp_path / "a"), tmp_path / "branch"
    assert bk.main(["backup", "--work", str(work), "--out", str(out), "--key-file", str(key)]) == 0
    manifest = json.loads((out / bk.MANIFEST_NAME).read_text())
    keys = _keys(manifest)
    assert not keys & FORBIDDEN_KEYS
    assert keys <= ALLOWED_KEYS, keys - ALLOWED_KEYS
    blob = (out / bk.MANIFEST_NAME).read_text()
    for needle in (SENTINEL_TEXT, str(SENTINEL_SCORE), "1234.5", '"pool"'):
        assert needle not in blob, needle
    for path in out.iterdir():  # nothing in the branch dir holds plaintext
        data = path.read_bytes()
        assert SENTINEL_TEXT.encode() not in data and str(SENTINEL_SCORE).encode() not in data, path.name
    assert bk.main(["restore", "--in", str(out), "--key-file", str(key), "--work",
                    str(tmp_path / "b" / "work_z"), "--run-log", str(tmp_path / "none.json")]) == 0
    printed = capsys.readouterr()
    assert SENTINEL_TEXT not in printed.out + printed.err
    assert str(SENTINEL_SCORE) not in printed.out + printed.err


@needs_openssl
def test_restore_names_what_the_public_run_log_banked_that_the_backup_lacks(tmp_path, key):
    out = tmp_path / "branch"
    bk.backup(_work(tmp_path / "a", rows_a=3), out, key)
    run_log = tmp_path / "run_log.json"
    run_log.write_text(json.dumps({"snapshots": [{"utc": "2026-10-20T10:00:00Z", "arms": [
        {"arm": z.artifact_name(1, "adapted"), "restarts": 2, "prefix_documents": 5,
         "prefix_sha256": "f" * 64}]}]}))
    fresh = tmp_path / "b" / "work_z"
    result = bk.restore(out, key, fresh, run_log=run_log)
    assert any("banked 5 documents" in n and "documents 3..4" in n for n in result["notes"])
    assert any("saw 2 restarts" in n for n in result["notes"])
    assert b"banked 5 documents" in (fresh / "run.log").read_bytes()


@needs_openssl
def test_restore_never_runs_under_a_live_runner(tmp_path, key, monkeypatch):
    out = tmp_path / "branch"
    bk.backup(_work(tmp_path / "a"), out, key)
    monkeypatch.setattr(bk, "runner_alive", lambda work, proc=None: True)
    with pytest.raises(bk.BackupError, match="runner is alive"):
        bk.restore(out, key, tmp_path / "b" / "work_z", force=True, run_log=tmp_path / "none.json")


def test_runner_detection_reads_a_proc_tree(tmp_path):
    proc, repo = tmp_path / "proc", tmp_path / "repo"
    (repo / "work" / "z").mkdir(parents=True)

    def process(pid, argv, cwd=repo):
        (proc / pid).mkdir(parents=True)
        (proc / pid / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
        os.symlink(cwd, proc / pid / "cwd")

    process("10", ["python3", "scripts/novel_schema_rerun.py", "--read"])
    assert not REAL_RUNNER_ALIVE(repo / "work" / "z", proc)
    process("11", ["python3", "scripts/novel_schema_rerun.py", "--run"])
    assert REAL_RUNNER_ALIVE(repo / "work" / "z", proc)
    assert not REAL_RUNNER_ALIVE(tmp_path / "elsewhere", proc)
    process("12", ["python3", "scripts/novel_schema_rerun.py", "--run", "--work", "/srv/zz"])
    assert REAL_RUNNER_ALIVE(pathlib.Path("/srv/zz"), proc)


@needs_openssl
def test_backup_warns_when_git_would_leave_it_out(tmp_path, key, capsys):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "public"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("work/\n")
    bk.backup(_work(tmp_path / "a"), repo / "work" / "z_backup", key)
    assert "git ignores" in capsys.readouterr().err
