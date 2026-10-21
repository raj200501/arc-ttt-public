"""The freeze tool fills every placeholder or refuses, strips a marked draft
header, and writes a byte-identical snapshot. It digests only committed,
unchanged files. Stamping itself needs the ots client and the network, so
it is checked only for its refusals here."""
import hashlib
import importlib.util
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("fp", REPO / "scripts" / "freeze_protocol.py")
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)

DRAFT = """# Addendum Q

<!-- DRAFT-HEADER-START -->
**DRAFT, NOT FROZEN.** It quotes STAMP_PLACEHOLDER here too.
<!-- DRAFT-HEADER-END -->
**Frozen. Anchored as `ADDENDUM_Q_PROTOCOL.md.STAMP_PLACEHOLDER.ots`; snapshot
`snapshots_ADDENDUM_Q_PROTOCOL_STAMP_PLACEHOLDER.md`. Code:
`scripts/a.py` sha256 `A_SHA_PLACEHOLDER` and
`scripts/b.py` sha256 `B_SHA_PLACEHOLDER`.**

Body.
"""


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout


def _setup(tmp_path, monkeypatch, text=DRAFT):
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts" / "a.py").write_text("print('a')\n")
    (tmp_path / "scripts" / "b.py").write_text("print('b')\n")
    research = tmp_path / "docs" / "research"
    research.mkdir(parents=True)
    protocol = research / "ADDENDUM_Q_PROTOCOL.md"
    protocol.write_text(text)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "draft")
    monkeypatch.setattr(fp, "REPO", tmp_path)
    return tmp_path, protocol


def test_prepare_fills_everything_strips_the_draft_header_and_snapshots(tmp_path, monkeypatch):
    root, protocol = _setup(tmp_path, monkeypatch)
    stamp = fp.prepare(protocol, ["scripts/a.py", "scripts/b.py"], "2026-10-21T2100Z")
    text = protocol.read_text()
    assert "PLACEHOLDER" not in text and "DRAFT" not in text
    assert "ADDENDUM_Q_PROTOCOL.md.2026-10-21T2100Z.ots" in text
    for name in ("a.py", "b.py"):
        digest = hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest()
        assert f"`scripts/{name}` sha256 `{digest}`" in text
    snap = fp.snapshot_path(protocol, stamp)
    assert snap.read_bytes() == protocol.read_bytes()


def test_prepare_refuses_a_missing_digest_placeholder_or_a_frozen_file(tmp_path, monkeypatch):
    root, protocol = _setup(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        fp.prepare(protocol, ["scripts/a.py"], "2026-10-21T2100Z")  # B's placeholder left over
    root, protocol = _setup(tmp_path / "x", monkeypatch, DRAFT.replace("STAMP_PLACEHOLDER", "2026-01-01T0000Z"))
    with pytest.raises(SystemExit):
        fp.prepare(protocol, ["scripts/a.py", "scripts/b.py"], "2026-10-21T2100Z")


def test_prepare_refuses_a_bad_stamp_and_a_half_marked_header(tmp_path, monkeypatch):
    root, protocol = _setup(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        fp.prepare(protocol, ["scripts/a.py", "scripts/b.py"], "today")
    with pytest.raises(SystemExit):
        fp.strip_draft_header("x <!-- DRAFT-HEADER-START --> y")


def test_stamp_refuses_without_a_quoted_stamp_or_a_matching_snapshot(tmp_path, monkeypatch):
    root, protocol = _setup(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        fp.stamp(protocol)  # nothing prepared
    fp.prepare(protocol, ["scripts/a.py", "scripts/b.py"], "2026-10-21T2100Z")
    fp.snapshot_path(protocol, "2026-10-21T2100Z").write_text("different")
    with pytest.raises(SystemExit):
        fp.stamp(protocol)


def test_prepare_refuses_a_code_file_that_is_not_committed_or_is_dirty(tmp_path, monkeypatch):
    root, protocol = _setup(tmp_path, monkeypatch)
    (root / "scripts" / "b.py").write_text("print('b, edited and not committed')\n")
    with pytest.raises(SystemExit, match="not committed"):
        fp.prepare(protocol, ["scripts/a.py", "scripts/b.py"], "2026-10-21T2100Z")
    assert "B_SHA_PLACEHOLDER" in protocol.read_text()  # nothing was written
    root, protocol = _setup(tmp_path / "untracked", monkeypatch)
    (root / "scripts" / "c.py").write_text("print('c')\n")
    with pytest.raises(SystemExit, match="not committed"):
        fp.prepare(protocol, ["scripts/a.py", "scripts/c.py"], "2026-10-21T2100Z")


def test_prepare_quotes_posix_paths(tmp_path, monkeypatch):
    root, protocol = _setup(tmp_path, monkeypatch)
    fp.prepare(protocol, [str(root / "scripts" / "a.py"), "scripts/b.py"], "2026-10-21T2100Z")
    assert "`scripts/a.py` sha256 `" in protocol.read_text()


def test_stamp_refuses_when_a_quoted_file_changed_after_prepare(tmp_path, monkeypatch):
    root, protocol = _setup(tmp_path, monkeypatch)
    fp.prepare(protocol, ["scripts/a.py", "scripts/b.py"], "2026-10-21T2100Z")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "prepared")
    (root / "scripts" / "a.py").write_text("print('a, edited after prepare')\n")
    with pytest.raises(SystemExit, match="scripts/a.py"):
        fp.stamp(protocol)
    _git(root, "commit", "-qam", "edited and committed")
    with pytest.raises(SystemExit, match="no longer hashes"):
        fp.stamp(protocol)
