"""The frozen spec must begin with its last anchored snapshot, byte for byte.

`docs/research/ENTERPRISE_EVAL_SPEC.md` is frozen text with OpenTimestamps
anchors. Everything after the anchored snapshot is appended (later
addenda, errata); nothing inside it may change. That rule was broken
three times without an erratum — once by an audit sweep editing a sign
test in place, twice by the test-count syncer — and a same-day audit
found two of the three and an outbound document then said the file
matched its anchors (erratum P15, CORRECTIONS 2026-09-23). This makes the
rule mechanical instead of remembered.
"""
import hashlib
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
RESEARCH = REPO / "docs" / "research"
SPEC = RESEARCH / "ENTERPRISE_EVAL_SPEC.md"
MAGIC = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"


def _anchored_digest(ots: pathlib.Path) -> str:
    blob = ots.read_bytes()
    assert blob.startswith(MAGIC) and blob[len(MAGIC)] == 1 and blob[len(MAGIC) + 1] == 0x08, ots
    return blob[len(MAGIC) + 2: len(MAGIC) + 34].hex()


def _snapshots():
    out = []
    for snap in sorted(RESEARCH.glob("snapshots_ENTERPRISE_EVAL_SPEC_*.md")):
        stamp = re.search(r"_(\d{4}-\d{2}-\d{2}T\d{4}Z)\.md$", snap.name).group(1)
        out.append((stamp, snap, RESEARCH / f"ENTERPRISE_EVAL_SPEC.md.{stamp}.ots"))
    return out


def test_every_snapshot_is_the_file_its_anchor_commits_to():
    snaps = _snapshots()
    assert snaps, "no anchored snapshots"
    for stamp, snap, ots in snaps:
        assert ots.exists(), ots.name
        assert hashlib.sha256(snap.read_bytes()).hexdigest() == _anchored_digest(ots), stamp


def test_the_spec_begins_with_its_latest_anchored_snapshot_byte_for_byte():
    stamp, snap, _ = _snapshots()[-1]
    current, anchored = SPEC.read_bytes(), snap.read_bytes()
    if current.startswith(anchored):
        return
    a, b = anchored.decode().splitlines(), current.decode().splitlines()
    first = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y) if any(
        x != y for x, y in zip(a, b)) else min(len(a), len(b))
    raise AssertionError(
        f"ENTERPRISE_EVAL_SPEC.md no longer begins with its anchored snapshot ({stamp}); "
        f"first differing line {first + 1}:\n  anchored: {a[first] if first < len(a) else '<end>'!r}"
        f"\n  current:  {b[first] if first < len(b) else '<end>'!r}\n"
        "Frozen text is corrected by a dated erratum appended after the anchored region, "
        "never in place.")


def test_the_syncer_can_no_longer_touch_frozen_files():
    import importlib.util
    spec = importlib.util.spec_from_file_location("sync", REPO / "scripts" / "sync_test_counts.py")
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    swept = {p.name for p in sync.documents()}
    assert "ENTERPRISE_EVAL_SPEC.md" not in swept
    assert not any(name.endswith("_PROTOCOL.md") for name in swept)
