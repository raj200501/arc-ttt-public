"""Every frozen addendum protocol since S carries an OpenTimestamps anchor
beside it, and ANCHORS.md lists it with an honest answer to the only
question that matters: does the anchor precede the run? For everything
anchored on 2026-09-17 the answer is no, and the page must say so."""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
RESEARCH = REPO / "docs" / "research"
FROZEN = ["S", "T", "U", "U_EXT", "V", "W", "X", "Y"]


def test_every_frozen_protocol_since_s_has_an_anchor():
    missing = [k for k in FROZEN
               if not list(RESEARCH.glob(f"ADDENDUM_{k}_PROTOCOL.md.*.ots"))]
    assert not missing, missing


def test_anchors_page_lists_them_and_does_not_overclaim():
    text = (RESEARCH / "ANCHORS.md").read_text()
    for k in FROZEN:
        assert f"ADDENDUM_{k}_PROTOCOL.md" in text, k
    # the after-the-run anchors must be marked as not proving the freeze
    rows = [l for l in text.splitlines() if l.startswith("| `ADDENDUM_")]
    assert rows, "no protocol rows"
    for row in rows:
        assert "2026-09-17" in row and "**no**" in row, row
    assert "does not prove the protocol was frozen before its data" in text


def test_anchor_files_are_real_ots_blobs():
    magic = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
    for p in list(RESEARCH.glob("*.ots")) + list((REPO / "experiments").glob("*.ots")):
        assert p.read_bytes().startswith(magic), p.name
    assert re.search(r"ADDENDUM_Y_PROTOCOL\.md\.\d{4}-\d{2}-\d{2}T\d{4}Z\.ots",
                     " ".join(q.name for q in RESEARCH.glob("*.ots")))
