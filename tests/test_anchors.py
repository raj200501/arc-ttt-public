"""Every frozen addendum protocol since S carries an OpenTimestamps anchor
beside it, and ANCHORS.md lists it with an honest answer to the only
question that matters: does the anchor precede the run? For everything
anchored on 2026-09-17 the answer is no, and the page must say so."""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
RESEARCH = REPO / "docs" / "research"
FROZEN = ["S", "T", "U", "U_EXT", "V", "W", "X", "Y", "Z"]
# anchored BEFORE the first arm (the ANCHORS.md policy); everything
# earlier was anchored after its run and must say so
BEFORE_RUN = {"Z"}


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
        key = row.split("`")[1].removeprefix("ADDENDUM_").removesuffix("_PROTOCOL.md")
        if key in BEFORE_RUN:
            assert "**yes**" in row and "2026-09-23T" in row, row
        else:
            assert "2026-09-17" in row and "**no**" in row, row
    assert "does not prove the protocol was frozen before its data" in text


def test_the_before_run_anchor_is_quoted_in_its_freeze_line():
    """Z's protocol names its own stamp, the stamp file exists with exactly
    that name, and the same stamp is in ANCHORS.md, the runner's PROTOCOL
    string (every arm carries it) and the VERDICT row — no placeholder
    survives anywhere the stamp must appear."""
    text = (RESEARCH / "ADDENDUM_Z_PROTOCOL.md").read_text()
    match = re.search(r"ADDENDUM_Z_PROTOCOL\.md\.(2026-09-23T\d{4}Z)\.ots", text)
    assert match, "freeze line does not quote a stamp"
    stamp = match.group(1)
    assert (RESEARCH / f"ADDENDUM_Z_PROTOCOL.md.{stamp}.ots").exists()
    for rel in ("docs/research/ANCHORS.md", "scripts/novel_schema_rerun.py", "VERDICT.md",
                "docs/research/ADDENDUM_Z_PROTOCOL.md"):
        body = (REPO / rel).read_text()
        assert "STAMP_PLACEHOLDER" not in body, rel
        assert stamp in body, rel
    row = [l for l in (RESEARCH / "ANCHORS.md").read_text().splitlines()
           if l.startswith("| `ADDENDUM_Z_PROTOCOL.md`")][0]
    assert f"| {stamp} |" in row


def test_anchor_files_are_real_ots_blobs():
    magic = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
    for p in list(RESEARCH.glob("*.ots")) + list((REPO / "experiments").glob("*.ots")):
        assert p.read_bytes().startswith(magic), p.name
    assert re.search(r"ADDENDUM_Y_PROTOCOL\.md\.\d{4}-\d{2}-\d{2}T\d{4}Z\.ots",
                     " ".join(q.name for q in RESEARCH.glob("*.ots")))


BITCOIN_TAG = bytes.fromhex("0588960d73d71901")  # BitcoinBlockHeaderAttestation


def test_every_row_that_names_a_block_has_a_bitcoin_attestation():
    """A calendar promise is not a timestamp. Until 2026-09-23 ten proofs
    here were PendingAttestation-only; a row that states a block height
    must be backed by a proof that actually carries one."""
    text = (RESEARCH / "ANCHORS.md").read_text()
    proofs = list(RESEARCH.glob("*.ots")) + list((REPO / "experiments").glob("*.ots"))
    for row in text.splitlines():
        if not row.startswith("| `") or not re.search(r"\| \**\d{6} \(", row):
            continue
        names = re.findall(r"`([^`]+)`", row.split("|")[1])
        for name in names:
            base = name.split("/")[-1]
            matches = [p for p in proofs if p.name.startswith(base.rstrip("`"))]
            assert matches, (base, "no proof on disk for a row naming a block")
            for proof in matches:
                assert BITCOIN_TAG in proof.read_bytes(), proof.name
