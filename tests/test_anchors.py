"""Every frozen addendum protocol since S carries an OpenTimestamps anchor
beside it, and ANCHORS.md lists it with an honest answer to the only
question that matters: does the anchor precede the run? For everything
anchored on 2026-09-17 the answer is no, and the page must say so."""
import pathlib
import re
from datetime import datetime, timedelta, timezone

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
RESEARCH = REPO / "docs" / "research"
FROZEN = ["S", "T", "U", "U_EXT", "V", "W", "X", "Y", "Z", "AA"]
# anchored BEFORE the first arm (the ANCHORS.md policy); everything
# earlier was anchored after its run and must say so. Each maps to the
# files that must quote its stamp.
BEFORE_RUN = {
    "Z": ("docs/research/ANCHORS.md", "scripts/novel_schema_rerun.py", "VERDICT.md"),
    "AA": ("docs/research/ANCHORS.md", "VERDICT.md"),
}
STAMP = r"\d{4}-\d{2}-\d{2}T\d{4}Z"
# A protocol registered here before its freeze is in one of three states,
# read from its own text: a draft (the freeze line still says
# STAMP_PLACEHOLDER; nothing may claim it is anchored), prepared (the stamp
# is quoted and the snapshot written, the proof not yet made: allowed for
# one day, the one commit `freeze_protocol.py stamp` requires before it
# runs), or stamped.
PREPARED_WINDOW = timedelta(hours=24)


def freeze_state(key: str) -> tuple[str, str | None]:
    """The protocols anchored after their run (S to Y) do not quote their
    own stamp; a proof beside them is what makes them stamped."""
    name = f"ADDENDUM_{key}_PROTOCOL.md"
    text = (RESEARCH / name).read_text()
    match = re.search(re.escape(name) + r"\.(" + STAMP + r")\.ots", text)
    stamp = match.group(1) if match else None
    if list(RESEARCH.glob(f"{name}.*.ots")):
        return "stamped", stamp
    if stamp is None:
        assert "STAMP_PLACEHOLDER" in text, f"{name}: no proof, no stamp and no placeholder"
        return "draft", None
    snapshot = RESEARCH / f"snapshots_ADDENDUM_{key}_PROTOCOL_{stamp}.md"
    assert snapshot.exists() and (RESEARCH / name).read_bytes().startswith(snapshot.read_bytes()), (
        f"{name} quotes {stamp} but has neither a proof nor its snapshot")
    made = datetime.strptime(stamp, "%Y-%m-%dT%H%MZ").replace(tzinfo=timezone.utc)
    assert datetime.now(timezone.utc) - made < PREPARED_WINDOW, (
        f"{name} was prepared at {stamp} and never stamped: run freeze_protocol.py stamp")
    return "prepared", stamp


def _draft_claims_nothing(key: str) -> None:
    name = f"ADDENDUM_{key}_PROTOCOL.md"
    assert not list(RESEARCH.glob(f"{name}.*.ots")), f"{name} is a draft with a proof"
    assert not list(RESEARCH.glob(f"snapshots_ADDENDUM_{key}_PROTOCOL_*.md")), name
    rows = [l for l in (RESEARCH / "ANCHORS.md").read_text().splitlines() if l.startswith(f"| `{name}`")]
    assert not rows, f"ANCHORS.md lists {name} before it is frozen"


def test_the_freeze_states_are_only_ever_draft_for_a_protocol_not_yet_frozen():
    for key in FROZEN:
        state, _ = freeze_state(key)
        if key != "AA":
            assert state == "stamped", key
        elif state == "draft":
            _draft_claims_nothing(key)


def test_every_frozen_protocol_since_s_has_an_anchor():
    missing = [k for k in FROZEN
               if freeze_state(k)[0] == "stamped"
               and not list(RESEARCH.glob(f"ADDENDUM_{k}_PROTOCOL.md.*.ots"))]
    assert not missing, missing


def test_anchors_page_lists_them_and_does_not_overclaim():
    text = (RESEARCH / "ANCHORS.md").read_text()
    for k in FROZEN:
        if freeze_state(k)[0] != "draft":
            assert f"ADDENDUM_{k}_PROTOCOL.md" in text, k
    # the after-the-run anchors must be marked as not proving the freeze
    rows = [l for l in text.splitlines() if l.startswith("| `ADDENDUM_")]
    assert rows, "no protocol rows"
    for row in rows:
        key = row.split("`")[1].removeprefix("ADDENDUM_").removesuffix("_PROTOCOL.md")
        if key in BEFORE_RUN:
            assert "**yes**" in row and re.search(r"\| \**" + STAMP, row), row
        else:
            assert "2026-09-17" in row and "**no**" in row, row
    assert "does not prove the protocol was frozen before its data" in text


@pytest.mark.parametrize("key", sorted(BEFORE_RUN))
def test_the_before_run_anchor_is_quoted_in_its_freeze_line(key):
    """Each before-run protocol names its own stamp, the proof exists with
    exactly that name, and the same stamp appears everywhere it must; no
    placeholder survives anywhere the stamp must appear. A protocol still in
    draft claims nothing; one prepared and not yet stamped quotes its stamp
    everywhere already, because the stamp is written in before `ots stamp`."""
    quoting = BEFORE_RUN[key]
    state, stamp = freeze_state(key)
    if state == "draft":
        _draft_claims_nothing(key)
        pytest.skip(f"Addendum {key} is a draft: nothing is anchored yet")
    name = f"ADDENDUM_{key}_PROTOCOL.md"
    if state == "stamped":
        assert (RESEARCH / f"{name}.{stamp}.ots").exists(), name
    for rel in (*quoting, f"docs/research/{name}"):
        body = (REPO / rel).read_text()
        assert "STAMP_PLACEHOLDER" not in body, (key, rel)
        assert stamp in body, (key, rel)
    row = [l for l in (RESEARCH / "ANCHORS.md").read_text().splitlines()
           if l.startswith(f"| `{name}`")][0]
    assert f"| {stamp} |" in row, key


def test_anchor_files_are_real_ots_blobs():
    magic = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
    for p in list(RESEARCH.glob("*.ots")) + list((REPO / "experiments").glob("*.ots")):
        assert p.read_bytes().startswith(magic), p.name
    assert re.search(r"ADDENDUM_Y_PROTOCOL\.md\.\d{4}-\d{2}-\d{2}T\d{4}Z\.ots",
                     " ".join(q.name for q in RESEARCH.glob("*.ots")))


BITCOIN_TAG = bytes.fromhex("0588960d73d71901")  # BitcoinBlockHeaderAttestation


def test_every_row_that_names_a_block_has_a_bitcoin_attestation():
    """A calendar promise is not a timestamp. Until 2026-09-23 fourteen of
    the sixteen proofs here were PendingAttestation-only (this docstring
    said ten until 2026-10-21); a row that states a block height must be
    backed by a proof that actually carries one."""
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
