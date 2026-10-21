"""A dated section is a record, and no count syncer may rewrite it.

A heading such as "## Update — 2026-09-17: ..." dates everything under it.
The artifact-count syncer rewrote "236 banked artifacts" inside a section
dated 2026-09-17 to 237, a count that first existed on 2026-09-23, so the
section quoted a number that did not exist on its date (CORRECTIONS
2026-10-21). All three syncers that rewrite counts now skip text under a
dated heading. These tests pin that for each of them, and pin that text
outside a dated section is still rewritten, so the skip cannot quietly
grow into a skip of everything.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"_dated_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sync = _load("sync_test_counts")
gate = _load("currency_gate")

DOC = (
    "# A page\n\n"
    "Current text: 312 banked artifacts, 40 offline tests and 50 dated corrections.\n\n"
    "## Update — 2026-09-17: what was true that day\n\n"
    "Then: 236 banked artifacts, 30 offline tests and 98 dated corrections.\n\n"
    "### A sub-heading inside the dated section\n\n"
    "Still dated: 235 banked artifacts.\n\n"
    "## A current section again\n\n"
    "Now: 311 banked artifacts.\n"
)
DATED_PART = DOC[DOC.index("## Update"):DOC.index("## A current section")]


def _targets() -> dict[str, int]:
    return {"artifacts": 237, "primary_verifiable": 56, "export_tests": 1,
            "export_artifacts": 1, "tool_tests": 1}


def test_the_dated_span_runs_to_the_next_heading_of_its_level():
    spans = sync.dated_spans(DOC)
    assert len(spans) == 1
    start, end = spans[0]
    assert DOC[start:end] == DATED_PART
    assert sync.in_dated_section(DOC, DOC.index("235 banked"))
    assert not sync.in_dated_section(DOC, DOC.index("311 banked"))
    assert not sync.in_dated_section(DOC, DOC.index("312 banked"))


def test_the_test_count_syncer_skips_a_dated_section(tmp_path):
    page = tmp_path / "page.md"
    page.write_text(DOC, encoding="utf-8")
    stale = sync.stale_claims(page, 999)
    assert [claimed for _, claimed in stale] == [40], stale
    rewritten = sync.rewrite(DOC, 999)
    assert "999 offline tests" in rewritten
    assert "30 offline tests" in rewritten, "a dated count was rewritten"
    assert DATED_PART in rewritten


def test_the_artifact_count_gate_skips_a_dated_section(monkeypatch):
    monkeypatch.setattr(gate, "_targets", _targets)
    assert gate._context_of(DOC, DOC.index("236 banked")) == "dated"
    fixed, changed = gate.fix(DOC)
    assert "237 banked artifacts, 40 offline" in fixed
    assert "Now: 237 banked artifacts" in fixed
    assert DATED_PART in fixed, "the gate rewrote a count under a dated heading"
    assert sorted(changed) == ["artifacts: 311 -> 237", "artifacts: 312 -> 237"]
    found = gate.scan(DOC)
    assert {line for _, _, _, line in found["stale"]} == {3, 15}
    assert not found["unowned"]


def test_the_correction_count_syncer_skips_a_dated_section(tmp_path, monkeypatch):
    ledger = _load("self_correction_ledger")
    page = tmp_path / "copy.md"
    page.write_text(DOC, encoding="utf-8")
    monkeypatch.setattr(ledger, "SYNCED_COPIES", (page,))
    changed = ledger.sync_copy(107, 72)      # must not refuse the dated 98
    text = page.read_text(encoding="utf-8")
    assert "107 dated corrections" in text
    assert "98 dated corrections" in text
    assert changed == ["copy.md total: 50 -> 107"]


def test_no_dated_section_in_the_tree_is_touched_by_the_gate(monkeypatch):
    """Over the real documents: whatever the targets, --fix leaves every
    byte under a dated heading as it was."""
    monkeypatch.setattr(gate, "_targets", lambda: {
        "artifacts": 9001, "primary_verifiable": 9002, "export_tests": 9003,
        "export_artifacts": 9004, "tool_tests": 9005})
    checked = 0
    for path in sync.documents():
        text = path.read_text(encoding="utf-8")
        spans = sync.dated_spans(text)
        if not spans:
            continue
        fixed, _ = gate.fix(text)
        rewritten = sync.rewrite(text, 9006)
        for start, end in spans:
            part = text[start:end]
            assert part in fixed, (path.name, part[:80])
            assert part in rewritten, (path.name, part[:80])
        checked += 1
    if not (REPO / "docs" / "strategy").is_dir():
        return    # the documents with dated sections are not in this cut
    assert checked, "no document with a dated section was found to check"
