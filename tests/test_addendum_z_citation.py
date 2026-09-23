"""Addendum Z's citation rule, held by a test on both sides of the landing.

Before the reading exists, every headline quote of +46.5 says the re-run
is pending, so no page presents the banked number as if nothing were
checking it. After the reading lands, the rule the protocol froze —
the primary-verifiable number replaces +46.5 whichever way it moved; a
non-GO withdraws it; DIFFERENT EXCLUSIONS moves nothing — is checked
against the pages mechanically, so it cannot be applied selectively on
the day the number is known.
"""
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
READING = REPO / "experiments" / "novel_schema_rerun_2026-09-23.json"
HEADLINE_PAGES = ("VERDICT.md", "EVIDENCE.md", "README.md")
PENDING = "re-run pending"


def headline_string(mean_delta: float) -> str:
    """The gate mean as every page quotes it: F1 points, one decimal, signed."""
    return f"{mean_delta * 100:+.1f}"


def _paragraph_with(text: str, needle: str) -> str:
    """The table row or prose paragraph holding the first `needle`."""
    at = text.index(needle)
    start = text.rfind("\n\n", 0, at)
    end = text.find("\n\n", at)
    return text[start + 2 if start >= 0 else 0: end if end >= 0 else len(text)]


def test_headline_string_matches_the_banked_quote():
    banked = json.loads((REPO / "experiments" / "novel_schema_summary_2026-08-12.json").read_text())
    assert headline_string(banked["gate_k30"]["mean_delta"]) == "+46.5"
    assert headline_string(-0.0049) == "-0.5" and headline_string(0.4) == "+40.0"


@pytest.mark.skipif(READING.exists(), reason="Addendum Z has landed; the landing test applies")
def test_before_landing_every_headline_quote_says_the_rerun_is_pending():
    for name in HEADLINE_PAGES:
        text = (REPO / name).read_text(encoding="utf-8")
        assert "+46.5" in text, name
        # the headline quote: the first row/paragraph carrying the number
        # together with its seed breakdown
        needle = "+36.0/+49.0/+54.4" if "+36.0/+49.0/+54.4" in text else "+36.0 / +49.0 / +54.4"
        block = _paragraph_with(text, needle)
        assert "+46.5" in block, name
        assert PENDING in block.lower() and "Addendum Z" in block, (name, block[:200])


@pytest.mark.skipif(not READING.exists(), reason="Addendum Z has not landed")
def test_after_landing_the_pages_obey_the_frozen_citation_rule():
    reading = json.loads(READING.read_text())
    headline = reading["headline"]
    word = reading["Z1_gate"]["reading"]
    comparable = reading["Z4_attrition"]["reading"] == "SAME EXCLUSIONS"
    for name in HEADLINE_PAGES:
        text = (REPO / name).read_text(encoding="utf-8")
        assert "Addendum Z" in text, name
        if not comparable:
            # nothing moves until the discrepancy is filed
            assert "NOT COMPARABLE" in text or "DIFFERENT EXCLUSIONS" in text, name
            continue
        if word != "GO":
            assert headline["quote"] is None
            assert "withdrawn" in text.lower(), name
            continue
        new = headline_string(headline["quote"])
        assert new in text, (name, new)
        # the banked number survives only as history beside the new one
        assert "+46.5" not in text or "banked" in text.lower(), name
    # CORRECTIONS.md: "a correction that moves a headline says so" — if
    # the quoted number changed at all, or the headline was withdrawn, the
    # ledger carries it
    corrections = (REPO / "CORRECTIONS.md").read_text(encoding="utf-8")
    if comparable and word == "GO" and headline_string(headline["quote"]) != "+46.5":
        assert headline_string(headline["quote"]) in corrections
    if comparable and word != "GO":
        assert "withdrawn" in corrections.lower() and "+46.5" in corrections
