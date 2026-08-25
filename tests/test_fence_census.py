"""The census classifier must be able to say "not exposed".

A classifier that returns EXPOSED for everything would produce a large,
meaningless and very shareable number. That is the most dangerous failure
available here, because the number would be repeated by people who cannot
check it — so the planted negative matters more than the planted positive.

The two synthetic packages below are the frozen mutation test from
`docs/research/FENCE_CENSUS_PREREGISTRATION.md`.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "fence_census.py"

pytestmark = pytest.mark.skipif(
    not SCRIPT.exists(), reason="the census is not in this cut")

_spec = importlib.util.spec_from_file_location("fence_census", SCRIPT)
census = importlib.util.module_from_spec(_spec)
sys.modules["fence_census"] = census
_spec.loader.exec_module(census)


EXPOSED_SOURCE = '''
import json

def score(completion, gold):
    """Parses the model's response with no fence handling and scores a
    failure as zero. This is the defect."""
    try:
        parsed = json.loads(completion)
    except json.JSONDecodeError:
        return 0.0
    return float(parsed == gold)
'''

CLEAN_SOURCE = '''
import json

def score(completion, gold):
    """Strips a markdown fence before parsing."""
    text = completion.strip()
    if text.startswith("```"):
        text = text.split("\\n", 1)[-1].rsplit("```", 1)[0]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return 0.0
    return float(parsed == gold)
'''

RAISES_SOURCE = '''
import json

def parse(response):
    """No fence handling, but the failure reaches the caller rather than
    becoming a score. Not exposed by the frozen definition."""
    return json.loads(response)
'''


def test_it_flags_the_planted_defect() -> None:
    sites = census.analyse_source(EXPOSED_SOURCE, "scorer.py")
    assert sites, "the parse call site was not found at all"
    assert any(s["exposed"] for s in sites), (
        "a scorer that parses a fenced payload with no handling and returns "
        "0.0 on failure was not flagged — the classifier cannot fire")


def test_it_clears_a_scorer_that_strips_the_fence() -> None:
    sites = census.analyse_source(CLEAN_SOURCE, "scorer.py")
    assert sites
    assert not any(s["exposed"] for s in sites), (
        "a scorer that strips the fence was flagged anyway — the classifier "
        "cannot say no, so its count would be meaningless")


def test_a_parser_that_raises_is_not_exposed() -> None:
    """The conjunction in the definition is load-bearing.

    Parsing without fence handling is not a defect. Scoring the failure
    is. A classifier that ignored the second half would indict every
    library that has ever called json.loads.
    """
    sites = census.analyse_source(RAISES_SOURCE, "parser.py")
    assert sites
    assert not any(s["exposed"] for s in sites)
    assert not any(s["zero_on_failure"] for s in sites)


def test_the_inclusion_test_excludes_unrelated_code() -> None:
    """A package that parses config files is not in the universe."""
    assert not census.MODEL_WORDS.search(
        "import json\ndef load_config(path):\n    return json.load(open(path))")
    assert census.MODEL_WORDS.search("parsed = json.loads(completion.text)")


def test_the_seed_list_is_frozen_and_substantial() -> None:
    """The frame's size is a preregistered commitment, not a dial.

    Widening the seed list after seeing a disappointing count is the
    obvious way to manufacture a better headline, so the floor is pinned
    in a test rather than left to discipline.
    """
    assert len(census.SEED_PACKAGES) >= 40
    assert len(set(census.SEED_PACKAGES)) == len(census.SEED_PACKAGES), (
        "a duplicate in the seed list would inflate the denominator")


def test_evidence_is_recorded_for_every_site() -> None:
    """A verdict a reader cannot check is an assertion, not a census."""
    for source in (EXPOSED_SOURCE, CLEAN_SOURCE, RAISES_SOURCE):
        for site in census.analyse_source(source, "x.py"):
            assert site["file"] and site["line"] > 0
            assert site["evidence"].strip(), "no source evidence banked"


ARTIFACT_READER_SOURCE = '''
import json, pathlib

def load_scores(path, model_name):
    """Reads a banked artifact off disk. Mentions a model, parses JSON,
    swallows failure -- and is not model output at all."""
    try:
        record = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return record.get(model_name, {})
'''


def test_reading_an_artifact_off_disk_is_not_model_output() -> None:
    """AMENDMENT, recorded before any package was inspected.

    Gating on the file was too coarse: a module that mentions a model and
    also loads a JSON config was indicted. Precision is what makes the
    census number survive a spot check, and a classifier that flags every
    `json.loads` in an ML repo produces a big number that falls apart on
    the first one.
    """
    sites = census.analyse_source(ARTIFACT_READER_SOURCE, "loader.py")
    assert sites, "the call site should still be recorded"
    assert all(s["reads_a_file"] for s in sites)
    assert not any(s["exposed"] for s in sites), (
        "an artifact reader was counted as exposed")


def test_the_amendment_does_not_clear_a_real_model_parse() -> None:
    """The narrowing must not swallow the case the census is about."""
    sites = census.analyse_source(EXPOSED_SOURCE, "scorer.py")
    assert any(s["exposed"] for s in sites)
    assert not any(s["reads_a_file"] for s in sites)
