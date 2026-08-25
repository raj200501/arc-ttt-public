"""Addendum I is the result that rescoped the company. Pin what it says.

What an outside reviewer found here, and what is now pinned so it cannot
come back:

1. A single hosted-API run is not a measurement. Temperature 0 is not
   deterministic on a hosted endpoint, and the first write-up published
   one run's mean as a fact. The summary must report a range.
2. micro-F1 folds letter case; canonical-JSON exact match does not. They
   disagree on this corpus, so "every document exactly right" is not a
   statement micro-F1 supports.
3. The packed-turn run's number depends on a fence-strip repair OUR OWN
   arms never received. Where the un-repaired mean differs from the
   headline, the headline is repair-dependent and must not be cited.
4. Per-document comparisons must be aligned by id, not by position.
5. The sign test must name its direction. Published beside 0W/14L, a
   p-value of 1.0 reads as "no difference" and means the opposite.
"""

from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"
SUMMARY = EXP / "waybill_market_summary_2026-08-22.json"
MATCHED = [EXP / "waybill_market_baseline_gemini-3.5-flash-lite"
                 "_matchedturns_run3_2026-08-22.json",
           EXP / "waybill_market_baseline_gemini-3.5-flash-lite"
                 "_matchedturns_run4_2026-08-22.json"]


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_headline_is_a_range_not_a_point():
    s = _load(SUMMARY)
    assert s["n_runs"] >= 3, "one or two runs cannot establish a range"
    low, high = s["hosted_range"]
    assert low < high, (
        "if every run returned the identical mean, say so explicitly; "
        "do not let a coincidence stand in for reproducibility")
    assert low < s["hosted_mean_of_run_means"] < high


def test_the_conclusion_is_stable_even_though_the_number_is_not():
    s = _load(SUMMARY)
    assert s["conclusion_stable_across_all_runs"] is True
    assert max(s["paired_delta_range"]) < 0, (
        "our arm must lose in every run for the published conclusion to "
        "hold; if any run flips, the rescoping must be revisited")


def test_exact_match_is_reported_and_is_not_the_same_as_micro_f1():
    for path in MATCHED:
        record = _load(path)
        exact, total = record["hosted_exact_match"].split("/")
        assert int(exact) < int(total), (
            "if exact match ever equals micro-F1 on this corpus, the "
            "casing documents changed and the prose must be re-checked")
        assert record["hosted_mean_micro_f1"] > int(exact) / int(total)


def test_the_matched_runs_need_no_output_repair():
    """The citable runs must not depend on a repair our own arms lacked."""
    for path in MATCHED:
        record = _load(path)
        assert record["fence_stripped_documents"] == []
        assert (record["mean_without_fence_strip"]
                == record["hosted_mean_micro_f1"])


def test_sign_tests_name_their_direction():
    for path in MATCHED:
        st = _load(path)["paired_our_adapted_minus_hosted"]["sign_test"]
        assert st["observed_direction"] == "theirs > ours"
        assert st["p_value_in_observed_direction"] < 0.01
        # The naive tail is ~1.0 here; publishing it alone was the bug.
        assert st["p_value_ours_greater"] > 0.9


def test_per_document_comparisons_are_aligned_by_id():
    s = _load(SUMMARY)
    per_doc = [set(r["per_doc"]) for r in s["runs"]]
    assert all(ids == per_doc[0] for ids in per_doc), (
        "runs cover different document sets; a positional comparison "
        "would silently compare different documents")
    assert len(per_doc[0]) == 30


def test_the_document_that_moves_is_named():
    s = _load(SUMMARY)
    assert s["documents_that_move_between_runs"], (
        "if no document moves, the range in the summary is unexplained")


def test_runner_refuses_a_key_file_inside_the_repository():
    source = (REPO / "scripts"
              / "run_market_baseline_waybills.py").read_text("utf-8")
    assert "refusing: the key file is inside the repository" in source
    # A missing completion (safety block, MAX_TOKENS) must raise, not be
    # scored as a model zero -- the runner's own docstring promises this and
    # an earlier version returned "" instead.
    assert "score this as a model zero" in source
    assert "raise RuntimeError(" in source
    assert 'return ""' not in source
