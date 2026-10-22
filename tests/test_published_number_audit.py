"""Addendum AD's runner (scripts/published_number_audit.py), checked on
synthetic answers and synthetic tables only.

The held-out rule: no LiveBench answer, judgment or published table value
is read here, before or after the freeze. The only LiveBench bytes these
tests touch are the vendored scorer code under scripts/ad_pinned/, which
the protocol pins and the scoping run already executed on synthetic input.
The stage tests serve synthetic files from file:// URLs; the ones that
write or read a parquet need pyarrow and skip without it.
"""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from datetime import datetime

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ad", REPO / "scripts" / "published_number_audit.py")
ad = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ad)
fp_spec = importlib.util.spec_from_file_location("fp_for_ad", REPO / "scripts" / "freeze_protocol.py")
fp = importlib.util.module_from_spec(fp_spec)
fp_spec.loader.exec_module(fp)

MANIFEST = ad.load_manifest()
FENCECHECK = ad.load_fencecheck()


def _utc(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -- what is pinned ----------------------------------------------------------------


def test_vendored_files_are_the_manifest_files_and_nothing_else():
    pinned = MANIFEST["pinned_files"]
    on_disk = {f"scripts/ad_pinned/{p.name}" for p in (REPO / "scripts" / "ad_pinned").glob("*.py")}
    assert on_disk == set(pinned), "a vendored file is unpinned, or a pinned file is missing"
    for rel, digest in pinned.items():
        assert _sha((REPO / rel).read_bytes()) == digest, rel
    lic = MANIFEST["license"]
    assert _sha((REPO / lic["file"]).read_bytes()) == lic["sha256"]
    assert (REPO / "scripts" / "ad_pinned" / "README.md").exists(), "provenance note missing"


def test_the_versions_are_the_first_parent_states_deduplicated():
    assert ad.version_consistency(MANIFEST) == []
    ids = [v["id"] for v in MANIFEST["livebench"]["versions"]]
    assert ids == ["905b71f", "f774d19", "19c7757", "c5c293c", "fa6a3c6", "f3501c7", "e647d72"]
    # the owner's fix is the last version, and nothing after the window is pinned
    assert MANIFEST["livebench"]["versions"][-1]["commit"].startswith("e647d72")
    assert _utc(MANIFEST["livebench"]["window_closes_utc"]) > _utc(
        MANIFEST["site"]["tables"]["2025_04_02"]["last_commit"]["committer_time_utc"])


def test_version_consistency_finds_a_version_list_that_is_not_the_states():
    broken = json.loads(json.dumps(MANIFEST))
    broken["livebench"]["versions"] = broken["livebench"]["versions"][1:]
    assert any("not the de-duplicated states" in p for p in ad.version_consistency(broken))
    broken = json.loads(json.dumps(MANIFEST))
    broken["livebench"]["versions"][2]["in_force_from_utc"] = "2024-12-18T00:00:00Z"
    assert any("landing on main" in p for p in ad.version_consistency(broken))
    broken = json.loads(json.dumps(MANIFEST))
    v = broken["livebench"]["versions"][0]
    broken["pinned_files"][v["tablejoin_utils"]] = "0" * 64
    assert any("tablejoin utils file is not the state's" in p for p in ad.version_consistency(broken))


def test_every_remote_file_is_named_with_a_size_and_a_digest_or_a_reason():
    ids = [e["id"] for e in MANIFEST["remote"]]
    assert len(ids) == len(set(ids)) == 73
    for entry in MANIFEST["remote"]:
        assert entry["url"].startswith("https://"), entry["id"]
        assert entry["on_mismatch"] in ("refuse", "record"), entry["id"]
        assert entry.get("bytes"), entry["id"]
        if entry["on_mismatch"] == "refuse":
            assert entry.get("sha256") or entry.get("git_blob_sha1"), entry["id"]
        assert entry.get("sha256_source"), entry["id"]
    by_id = {e["id"]: e for e in MANIFEST["remote"]}
    assert by_id["answers"]["sha256"].startswith("1d4b5547")
    assert by_id["questions"]["sha256"].startswith("fb86a7a0")
    for date, meta in MANIFEST["site"]["tables"].items():
        assert by_id[meta["live_id"]]["expect_equal_to"] == meta["pinned_id"], date
        assert by_id[meta["pinned_id"]]["bytes"] == by_id[meta["live_id"]]["bytes"], date
        # landing is committer time on both sides of the in-force rule
        assert meta["last_commit"]["committer_time_utc"], date


def test_the_manifest_does_not_pin_fencecheck_a_second_time():
    """The protocol's digest line is the one pin of tools/fencecheck.py; a
    second digest in the manifest went stale once and disagreed with it."""
    assert MANIFEST["fencecheck"]["path"] == "tools/fencecheck.py"
    assert MANIFEST["fencecheck"].get("sha256") is None
    assert "tools/fencecheck.py" in ad.required_files(MANIFEST)


def test_fencecheck_scan_on_the_vendored_scorers_is_what_the_protocol_says():
    """The protocol records what the pinned scanner reports on the five
    vendored tablejoin scorers. If tools/fencecheck.py changes what it
    reports, this fails before a freeze could anchor a false sentence."""
    flagged = {}
    for vid in ("905b71f", "f774d19", "19c7757", "c5c293c", "e647d72"):
        path = REPO / "scripts" / "ad_pinned" / f"tablejoin_utils_{vid}.py"
        run = subprocess.run([sys.executable, "-I", "-B", str(REPO / "tools" / "fencecheck.py"), "scan",
                              str(path)], capture_output=True, text=True, cwd=REPO)
        flagged[vid] = (run.returncode, "literal_eval" in run.stdout)
    assert flagged == {"905b71f": (1, True), "f774d19": (1, True), "19c7757": (1, True),
                       "c5c293c": (1, True), "e647d72": (0, False)}
    text = " ".join(ad.PROTOCOL_PATH.read_text(encoding="utf-8").split())
    assert "flags four of the five vendored `tablejoin` scorer versions and passes the fifth" in text
    assert 'On `e647d72`, the owner\'s fix, it prints "1 file(s) scanned, nothing exposed."' in text


# -- LiveBench's code, run as shipped -------------------------------------------------


def test_the_extraction_is_lifted_from_each_vendored_file_and_think_removal_starts_at_fa6a3c6():
    answer = {"choices": [{"index": 0, "turns": ["first", "<think>x</think>```json\n{}\n```"]}]}
    for v in MANIFEST["livebench"]["versions"]:
        source = (REPO / v["gen_ground_truth_judgment"]).read_text(encoding="utf-8")
        assert ad.extraction_sha256(source) == v["extraction_sha256"], v["id"]
        text = ad.compile_extraction(source)(answer)
        removes = v["id"] in ("fa6a3c6", "f3501c7", "e647d72")
        assert text == ("```json\n{}\n```" if removes else answer["choices"][0]["turns"][-1]), v["id"]


def test_g1_every_pinned_version_repeats_the_scoping_synthetic_check():
    pytest.importorskip("numpy")
    versions = ad.load_versions(MANIFEST)
    expected = json.loads(ad.SYNTHETIC_PATH.read_text())
    g1 = ad.synthetic_check(versions, expected, FENCECHECK)
    assert g1["passed"], g1["mismatches"]
    r = g1["results"]
    # the defect, executed: a correct json-fenced answer scores 0 until the fix
    for vid in ("905b71f", "f774d19", "19c7757", "c5c293c", "fa6a3c6", "f3501c7"):
        assert r[vid]["fence_json"] == {"O": 0.0, "F": 1.0}, vid
    assert r["e647d72"]["fence_json"] == {"O": 1.0, "F": 1.0}
    assert r["905b71f"]["fence_bare"] == {"O": 0.0, "F": 1.0}
    assert r["f774d19"]["fence_bare"] == {"O": 1.0, "F": 1.0}
    # the leading strip does not reach a fence after prose
    assert r["905b71f"]["prose_then_fence_json"] == {"O": 0.0, "F": 0.0}


def test_g1_fails_on_a_changed_expectation():
    pytest.importorskip("numpy")
    versions = ad.load_versions(MANIFEST)[:1]
    expected = json.loads(ad.SYNTHETIC_PATH.read_text())
    expected["905b71f"]["fence_json"]["O"] = 1.0
    g1 = ad.synthetic_check(versions, expected, FENCECHECK)
    assert not g1["passed"] and g1["mismatches"][0]["case"] == "fence_json"


def test_a_second_fence_after_the_first_can_make_the_strip_score_lower():
    """The strip keeps everything up to the last fence, so an answer that
    corrects itself in a second fence can score lower under F than under
    LiveBench's own regex. That is the strip's error, which is why a
    downward F - O never counts as the fence failure moving a number."""
    pytest.importorskip("numpy")
    versions = {v.id: v for v in ad.load_versions(MANIFEST)}
    gt = json.dumps({"colA": "x1", "colB": "y2"})
    text = ("```python\n{'colA': 'x9'}\n```\nWait, corrected:\n"
            "```python\n{'colA': 'x1', 'colB': 'y2'}\n```")
    row = ad.score_one(versions["19c7757"], {"turns": ["q"], "ground_truth": gt},
                       {"choices": [{"index": 0, "turns": [text]}]}, FENCECHECK)
    assert (row["O"], row["F"]) == (1.0, 0.0)


class _Recorder:
    """A stand-in version that records the exact text its scorer receives."""

    id = "rec"

    def __init__(self):
        self.seen = []

    @staticmethod
    def extract(answer):
        return answer["choices"][0]["turns"][-1]

    def score(self, question_text, ground_truth, text):
        self.seen.append(text)
        return 1.0 if text.strip().startswith("{") else 0.0


def _answer(text):
    return {"choices": [{"index": 0, "turns": [text]}]}


def test_f_is_exactly_one_change_and_only_on_a_fenced_answer():
    q = {"turns": ["q"], "ground_truth": "{}"}
    rec = _Recorder()
    unfenced = "  {'a': 'b'}\n"
    row = ad.score_one(rec, q, _answer(unfenced), FENCECHECK)
    # the scorer sees the original bytes once: strip_fence's own .strip() is
    # not a second change, and F equals O by construction
    assert rec.seen == [unfenced]
    assert row["O"] == row["F"] == 1.0 and row["fence"] == "none"
    rec = _Recorder()
    row = ad.score_one(rec, q, _answer("```json\n{\"a\": 1}\n```"), FENCECHECK)
    assert rec.seen[0] == "```json\n{\"a\": 1}\n```" and rec.seen[1] == "{\"a\": 1}"
    assert (row["O"], row["F"], row["fence"]) == (0.0, 1.0, "json")
    rec = _Recorder()
    row = ad.score_one(rec, q, _answer("Here:\n```json\n{\"a\": 1}\n```"), FENCECHECK)
    assert (row["O"], row["F"], row["A"]) == (0.0, 0.0, 1.0)  # A is the wider scope, counts only


def test_the_wider_scope_never_finds_a_single_quoted_python_dict_in_prose():
    """AD-S2's arm A: strict JSON only, as the protocol says."""
    assert FENCECHECK.extract_json("Here is the mapping: {'colA': 'x1'}", scope="any")[1] == "none"
    assert FENCECHECK.extract_json('Here is the mapping: {"colA": "x1"}', scope="any")[1] == \
        "bare_object_in_prose"


def test_a_raising_extraction_or_scorer_is_recorded_not_hidden():
    class Broken(_Recorder):
        @staticmethod
        def extract(answer):
            raise KeyError("choices")

    row = ad.score_one(Broken(), {"turns": ["q"], "ground_truth": "{}"}, {}, FENCECHECK)
    assert row["O"].startswith("error:") and row["F"].startswith("error:")


@pytest.mark.parametrize("text,klass", [
    ("```python\n{}\n```", "python"), ("```json\n{}\n```", "json"), ("```\n{}\n```", "empty"),
    ("```Python\n{}\n```", "other"), ("```JSON\n{}\n```", "other"), ("```{}```", "one_line"),
    ("{}", "none"), ("Sure:\n```json\n{}\n```", "none"), ("  \n```python\n{}\n```", "python"),
])
def test_fence_class_is_exact_case(text, klass):
    assert ad.fence_class(text)[0] == klass


# -- the frame ---------------------------------------------------------------------------


def test_models_match_exactly_or_case_insensitively_and_aliases_are_never_guessed():
    ids = ["gpt-4o-2024-05-13", "Meta-Llama-3-70B", "a/b-local", "X-1", "x-1"]
    m = ad.match_models(["gpt-4o-2024-05-13", "meta-llama-3-70b", "a/b-local", "gpt-4o", "x-1",
                         "X-1", "Unknown"], ids)
    assert m["gpt-4o-2024-05-13"] == {"model_id": "gpt-4o-2024-05-13", "match": "exact"}
    assert m["meta-llama-3-70b"] == {"model_id": "Meta-Llama-3-70B", "match": "case-insensitive"}
    assert m["a/b-local"]["model_id"] is None  # local paths are out
    assert m["gpt-4o"]["model_id"] is None
    assert m["x-1"]["match"] == "exact" and m["X-1"]["match"] == "exact"
    assert ad.match_models(["X-ONE"], ["x-one", "X-One"])["X-ONE"]["match"] == "ambiguous"


def test_in_force_sets_follow_the_manifest_times():
    versions = MANIFEST["livebench"]["versions"]
    closes = _utc(MANIFEST["livebench"]["window_closes_utc"])
    tables = MANIFEST["site"]["tables"]
    answers = _utc("2024-07-01T00:00:00Z")
    upper = _utc(tables["2024_06_24"]["last_commit"]["committer_time_utc"])
    assert ad.in_force(versions, answers, upper, closes) == ["905b71f"]
    upper = _utc(tables["2024_08_31"]["last_commit"]["committer_time_utc"])
    assert ad.in_force(versions, answers, upper, closes) == ["905b71f", "f774d19"]
    upper = _utc(tables["2025_04_02"]["last_commit"]["committer_time_utc"])
    assert ad.in_force(versions, _utc("2024-10-17T20:22:17Z"), upper, closes) == [
        v["id"] for v in versions]
    # answers newer than the table: no version could have scored them
    assert ad.in_force(versions, _utc("2024-09-10T00:00:00Z"),
                       _utc(tables["2024_07_26"]["last_commit"]["committer_time_utc"]), closes) == []
    # past the pinned window nothing is in force
    assert ad.in_force(versions, _utc("2025-05-01T00:00:00Z"), _utc("2026-01-01T00:00:00Z"), closes) == []


def test_the_new_lower_bound_can_leave_no_version_where_the_drafts_kept_one():
    """Correction 3 says so: for the first two tables, answers newer than
    the table's last commit leave no version in force, while the draft's
    lower bound (the table's first commit) kept 905b71f."""
    versions = MANIFEST["livebench"]["versions"]
    closes = _utc(MANIFEST["livebench"]["window_closes_utc"])
    for date in ("2024_06_24", "2024_07_26"):
        meta = MANIFEST["site"]["tables"][date]
        upper = _utc(meta["last_commit"]["committer_time_utc"])
        first = _utc(meta["first_commit"]["author_time_utc"])
        late = _utc("2024-10-17T20:22:17Z")
        assert ad.in_force(versions, late, upper, closes) == []
        assert ad.in_force(versions, first, upper, closes) == ["905b71f"]


def test_a_tables_upper_bound_is_its_last_commits_committer_time(tmp_path):
    table, cats = tmp_path / "t.csv", tmp_path / "c.json"
    table.write_text(_table([["m", "1", "50.000", "1"]]))
    cats.write_text(json.dumps(CATS))
    got = {"bytes": 1, "sha256": "x"}
    fetch = {"remote": [{"id": "p", "path": str(table), "got": got, "matches": True},
                        {"id": "l", "path": str(table), "got": got, "matches": True,
                         "received": {"fetched_utc": "2026-11-01T00:00:00Z"}},
                        {"id": "c", "path": str(cats), "got": got, "matches": True}]}
    manifest = {"site": {"tables": {"t": {
        "pinned_id": "p", "live_id": "l", "categories_id": "c",
        "first_commit": {"author_time_utc": "2024-08-03T04:30:35Z"},
        "last_commit": {"author_time_utc": "2024-09-01T00:00:00Z",
                        "committer_time_utc": "2024-09-07T04:35:50Z"}}}}}
    t = ad._tables_from(fetch, manifest)["t"]
    assert t["upper_utc"] == "2024-09-07T04:35:50Z" and t["first_utc"] == "2024-08-03T04:30:35Z"
    # a live table that differs from its pinned twin: the fetch time bounds it
    fetch["remote"][1]["matches"] = False
    assert ad._tables_from(fetch, manifest)["t"]["upper_utc"] == "2026-11-01T00:00:00Z"


def test_column_tolerance_uses_the_most_decimals_capped_at_three():
    assert ad.column_tolerance(["45.5", "46.68", "-", ""]) == (2, 0.005)
    assert ad.column_tolerance(["45.5", "46.667"]) == (3, 0.0005)
    assert ad.column_tolerance(["45.12345"]) == (3, 0.0005)
    assert ad.column_tolerance(["50.0", "48.0"]) == (1, 0.05)
    assert ad.column_tolerance(["46", "47"]) == (0, 0.5)
    assert ad.column_tolerance(["-", ""]) == (None, None)


# -- cells, synthetic ---------------------------------------------------------------------

VERS = [{"id": "A", "in_force_from_utc": "2024-06-01T00:00:00Z"},
        {"id": "B", "in_force_from_utc": "2024-10-01T00:00:00Z"}]
CATS = {"Data Analysis": ["cta", "tablejoin"], "Other": ["x"]}


def _answers(model, scores_a, scores_b, fa=None, fb=None, tstamp=1719792000.0):
    """One synthetic answer per score; fa/fb are the F values (default: O)."""
    out = []
    for i, (oa, ob) in enumerate(zip(scores_a, scores_b)):
        out.append({"model_id": model, "question_id": f"q{i}", "tstamp": tstamp, "v": {
            "A": {"O": oa, "F": (fa or scores_a)[i], "A": (fa or scores_a)[i], "fence": "python"},
            "B": {"O": ob, "F": (fb or scores_b)[i], "A": (fb or scores_b)[i], "fence": "json"}}})
    return out


def _table(rows):
    lines = ["model,cta,tablejoin,x"] + [",".join(r) for r in rows]
    return "\n".join(lines) + "\n"


def _synthetic():
    per_answer = (
        _answers("m-one", [1.0, 0.0], [1.0, 1.0], fa=[1.0, 1.0])               # A reproduces 50.0
        + _answers("m-two", [1.0, 0.0], [1.0, 0.0], fa=[1.0, 1.0], fb=[1.0, 0.0])  # both reproduce
        + _answers("m-three", [0.0, 0.0], [0.0, 0.0])                            # published 80: no
        + _answers("m-late", [1.0, 1.0], [1.0, 1.0], tstamp=1735689600.0)        # answers 2025-01-01
    )
    tables = {"2024_09_30": {"text": _table([
        ["m-one", "10", "50.000", "20"], ["M-Two", "10", "50.0", "30"], ["m-three", "10", "80", "40"],
        ["m-late", "10", "100", "50"], ["no-such-model", "10", "70", "60"], ["m-one", "10", "", "1"],
    ]), "categories": CATS, "upper_utc": "2024-12-01T00:00:00Z", "first_utc": "2024-09-30T00:00:00Z"}}
    return per_answer, tables


def test_cells_reproduce_from_o_or_are_excluded_and_counted():
    per_answer, tables = _synthetic()
    built = ad.build_cells(tables, per_answer, VERS, ["m-one", "m-two", "m-three", "m-late"])
    cells = {(c["model"], c["row"]): c for c in built["cells"]}
    one = cells[("m-one", 0)]
    assert one["in_force"] == ["A", "B"] and one["status"] == ad.REPRODUCED
    assert one["reproduced_by"] == ["A"] and one["delta"] == pytest.approx(50.0)
    assert one["answers_up"] == 1 and one["answers_down"] == 0
    two = cells[("M-Two", 1)]
    assert two["match"] == "case-insensitive" and two["reproduced_by"] == ["A", "B"]
    # several versions reproduce and disagree: the smallest F - O is the claim's
    assert two["chosen_version"] == "B" and two["delta"] == 0.0
    assert two["delta_max_version"] == "A" and two["delta_max"] == pytest.approx(50.0)
    assert two["deltas"] == {"A": pytest.approx(50.0), "B": 0.0}
    three = cells[("m-three", 2)]
    assert three["status"] == ad.NOT_REPRODUCED and set(three["per_version"]) == {"A", "B"}
    late = cells[("m-late", 3)]
    assert late["status"] == ad.POSTDATES
    # the draft's lower bound (the table's first commit) would have kept it
    assert late["in_force_from_the_tables_first_commit"] == ["A", "B"]
    frame = built["frame"]["2024_09_30"]
    assert frame["postdates_cells_the_draft_lower_bound_kept"] == 1
    assert frame["unmatched"] == ["no-such-model"] and frame["no_published_value"] == ["m-one"]
    assert frame["decimals"] == 3 and frame["tolerance_points"] == 0.0005


def test_an_answer_that_raises_under_o_makes_that_version_unable_to_reproduce():
    per_answer = _answers("m", ["error: AssertionError", 1.0], [1.0, 1.0])
    tables = {"t": {"text": _table([["m", "1", "100", "1"]]), "categories": CATS,
                    "upper_utc": "2024-12-01T00:00:00Z"}}
    cell = ad.build_cells(tables, per_answer, VERS, ["m"])["cells"][0]
    assert cell["per_version"]["A"]["O"] is None and not cell["per_version"]["A"]["reproduces"]
    assert cell["per_version"]["B"]["reproduces"] and cell["chosen_version"] == "B"


# -- the readings ---------------------------------------------------------------------------


def _score_record(per_answer, tables, versions=VERS):
    built = ad.build_cells(tables, per_answer, versions, sorted({r["model_id"] for r in per_answer}))
    return {"g1": {"passed": True},
            "g2": {"texts_equal": True, "tags_match_disclosure": True,
                   "page_matches_scoping_digest": True},
            "versions": versions, "table_order": list(tables), "frame": built["frame"],
            "cells": built["cells"], "per_answer": per_answer, "ad_s3": None}


def _manifest_for(n_cells):
    return {"frame": {"scoping_counts": {"in_frame_cells": n_cells}},
            "livebench": {"versions": [{"id": "B"}]}}


FETCH_OK = {"g0": {"passed": True, "recorded_differences_published_first": []}}


def _one_cell_table(rows):
    return {"t": {"text": _table(rows), "categories": CATS, "upper_utc": "2024-12-01T00:00:00Z"}}


def _read(per_answer, tables, n_cells=None):
    score = _score_record(per_answer, tables)
    n = len(score["cells"]) if n_cells is None else n_cells
    return score, ad.reading_from(FETCH_OK, score, tables, _manifest_for(n))


def _models(n_moved, n_still, delta_up=True):
    """n_moved models whose one cell moves by a full question (of two), n_still that do not."""
    per_answer, rows = [], []
    for i in range(n_moved):
        f = [1.0, 1.0] if delta_up else [0.0, 0.0]
        o = [1.0, 0.0]
        per_answer += _answers(f"moved{i}", o, o, fa=f, fb=f)
        rows.append([f"moved{i}", "10", "50.000", str(10 + i)])
    for i in range(n_still):
        per_answer += _answers(f"still{i}", [1.0, 1.0], [1.0, 1.0])
        rows.append([f"still{i}", "10", "100.000", str(50 + i)])
    return per_answer, _one_cell_table(rows)


@pytest.mark.parametrize("n_moved,words", [(0, "NULL"), (1, "MOVES A NUMBER"),
                                           (2, "MOVES A NUMBER"), (3, "MOVES NUMBERS")])
def test_ad2_words_are_the_whole_vocabulary(n_moved, words):
    per_answer, tables = _models(n_moved, 3)
    _, reading = _read(per_answer, tables)
    assert reading["ad2"]["words"] == words == reading["words"]
    sentence = reading["licensed_sentences"]["ad2"]
    reach = f"{3 + n_moved} of the {3 + n_moved} in-frame cells reproduced"
    assert reach in sentence and "0 of the" in sentence
    if words == "NULL":
        assert sentence.startswith("Re-scoring LiveBench's stored 2024 `tablejoin` answers")
        assert f"in {3 + n_moved} reproduced cells over {3 + n_moved} models" in sentence
        # every delta is zero under every reproducing version and no rank moved:
        # only then may it say the failure changed nothing in these cells
        assert sentence.endswith(f"in these {3 + n_moved} cells it changed no published number and no rank.")
    elif words == "MOVES A NUMBER":
        assert "`moved0`" in sentence and "moved a published evaluation number" in sentence
    else:
        assert sentence.startswith("The fence failure moved published evaluation numbers")
        assert "3 models" in sentence
    assert "did not reach the leaderboard" not in sentence


def test_a_null_with_a_small_move_and_a_rank_change_says_both():
    """One partial answer rises from F1 0.5 to 1.0 (+1.0 point over 50):
    no cell moves by a question, yet a rank changes. The sentence must say
    so, and the README replacement carries it."""
    o = [0.5] + [1.0] * 49
    f = [1.0] * 50
    per_answer = (_answers("near", o, o, fa=f, fb=f)
                  + _answers("rival", [1.0] * 49 + [0.75], [1.0] * 49 + [0.75]))
    near = sum(x * 100 for x in o) / 50
    rival = (49 * 100 + 75) / 50
    tables = _one_cell_table([["near", "10", f"{near:.3f}", "1"], ["rival", "10", f"{rival:.3f}", "1"]])
    _, reading = _read(per_answer, tables)
    assert reading["ad2"]["words"] == "NULL" and reading["ad3"]["words"] == "RANK CHANGED"
    s2 = reading["licensed_sentences"]["ad2"]
    assert "1 reproduced cell(s) moved up by less than that (largest 1.00 points)" in s2
    assert "Ranks still changed: t: 2 model(s); largest move 1" in s2
    assert "changed no published number" not in s2 and "did not reach the leaderboard" not in s2
    assert reading["licensed_sentences"]["readme_replacement"] == s2
    assert reading["predictions"]["AD2_NULL"]["held"] is True
    assert reading["predictions"]["AD3_NO_RANK_CHANGED"]["held"] is False
    assert any("readme_replacement" in o for o in reading["obligations_the_same_day"])


def test_a_downward_move_is_the_strips_error_and_never_counts_toward_the_words():
    for n_down in (1, 3):
        per_answer, tables = _models(n_down, 2, delta_up=False)
        _, reading = _read(per_answer, tables)
        assert reading["ad2"]["words"] == "NULL"
        assert reading["ad2"]["moved_down_cells"] == n_down and reading["ad2"]["moved_up_cells"] == 0
        s2 = reading["licensed_sentences"]["ad2"]
        assert f"The strip itself lowered {n_down} reproduced cell(s), {n_down} of them" in s2
        assert "fencecheck's own error, not LiveBench's" in s2
        assert "fence failure moved" not in s2
        assert reading["licensed_sentences"]["directions"].startswith("0 cell(s) moved up")
        # a lowered cell is never added in AD3: the fence failure did not lower it
        assert reading["ad3"]["words"] == "NO RANK CHANGED"
        # the prediction was written when a move either way counted: under
        # that rule this is a move, so the change of rule does not rescue it
        assert reading["predictions"]["AD2_NULL"]["words_as_drafted"] != "NULL"
        assert reading["predictions"]["AD2_NULL"]["held"] is False


def test_a_lowered_cell_is_never_added_in_ad3_even_where_it_would_move_a_rank():
    """The strip lowers `down` by a question; added in full it would drop
    below `other`. The claim adds only upward F - O, so no rank moves;
    the drafted rule, kept for the prediction, adds it and sees one."""
    per_answer = (_answers("down", [1.0, 1.0], [1.0, 1.0], fa=[1.0, 0.0], fb=[1.0, 0.0])
                  + _answers("other", [1.0, 1.0], [1.0, 1.0]))
    tables = _one_cell_table([["down", "10", "100.000", "5"], ["other", "10", "100.000", "4"]])
    _, reading = _read(per_answer, tables)
    assert reading["ad3"]["words"] == "NO RANK CHANGED"
    assert reading["ad3_largest_delta"]["words"] == "NO RANK CHANGED"
    assert reading["predictions"]["AD3_NO_RANK_CHANGED"]["words_as_drafted"] == "RANK CHANGED"
    assert reading["predictions"]["AD3_NO_RANK_CHANGED"]["held"] is False


def test_mixed_signs_across_reproducing_versions_never_make_a_positive():
    """Under A the strip lowers the cell by a question; under B nothing
    moves. Both reproduce. That is not a move of the fence failure."""
    per_answer = (_answers("m", [1.0, 0.0], [1.0, 0.0], fa=[0.0, 0.0], fb=[1.0, 0.0])
                  + _answers("s", [1.0, 1.0], [1.0, 1.0]))
    tables = _one_cell_table([["m", "10", "50.000", "1"], ["s", "10", "100.000", "1"]])
    score, reading = _read(per_answer, tables)
    cell = next(c for c in score["cells"] if c["model"] == "m")
    assert cell["reproduced_by"] == ["A", "B"] and cell["delta"] == pytest.approx(-50.0)
    assert reading["ad2"]["words"] == "NULL" and reading["ad2_largest_delta"]["words"] == "NULL"
    assert reading["ad2"]["moved_down_cells"] == 1
    assert reading["predictions"]["AD2_NULL"]["held"] is False  # as drafted, it moved


def test_the_predictions_use_the_largest_delta_and_the_null_sentence_says_what_it_would_show():
    """A json+null answer: under A the strip does not help (no null
    replacement), under B it does. Both versions reproduce O. The claim
    reads A (smallest); the prediction is checked on B (largest)."""
    o = [0.0, 0.0, 0.0, 1.0]
    per_answer = (_answers("j", o, o, fa=o, fb=[1.0, 0.0, 0.0, 1.0])
                  + _answers("s", [1.0] * 4, [1.0] * 4))
    tables = _one_cell_table([["j", "10", "25.000", "1"], ["s", "10", "100.000", "1"]])
    _, reading = _read(per_answer, tables)
    assert reading["ad2"]["words"] == "NULL"
    assert reading["ad2_largest_delta"]["words"] == "MOVES A NUMBER"
    assert reading["predictions"]["AD2_NULL"] == {
        "words_smallest": "NULL", "words_largest": "MOVES A NUMBER", "words_as_drafted": "NULL",
        "held": False}
    s2 = reading["licensed_sentences"]["ad2"]
    assert ("Under another scorer version that also reproduces them, 1 cell(s) over 1 model(s) "
            "would move up by a question's worth or more.") in s2
    assert reading["words_largest_delta"] == "MOVES A NUMBER"


def test_less_than_a_question_is_not_a_move_but_is_said():
    # 50 answers: one answer from 0.01 to 1.0 moves the cell by 1.98 points
    o = [0.01] + [1.0] * 49
    f = [1.0] + [1.0] * 49
    per_answer = _answers("near", o, o, fa=f, fb=f)
    cell = sum(x * 100 for x in o) / 50
    tables = _one_cell_table([["near", "10", f"{cell:.3f}", "1"]])
    score, reading = _read(per_answer, tables, 1)
    assert score["cells"][0]["status"] == ad.REPRODUCED
    assert score["cells"][0]["delta"] == pytest.approx(1.98)
    assert reading["ad2"]["words"] == "NULL"
    assert "(largest 1.98 points)" in reading["licensed_sentences"]["ad2"]


def test_g3_withholds_ad2_and_ad3_when_fewer_than_half_reproduce():
    per_answer, tables = _models(1, 0)
    per_answer += _answers("lost1", [1.0, 1.0], [1.0, 1.0]) + _answers("lost2", [1.0, 1.0], [1.0, 1.0])
    text = tables["t"]["text"] + "lost1,10,12.5,1\nlost2,10,12.5,1\n"
    tables["t"]["text"] = text
    _, reading = _read(per_answer, tables, 3)
    assert reading["g3"]["withholds"] and reading["g3"]["by_status"][ad.NOT_REPRODUCED] == 2
    assert reading["ad2"]["words"] == reading["ad3"]["words"] == "WITHHELD (NOT REPRODUCED)"
    assert reading["licensed_sentences"]["ad2"] == (
        "We could not reproduce LiveBench's published `tablejoin` numbers from its stored "
        "answers, so nothing is said about them.")
    assert reading["licensed_sentences"]["readme_replacement"] is None
    assert reading["predictions"]["AD2_NULL"]["held"] is None
    assert "the sentence stays" in reading["obligations_the_same_day"][0]
    # AD1 still publishes: it has no bar
    assert reading["ad1"]["totals"]["answers"] == 6


def test_a_count_that_differs_from_scoping_publishes_first():
    per_answer, tables = _models(0, 2)
    _, reading = _read(per_answer, tables, 201)
    assert any("scoping counted 201" in line for line in reading["published_first"])


def test_g1_failure_is_the_result():
    reading = ad.reading_from(FETCH_OK, {"g1": {"passed": False, "mismatches": [1]}}, {},
                              _manifest_for(0))
    assert reading["words"] == "G1 FAILED" and "nothing was scored" in reading["licensed_sentences"]["ad2"]


def test_a_live_table_and_the_scoping_page_publish_first_in_their_own_words():
    fetch = {"g0": {"passed": True,
                    "recorded_differences_published_first": ["table_t_live", "feasibility_page"],
                    "recorded_roles": {"table_t_live": "table_live", "feasibility_page": "feasibility"}}}
    per_answer, tables = _models(0, 2)
    score = _score_record(per_answer, tables)
    first = ad.reading_from(fetch, score, tables, _manifest_for(2))["published_first"]
    assert first[0].startswith("table_t_live: the live file served at fetch differs")
    assert first[1].startswith("feasibility_page: the scoping page served at fetch is not the page")
    assert "published number" not in first[1]


def test_ad1_counts_each_answer_once_under_the_earliest_tables_version():
    per_answer, tables = _models(1, 1)
    later = {"u": {"text": _table([["moved0", "10", "50.000", "1"]]), "categories": CATS,
                   "upper_utc": "2025-01-01T00:00:00Z"}}
    tables.update(later)
    score = _score_record(per_answer, tables)
    record = ad.ad1(score)
    assert record["totals"]["answers"] == 4  # two models x two questions, once each
    assert record["reading_version_by_model"]["moved0"]["table"] == "t"
    assert record["totals"]["up"] == 1 and record["totals"]["full_flips"] == 1
    assert record["reading_version_by_model"]["moved0"]["version"] == "A"  # tie: earliest
    assert record["by_opening_fence"]["python"]["up"] == 1  # version A's synthetic fence class


def test_ad1_counts_a_model_whose_every_cell_postdates_its_table():
    """Its answers are in frame: they are counted under the version on main
    when its newest answer was made, and the model is listed."""
    per_answer = (_answers("late", [0.0, 1.0], [0.0, 1.0], fa=[1.0, 1.0], fb=[1.0, 1.0],
                           tstamp=1735689600.0)   # 2025-01-01: after the table's upper bound
                  + _answers("s", [1.0, 1.0], [1.0, 1.0]))
    tables = _one_cell_table([["late", "10", "50.000", "1"], ["s", "10", "100.000", "1"]])
    score = _score_record(per_answer, tables)
    assert next(c for c in score["cells"] if c["model"] == "late")["status"] == ad.POSTDATES
    record = ad.ad1(score)
    assert record["totals"]["answers"] == 4 and record["totals"]["up"] == 1
    assert record["models_with_no_version_in_force"] == ["late"]
    rv = record["reading_version_by_model"]["late"]
    assert rv["version"] == "B" and rv["table"] == "t" and "newest answer" in rv["rule"]
    preds = ad.predictions(score, record, None, None)
    assert preds["AD1_upward_under_2pct"]["answers"] == 4


def test_ad1_takes_the_earliest_table_even_when_its_cell_postdates_it():
    per_answer = _answers("m", [1.0, 1.0], [1.0, 1.0], tstamp=1727740800.0)  # 2024-10-01
    tables = {"t1": {"text": _table([["m", "1", "100.000", "1"]]), "categories": CATS,
                     "upper_utc": "2024-09-07T00:00:00Z"},
              "t2": {"text": _table([["m", "1", "100.000", "1"]]), "categories": CATS,
                     "upper_utc": "2025-03-31T00:00:00Z"}}
    record = ad.ad1(_score_record(per_answer, tables))
    assert record["reading_version_by_model"]["m"]["table"] == "t1"
    assert record["totals"]["answers"] == 2


def test_the_up_and_down_predictions_count_any_version_in_force():
    """The reading version (A, the smallest) shows no flip; B, also in
    force, shows one up and one down. The predictions count B's."""
    per_answer = (_answers("m", [1.0, 0.0, 1.0], [1.0, 0.0, 1.0], fa=[1.0, 0.0, 1.0],
                           fb=[1.0, 1.0, 0.0])
                  + _answers("s", [1.0] * 3, [1.0] * 3))
    tables = _one_cell_table([["m", "10", "66.667", "1"], ["s", "10", "100.000", "1"]])
    score = _score_record(per_answer, tables)
    record = ad.ad1(score)
    assert record["totals"].get("up", 0) == 0 and record["totals"].get("down", 0) == 0
    assert record["any_version_in_force"] == {"answers": 6, "up": 1, "down": 1}
    preds = ad.predictions(score, record, None, None)
    assert preds["AD1_upward_under_2pct"]["up_under_any_version_in_force"] == 1
    assert preds["AD1_upward_under_2pct"]["up_under_the_reading_version"] == 0
    assert not preds["AD1_upward_under_2pct"]["held"]
    assert preds["AD1_downward_at_most_5"]["down_under_any_version_in_force"] == 1


def test_predictions_take_the_harder_reading_of_an_upward_flip():
    per_answer = (_answers("p", [0.5, 1.0], [0.5, 1.0], fa=[0.51, 1.0], fb=[0.51, 1.0])  # a partial rise
                  + _answers("q", [1.0, 1.0], [1.0, 1.0]))
    tables = _one_cell_table([["p", "10", "75.000", "1"], ["q", "10", "100.000", "1"]])
    score = _score_record(per_answer, tables)
    assert {c["status"] for c in score["cells"]} == {ad.REPRODUCED}
    record = ad.ad1(score)
    preds = ad.predictions(score, record, None, None)
    assert preds["AD1_upward_under_2pct"]["up_under_the_reading_version"] == 1  # any F > O counts
    assert not preds["AD1_upward_under_2pct"]["held"]


# -- the site's arithmetic ---------------------------------------------------------------------


def test_to_fixed_is_javascripts():
    assert ad.to_fixed(0.125) == "0.13"      # exact tie: away from zero, as JavaScript does
    assert ad.to_fixed(1.005) == "1.00"      # 1.005 is 1.00499... in binary
    assert ad.to_fixed(45.675) == "45.67"    # 45.67499...
    assert ad.to_fixed(-0.125) == "-0.13"


def test_js_parse_float_and_global_average():
    assert ad.js_parse_float("45.3abc") == 45.3 and ad.js_parse_float(" 7") == 7.0
    assert ad.js_parse_float("-") != ad.js_parse_float("-")  # NaN
    cats = {"A": ["a1", "a2"], "B": ["b1"]}
    assert ad.global_average({"model": "m", "a1": "10", "a2": "", "b1": "20"}, cats) == 15.0
    assert ad.global_average({"model": "m", "a1": "", "a2": "", "b1": "20"}, cats) == "-"
    assert ad.global_average({"model": "grok-3", "a1": "1", "a2": "1", "b1": "1"}, cats) == 58.0


def test_competition_rank_shares_the_higher_position():
    assert ad.competition_rank({"a": 50.0, "b": 40.0, "c": 40.0, "d": 30.0, "e": "-"}) == {
        "a": 1, "b": 2, "c": 2, "d": 4, "e": None}


def test_ad3_replaces_only_reproduced_cells_and_detects_a_rank_change():
    # moved0 sits below still0 until its tablejoin cell rises by 50 points
    per_answer = (_answers("moved0", [1.0, 0.0], [1.0, 0.0], fa=[1.0, 1.0], fb=[1.0, 1.0])
                  + _answers("still0", [1.0, 1.0], [1.0, 1.0])
                  + _answers("lost", [1.0, 1.0], [1.0, 1.0]))
    tables = _one_cell_table([["moved0", "60", "50.000", "40"], ["still0", "20", "100.000", "50"],
                              ["lost", "0", "33.333", "0"]])
    score = _score_record(per_answer, tables)
    out = ad.ad2_ad3(score, tables)
    t = out["ad3"]["per_table"]["t"]
    assert t["cells_replaced"] == 2  # moved0 and still0; lost does not reproduce
    assert {r["model"] for r in t["rank_changed"]} == {"moved0", "still0"}
    assert t["largest_move"] == 1 and t["top10_position_changed"]
    assert out["ad3"]["words"] == "RANK CHANGED"
    detail = out["ad2"]["moved_cell_detail"][0]
    assert detail["direction"] == "up"
    assert detail["data_analysis_before"] == pytest.approx(55.0)
    assert detail["data_analysis_after"] == pytest.approx(80.0)


# -- made once, and the anchor -------------------------------------------------------------------


def test_write_once_is_exclusive_and_leaves_no_temp_file(tmp_path):
    out = tmp_path / "x.json"
    ad.write_once(out, {"a": 1})
    with pytest.raises(SystemExit):
        ad.write_once(out, {"a": 2})
    assert json.loads(out.read_text()) == {"a": 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.json"]


def test_write_once_is_not_blocked_by_a_stale_temp_file_of_the_same_pid(tmp_path):
    stale = tmp_path / f".x.json.{os.getpid()}.tmp"
    stale.write_text("left by a crashed run")
    ad.write_once(tmp_path / "x.json", {"a": 1})
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": 1}
    assert stale.read_text() == "left by a crashed run"


def test_attempts_are_appended_before_work(tmp_path):
    log = tmp_path / "attempts.jsonl"
    ad.append_attempt("score", log)
    ad.append_attempt("score", log)
    ad.append_attempt("read", log)
    assert len(ad.attempts("score", log)) == 2 and len(ad.attempts("read", log)) == 1


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(cwd), "PATH": "/usr/bin:/bin"})


def _varuint(n: int) -> bytes:
    out = bytearray()
    while True:
        low, n = n & 0x7F, n >> 7
        if n:
            out.append(low | 0x80)
        else:
            out.append(low)
            return bytes(out)


def _varbytes(data: bytes) -> bytes:
    return _varuint(len(data)) + data


PENDING_URI = b"https://alice.btc.calendar.opentimestamps.org"


def _pending(uri: bytes = PENDING_URI) -> bytes:
    return b"\x00" + ad.ATTESTATIONS["pending"] + _varbytes(_varbytes(uri))


def _bitcoin(height: int) -> bytes:
    return b"\x00" + ad.ATTESTATIONS["bitcoin"] + _varbytes(_varuint(height))


def _proof(digest: str, bitcoin: bool, height: int = 968216, uri: bytes = PENDING_URI,
           version: int = 1) -> bytes:
    """A well-formed OpenTimestamps file proof: magic, version, sha256 op,
    the digest, an append and a sha256 op, then a pending attestation and
    (with bitcoin) a fork into a Bitcoin one with a block height."""
    head = ad.OTS_MAGIC + _varuint(version) + b"\x08" + bytes.fromhex(digest)
    ops = b"\xf0" + _varbytes(b"\x01" * 16) + b"\x08"
    tail = (b"\xff" + _pending(uri) + _bitcoin(height)) if bitcoin else _pending(uri)
    return head + ops + tail


def _lift_aa_parser():
    """Addendum AA's proof reader, lifted from its source by syntax tree
    (importing the module would pull in AA's dependencies)."""
    source = (REPO / "scripts" / "decoder_isolation_read.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"_Cursor", "_ots_tag_or_attestation", "_ots_timestamp", "parse_ots"}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    consts = [n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in {"_UNARY_OPS", "_BINARY_OPS"}]
    namespace = {"OTS_MAGIC": ad.OTS_MAGIC, "ATTESTATIONS": ad.ATTESTATIONS}
    code = "from __future__ import annotations\n" + "\n\n".join(
        ast.get_source_segment(source, n) for n in consts + nodes)
    exec(compile(code, "<aa parse_ots>", "exec"), namespace)  # noqa: S102
    assert {"parse_ots", "_Cursor"} <= set(namespace)
    return namespace["parse_ots"]


def test_the_proof_reader_walks_the_format_and_agrees_with_aas():
    aa_parse = _lift_aa_parser()
    digest = _sha(b"snapshot")
    good = _proof(digest, bitcoin=True, height=968216)
    pending = _proof(digest, bitcoin=False)
    appended = pending + ad.ATTESTATIONS["bitcoin"]
    in_uri = _proof(digest, bitcoin=False, uri=b"https://x/" + ad.ATTESTATIONS["bitcoin"] + b"y")
    old_layout = ad.OTS_MAGIC + bytes([1, 0x08]) + bytes.fromhex(digest) + b"\x00\xf0" + \
        ad.ATTESTATIONS["bitcoin"] + b"\x00"   # the earlier tests' malformed layout
    version2 = _proof(digest, bitcoin=True, version=2)
    cases = {"good": good, "pending": pending, "appended": appended, "in_uri": in_uri,
             "old_layout": old_layout, "version2": version2}
    for name, blob in cases.items():
        assert ad.parse_ots(blob) == aa_parse(blob), name
    assert ad.bitcoin_heights(good) == [968216] and ad.anchored_digest(good) == digest
    assert ad.attestation_kinds(pending) == ["pending"] and ad.bitcoin_heights(pending) == []
    # eight tag bytes appended, or inside a calendar URI, are not an attestation
    assert ad.parse_ots(appended) is None and ad.attestation_kinds(appended) == []
    assert ad.anchored_digest(appended) is None
    assert ad.attestation_kinds(in_uri) == ["pending"] and ad.bitcoin_heights(in_uri) == []
    assert ad.bitcoin_heights(old_layout) == [] and ad.anchored_digest(old_layout) is None
    assert ad.parse_ots(version2) is None
    # every real proof in the repo reads the same through both readers
    real = sorted((REPO / "docs" / "research").glob("*.ots"))
    assert real
    for path in real:
        blob = path.read_bytes()
        assert ad.parse_ots(blob) == aa_parse(blob), path.name
    assert any(ad.bitcoin_heights(p.read_bytes()) for p in real), "no real Bitcoin attestation parsed"


def _frozen_repo(tmp_path, bitcoin=True, commit=True, quote_wrong=False, proof_of=None):
    (tmp_path / "scripts").mkdir(parents=True)
    code = tmp_path / "scripts" / "r.py"
    code.write_text("print('r')\n")
    digest = "0" * 64 if quote_wrong else _sha(code.read_bytes())
    research = tmp_path / "docs" / "research"
    research.mkdir(parents=True)
    stamp = "2026-10-22T0100Z"
    text = (f"# AD\n**Anchored as `ADDENDUM_AD_PROTOCOL.md.{stamp}.ots`.**\n"
            f"- `scripts/r.py` sha256 `{digest}`\n")
    protocol = research / "ADDENDUM_AD_PROTOCOL.md"
    protocol.write_text(text)
    snapshot = research / f"snapshots_ADDENDUM_AD_PROTOCOL_{stamp}.md"
    snapshot.write_text(text)
    target = _sha(snapshot.read_bytes()) if proof_of is None else _sha(proof_of)
    (research / f"ADDENDUM_AD_PROTOCOL.md.{stamp}.ots").write_bytes(_proof(target, bitcoin))
    _git(tmp_path, "init", "-q")
    if commit:
        _git(tmp_path, "add", "-A")
        _git(tmp_path, "commit", "-q", "-m", "freeze")
    return protocol


def test_the_anchor_needs_a_committed_snapshot_a_bitcoin_attestation_and_matching_code(tmp_path):
    protocol = _frozen_repo(tmp_path / "ok")
    record = ad.anchor_record(protocol, tmp_path / "ok", ["scripts/r.py"])
    assert record["ok"], record.get("why")
    assert record["bitcoin_block_heights"] == [968216] and record["proof_committed_unchanged"]
    pending = _frozen_repo(tmp_path / "pending", bitcoin=False)
    assert "Bitcoin" in ad.anchor_record(pending, tmp_path / "pending", ["scripts/r.py"])["why"]
    loose = _frozen_repo(tmp_path / "loose", commit=False)
    assert "not committed" in ad.anchor_record(loose, tmp_path / "loose", ["scripts/r.py"])["why"]
    wrong = _frozen_repo(tmp_path / "wrong", quote_wrong=True)
    assert "differ" in ad.anchor_record(wrong, tmp_path / "wrong", ["scripts/r.py"])["why"]
    other = _frozen_repo(tmp_path / "other", proof_of=b"some other bytes")
    rec = ad.anchor_record(other, tmp_path / "other", ["scripts/r.py"])
    assert not rec["ok"] and "does not commit to the snapshot" in rec["why"]
    unquoted = ad.anchor_record(protocol, tmp_path / "ok", ["scripts/r.py", "scripts/other.py"])
    assert not unquoted["ok"] and unquoted["code"]["not_quoted"] == ["scripts/other.py"]
    # appending an erratum keeps the anchor valid; editing the anchored bytes does not
    protocol.write_text(protocol.read_text() + "\n## Erratum\n")
    _git(tmp_path / "ok", "commit", "-qam", "erratum")
    assert ad.anchor_record(protocol, tmp_path / "ok", ["scripts/r.py"])["ok"]
    protocol.write_text(protocol.read_text().replace("# AD", "# AD!"))
    assert not ad.anchor_record(protocol, tmp_path / "ok", ["scripts/r.py"])["ok"]


def test_an_uncommitted_or_forged_proof_never_opens_a_stage(tmp_path):
    protocol = _frozen_repo(tmp_path / "r", bitcoin=False)
    proof = next((tmp_path / "r" / "docs" / "research").glob("*.ots"))
    snapshot_digest = _sha(next((tmp_path / "r" / "docs" / "research").glob("snapshots_*")).read_bytes())
    # a genuine-looking upgrade left uncommitted
    proof.write_bytes(_proof(snapshot_digest, bitcoin=True))
    rec = ad.anchor_record(protocol, tmp_path / "r", ["scripts/r.py"])
    assert not rec["ok"] and "proof is not committed" in rec["why"]
    # a well-formed pending proof whose calendar URI holds the Bitcoin tag bytes
    proof.write_bytes(_proof(snapshot_digest, bitcoin=False,
                             uri=b"https://x/" + ad.ATTESTATIONS["bitcoin"] + b"y"))
    _git(tmp_path / "r", "commit", "-qam", "tag in a uri")
    rec = ad.anchor_record(protocol, tmp_path / "r", ["scripts/r.py"])
    assert rec["proof_parses"] and rec["proof_commits_to_the_snapshot"]
    assert not rec["ok"] and "Bitcoin" in rec["why"] and rec["bitcoin_block_heights"] == []
    # eight Bitcoin tag bytes appended to a pending proof, committed
    proof.write_bytes(_proof(snapshot_digest, bitcoin=False) + ad.ATTESTATIONS["bitcoin"])
    _git(tmp_path / "r", "commit", "-qam", "forged")
    rec = ad.anchor_record(protocol, tmp_path / "r", ["scripts/r.py"])
    assert not rec["ok"] and not rec["proof_parses"] and rec["attestations"] == []


def test_every_stage_refuses_before_the_freeze():
    text = ad.PROTOCOL_PATH.read_text(encoding="utf-8")
    if ad.stamp_of(text) is not None:
        pytest.skip("the protocol is frozen; the pre-freeze refusal is history")
    for stage in ("fetch", "score", "read"):
        problems = ad.preconditions(stage)
        assert any(p.startswith("anchor:") for p in problems), stage
    assert ad.main(["check"]) == 2


# -- the stages, end to end on synthetic files -----------------------------------------------

GT = json.dumps({"colA": "x1", "colB": "y2"})
FENCED = {
    "model-a": "```python\n{'colA': 'x1', 'colB': 'y2'}\n```",
    "model-b": '```json\n{"colA": "x1", "colB": "y2"}\n```',
    "Model-C": "{'colA': 'x1', 'colB': 'y2'}",
    "local/path-model": "{'colA': 'x1', 'colB': 'y2'}",
}
QIDS = ["q1", "q2", "q3"]
T0 = 1718841600.0  # 2024-06-20


def _body(model, i):
    return FENCED[model] if (model != "model-b" or i < 2) else "{'colA': 'WRONG'}"


def _remote_entry(eid, local, role, on_mismatch="refuse", digest=True, **extra):
    data = local.read_bytes()
    entry = {"id": eid, "role": role, "url": local.as_uri(), "bytes": len(data),
             "sha256": _sha(data) if digest else None,
             "git_blob_sha1": ad.git_blob_sha1(data) if digest else None,
             "sha256_source": "synthetic", "on_mismatch": on_mismatch}
    entry.update(extra)
    return entry


def _write_parquets(src):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pq.write_table(pa.table({
        "question_id": QIDS + ["q-other"], "category": ["data_analysis"] * 4,
        "ground_truth": [GT, GT, GT, "x"], "turns": [["Join the tables."]] * 4,
        "task": ["tablejoin"] * 3 + ["cta"]}), src / "questions.parquet")
    rows = []
    for model in FENCED:
        for i, q in enumerate(QIDS):
            rows.append({"question_id": q, "answer_id": f"{model}-{q}", "model_id": model,
                         "choices": [{"index": 0, "turns": [_body(model, i)]}], "tstamp": T0 + i,
                         "category": "data_analysis", "task": "tablejoin"})
    rows.append(dict(rows[0], answer_id="dup", tstamp=T0 + 100))           # the later one wins
    rows.append(dict(rows[0], question_id="q-unknown", answer_id="unk"))   # an unknown question
    rows.append(dict(rows[0], question_id="q-other", task="cta", answer_id="cta"))
    schema = pa.schema([
        ("question_id", pa.string()), ("answer_id", pa.string()), ("model_id", pa.string()),
        ("choices", pa.list_(pa.struct([("index", pa.int64()), ("turns", pa.list_(pa.string()))]))),
        ("tstamp", pa.float64()), ("category", pa.string()), ("task", pa.string())])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), src / "answers.parquet")


def _world(tmp_path, monkeypatch, parquet=False):
    """Synthetic remote files and a manifest naming them by file:// URL;
    every banked path inside tmp_path; the anchor stubbed open (it is
    tested above). The scorer files are the vendored ones."""
    src = tmp_path / "remote"
    src.mkdir()
    if parquet:
        _write_parquets(src)
    else:
        (src / "questions.parquet").write_bytes(b"not a parquet: fetch never parses it")
        (src / "answers.parquet").write_bytes(b"neither is this")
    published = {}
    if parquet:
        pytest.importorskip("numpy")
        v0 = ad.Version(MANIFEST["livebench"]["versions"][0], REPO)
        for model in FENCED:
            vals = [ad.score_one(v0, {"turns": ["Join the tables."], "ground_truth": GT},
                                 {"choices": [{"index": 0, "turns": [_body(model, i)]}]}, FENCECHECK)["O"]
                    for i in range(len(QIDS))]
            published[model] = sum(v * 100 for v in vals) / len(vals)
    lines = ["model,cta,tablejoin,x"]
    for model in ("model-a", "model-b", "model-c", "unknown-model"):
        value = published.get({"model-c": "Model-C"}.get(model, model), 50.0)
        lines.append(f"{model},10,{value:.3f},20")
    table = "\n".join(lines) + "\n"
    remote = [_remote_entry("answers", src / "answers.parquet", "answers"),
              _remote_entry("questions", src / "questions.parquet", "questions")]
    tables = {}
    for d in ("2024_06_24", "2024_07_26"):
        (src / f"table_{d}.csv").write_text(table)
        (src / f"table_{d}_live.csv").write_text(table)
        (src / f"categories_{d}.json").write_text(json.dumps(CATS))
        remote.append(_remote_entry(f"table_{d}_pinned", src / f"table_{d}.csv", "table"))
        remote.append(_remote_entry(f"table_{d}_live", src / f"table_{d}_live.csv", "table_live",
                                    "record", digest=False, expect_equal_to=f"table_{d}_pinned"))
        remote.append(_remote_entry(f"categories_{d}", src / f"categories_{d}.json", "categories"))
        tables[d] = dict(MANIFEST["site"]["tables"][d], pinned_id=f"table_{d}_pinned",
                         live_id=f"table_{d}_live", categories_id=f"categories_{d}")
    (src / "table_before_edit.csv").write_text(table.replace("model-b,10", "model-b,11"))
    remote.append(_remote_entry("table_before_edit", src / "table_before_edit.csv", "table_parent"))
    page = {"rows": [{"row": {"question_id": q, "choices": [{"index": 0, "turns": [FENCED["model-a"]]}]},
                      "truncated_cells": []} for q in QIDS]}
    (src / "feasibility.json").write_text(json.dumps(page))
    remote.append(_remote_entry("feasibility_page", src / "feasibility.json", "feasibility", "record"))
    src_ids = {}
    for rel in MANIFEST["pinned_files"]:
        s = MANIFEST["pinned_sources"][rel]
        eid = "src_" + pathlib.Path(rel).stem
        remote.append(_remote_entry(eid, REPO / rel, "scorer_source", commit=s["commit"], path=s["path"]))
        src_ids[rel] = eid
    pinned = MANIFEST["pinned_files"]
    states = []
    for v in MANIFEST["livebench"]["versions"]:
        ids = {"tablejoin_utils_sha256": src_ids[v["tablejoin_utils"]],
               "gen_ground_truth_judgment_sha256": src_ids[v["gen_ground_truth_judgment"]]}
        if v.get("util"):
            ids["util_sha256"] = src_ids[v["util"]]
        states.append({"commit": v["commit"], "committer_time": v["in_force_from_utc"], "remote_ids": ids,
                       "tablejoin_utils_sha256": pinned[v["tablejoin_utils"]],
                       "util_sha256": pinned[v["util"]] if v.get("util") else None,
                       "gen_ground_truth_judgment_sha256": pinned[v["gen_ground_truth_judgment"]],
                       "extraction_sha256": v["extraction_sha256"]})
    manifest = {"remote": remote, "pinned_files": pinned, "pinned_sources": MANIFEST["pinned_sources"],
                "livebench": dict(MANIFEST["livebench"], first_parent_states=states),
                "site": {"tables": tables, "table_edit_after_fix": {"before_id": "table_before_edit",
                                                                    "after_id": "table_2024_07_26_pinned"}},
                "frame": {"scoping_counts": {"in_frame_cells": 6}},
                "feasibility_disclosed": {"model_id": "model-a", "answers": 3,
                                          "leading_fence_info_strings": {"python": 3}}}
    banked = tmp_path / "experiments"
    monkeypatch.setattr(ad, "FETCH_OUT", banked / "fetch.json")
    monkeypatch.setattr(ad, "SCORE_OUT", banked / "score.json")
    monkeypatch.setattr(ad, "READ_OUT", banked / "reading.json")
    monkeypatch.setattr(ad, "ATTEMPTS_LOG", banked / "attempts.jsonl")
    monkeypatch.setattr(ad, "DEFAULT_DATA_DIR", tmp_path / "data" / "ad")
    monkeypatch.setattr(ad, "anchor_record", lambda *a, **k: {"ok": True, "why": None})
    monkeypatch.setattr(ad, "load_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(ad, "lower_priority", lambda: None)  # never renice the test process
    return src, manifest


def _quiet(fn, *args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*args)
    return rc, buf.getvalue()


def _log_lines():
    return [json.loads(l) for l in ad.ATTEMPTS_LOG.read_text().splitlines()] if ad.ATTEMPTS_LOG.exists() else []


def test_fetch_banks_once_and_a_second_run_refuses_before_any_work(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch)
    rc, out = _quiet(ad.stage_fetch, ad.DEFAULT_DATA_DIR)
    assert rc == 0 and "G0 PASSED" in out
    fetch = json.loads(ad.FETCH_OUT.read_text())
    assert fetch["g0"]["passed"] and fetch["unreceived"] == []
    assert len(_log_lines()) == 1
    rc, out = _quiet(ad.stage_fetch, ad.DEFAULT_DATA_DIR)
    assert rc == 2 and "already banked" in out
    assert len(_log_lines()) == 1  # refused before the attempt was logged


def test_a_failed_transfer_banks_nothing_whatever_the_entry(tmp_path, monkeypatch):
    src, manifest = _world(tmp_path, monkeypatch)
    for eid in ("table_2024_06_24_live", "feasibility_page"):
        next(e for e in manifest["remote"] if e["id"] == eid)["url"] = (src / "gone.csv").as_uri()
    rc, out = _quiet(ad.stage_fetch, ad.DEFAULT_DATA_DIR)
    assert rc == 3 and not ad.FETCH_OUT.exists()
    assert "table_2024_06_24_live" in out and "feasibility_page" in out
    outcome = _log_lines()[-1]
    assert set(outcome["unreceived"]) == {"table_2024_06_24_live", "feasibility_page"}
    # nothing was banked, so fetch may run again once the files are back
    for eid in ("table_2024_06_24_live", "feasibility_page"):
        e = next(e for e in manifest["remote"] if e["id"] == eid)
        e["url"] = (src / ("table_2024_06_24_live.csv" if "live" in eid else "feasibility.json")).as_uri()
    rc, _ = _quiet(ad.stage_fetch, ad.DEFAULT_DATA_DIR)
    assert rc == 0 and ad.FETCH_OUT.exists()


def test_a_short_body_on_a_pinned_file_is_not_received_not_a_g0_failure(tmp_path, monkeypatch):
    src, manifest = _world(tmp_path, monkeypatch)
    real = ad.download

    def short(url, dest, cap, expect_sha256=None):
        got = real(url, dest, cap, expect_sha256)
        if url.endswith("questions.parquet"):
            dest.write_bytes(dest.read_bytes()[:10])
            got = dict(got, bytes=10)
        return got

    result = ad.fetch_all(manifest, tmp_path / "d", downloader=short)
    assert result["unreceived"] == ["questions"] and result["g0"]["refused_entries"] == []


def test_a_cut_transfer_is_caught_by_the_announced_length(tmp_path, monkeypatch):
    class Cut:
        status = 200

        def __init__(self):
            self.headers = {"Content-Length": "1000"}
            self.parts = [b"x" * 100]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            return self.parts.pop() if self.parts else b""

    monkeypatch.setattr(ad.urllib.request, "urlopen", lambda *a, **k: Cut())
    dest = tmp_path / "f.bin"
    with pytest.raises(ad.NotReceived):
        ad.download("https://example.invalid/f", dest, 10_000)
    assert not dest.exists() and list(tmp_path.iterdir()) == []


def test_a_differing_live_table_publishes_first_and_stops_nothing(tmp_path, monkeypatch):
    src, manifest = _world(tmp_path, monkeypatch)
    (src / "table_2024_06_24_live.csv").write_text("model,cta,tablejoin,x\nmodel-a,10,1.000,20\n")
    result = ad.fetch_all(manifest, tmp_path / "d")
    assert result["g0"]["passed"] and result["unreceived"] == []
    assert result["g0"]["recorded_differences_published_first"] == ["table_2024_06_24_live"]
    assert result["g0"]["recorded_roles"] == {"table_2024_06_24_live": "table_live"}


def test_a_live_table_of_the_same_size_with_other_bytes_is_a_difference(tmp_path, monkeypatch):
    src, manifest = _world(tmp_path, monkeypatch)
    data = (src / "table_2024_07_26_live.csv").read_bytes()
    (src / "table_2024_07_26_live.csv").write_bytes(data[:-2] + b"9\n")
    result = ad.fetch_all(manifest, tmp_path / "d")
    assert result["g0"]["recorded_differences_published_first"] == ["table_2024_07_26_live"]


def test_a_pinned_file_whose_bytes_differ_at_its_size_fails_g0(tmp_path, monkeypatch):
    src, manifest = _world(tmp_path, monkeypatch)
    entry = next(e for e in manifest["remote"] if e["id"] == "table_2024_06_24_pinned")
    data = (src / "table_2024_06_24.csv").read_bytes()
    (src / "table_2024_06_24.csv").write_bytes(data[:-2] + b"9\n")
    assert len((src / "table_2024_06_24.csv").read_bytes()) == entry["bytes"]
    result = ad.fetch_all(manifest, tmp_path / "d")
    assert not result["g0"]["passed"] and result["g0"]["refused_entries"] == ["table_2024_06_24_pinned"]
    assert result["unreceived"] == []


def test_g0_fails_when_a_vendored_file_is_not_its_upstream_or_a_state_differs(tmp_path, monkeypatch):
    src, manifest = _world(tmp_path, monkeypatch)
    entry = next(e for e in manifest["remote"] if e["id"] == "src_tablejoin_utils_905b71f")
    fake = tmp_path / "fake_upstream.py"
    fake.write_bytes((REPO / "scripts/ad_pinned/tablejoin_utils_905b71f.py").read_bytes() + b"# x\n")
    entry.update(url=fake.as_uri(), bytes=len(fake.read_bytes()), sha256=_sha(fake.read_bytes()),
                 git_blob_sha1=ad.git_blob_sha1(fake.read_bytes()))
    result = ad.fetch_all(manifest, tmp_path / "d")
    assert not result["g0"]["passed"]
    assert "scripts/ad_pinned/tablejoin_utils_905b71f.py" in result["g0"]["vendored_files_that_differ"]
    assert not result["g0"]["first_parent_states"]["matches"]


def test_g0_fails_on_a_version_list_that_is_not_the_states(tmp_path, monkeypatch):
    src, manifest = _world(tmp_path, monkeypatch)
    manifest["livebench"]["versions"] = manifest["livebench"]["versions"][1:]
    result = ad.fetch_all(manifest, tmp_path / "d")
    assert not result["g0"]["passed"] and result["g0"]["version_list_problems"]
    assert result["g0"]["refused_entries"] == [] and result["unreceived"] == []


def test_a_git_blob_id_is_checked(tmp_path, monkeypatch):
    src, manifest = _world(tmp_path, monkeypatch)
    entry = next(e for e in manifest["remote"] if e["id"] == "table_before_edit")
    entry.update(sha256=None, git_blob_sha1="0" * 40)
    result = ad.fetch_all(manifest, tmp_path / "d")
    assert "table_before_edit" in result["g0"]["refused_entries"]


def test_the_download_cap_holds(tmp_path):
    src = tmp_path / "big.bin"
    src.write_bytes(b"x" * 5000)
    with pytest.raises(ValueError, match="more than"):
        ad.download(src.as_uri(), tmp_path / "out.bin", 1000)
    assert not (tmp_path / "out.bin").exists()


def test_a_recompute_fetch_never_writes_into_the_banked_data_dir(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch)
    assert _quiet(ad.stage_fetch, ad.DEFAULT_DATA_DIR)[0] == 0
    banked_files = {p: p.read_bytes() for p in ad.DEFAULT_DATA_DIR.iterdir()}
    alias = pathlib.Path(os.path.relpath(ad.DEFAULT_DATA_DIR, pathlib.Path.cwd()))
    for data_dir in (alias, ad.DEFAULT_DATA_DIR / ".." / "ad"):
        rc, out = _quiet(ad.stage_fetch, data_dir, tmp_path / "recompute.json")
        assert rc == 2 and "banked fetch's data directory" in out
    assert {p: p.read_bytes() for p in ad.DEFAULT_DATA_DIR.iterdir()} == banked_files
    # a recompute into its own directory runs, and its attempt names its output
    rc, _ = _quiet(ad.stage_fetch, tmp_path / "data" / "mine", tmp_path / "recompute.json")
    assert rc == 0 and json.loads((tmp_path / "recompute.json").read_text())["g0"]["passed"]
    assert _log_lines()[-1]["out"] == str((tmp_path / "recompute.json").resolve())
    # no recompute may write over a banked artifact
    rc, out = _quiet(ad.stage_fetch, tmp_path / "data" / "other", ad.SCORE_OUT)
    assert rc == 2 and "names a banked file" in out


def test_the_cli_sends_a_recompute_fetch_to_a_new_data_dir(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch)
    seen = {}
    monkeypatch.setattr(ad, "stage_fetch", lambda data_dir, out=None: seen.update(d=data_dir, o=out) or 0)
    ad.main(["fetch", "--out", str(tmp_path / "r.json")])
    assert seen["d"].parent == ad.DEFAULT_DATA_DIR.parent and seen["d"].name.startswith("ad_recompute_")
    ad.main(["fetch"])
    assert seen["d"] == ad.DEFAULT_DATA_DIR and seen["o"] is None


def test_the_banked_score_and_reading_are_made_only_from_the_banked_inputs(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch)
    assert _quiet(ad.stage_fetch, ad.DEFAULT_DATA_DIR)[0] == 0
    other = tmp_path / "recompute_fetch.json"
    assert _quiet(ad.stage_fetch, tmp_path / "data" / "r", other)[0] == 0
    problems = ad.preconditions("score", None, other)
    assert any("made only from the banked fetch artifact" in p for p in problems)
    rc, _ = _quiet(ad.main, ["score", "--fetch", str(other)])
    assert rc == 2 and not ad.SCORE_OUT.exists()
    ad.write_once(ad.SCORE_OUT, {"complete": True, "fetch_sha256": ad.sha256_file(ad.FETCH_OUT)})
    problems = ad.preconditions("read", None, other)
    assert any("made only from the banked fetch artifact" in p for p in problems)
    problems = ad.preconditions("read", None, None, tmp_path / "elsewhere.json")
    assert any("made only from the banked score artifact" in p for p in problems)
    rc, _ = _quiet(ad.main, ["read", "--fetch", str(other)])
    assert rc == 2 and not ad.READ_OUT.exists()


def test_read_refuses_an_incomplete_score_and_a_score_of_another_fetch_separately(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch)
    assert _quiet(ad.stage_read)[0] == 2  # nothing fetched
    assert _quiet(ad.stage_fetch, ad.DEFAULT_DATA_DIR)[0] == 0
    assert _quiet(ad.stage_read)[0] == 2  # nothing scored
    digest = ad.sha256_file(ad.FETCH_OUT)
    ad.write_once(ad.SCORE_OUT, {"complete": False, "fetch_sha256": digest})
    problems = ad.preconditions("read")
    assert problems == ["the score artifact is incomplete"]
    ad.SCORE_OUT.unlink()
    ad.write_once(ad.SCORE_OUT, {"complete": True, "fetch_sha256": "0" * 64})
    assert ad.preconditions("read") == ["the score artifact was not made from this fetch artifact"]
    rc, out = _quiet(ad.stage_read)
    assert rc == 2 and "WITHHELD" in out and not ad.READ_OUT.exists()
    assert [l["stage"] for l in _log_lines()] == ["fetch"]  # read logged no attempt


def test_score_refuses_files_changed_on_disk_and_restore_puts_back_only_the_banked_bytes(
        tmp_path, monkeypatch):
    src, _ = _world(tmp_path, monkeypatch)
    assert _quiet(ad.stage_fetch, ad.DEFAULT_DATA_DIR)[0] == 0
    fetch = json.loads(ad.FETCH_OUT.read_text())
    paths = {r["id"]: ad.data_file(r["path"]) for r in fetch["remote"]}
    paths["questions"].unlink()                                   # lost
    paths["table_2024_06_24_live"].write_text("edited\n")         # changed
    problems = ad.preconditions("score")
    assert any("2 file(s) of the fetch artifact are missing" in p and "restore" in p for p in problems)
    n = len(_log_lines())
    assert _quiet(ad.stage_score)[0] == 2 and len(_log_lines()) == n  # refused before any work
    # the host now serves different bytes for one of them: it cannot be restored
    (src / "table_2024_06_24_live.csv").write_text("model,cta,tablejoin,x\n")
    rc, out = _quiet(ad.stage_restore)
    assert rc == 3 and "table_2024_06_24_live" in out
    assert paths["questions"].read_bytes() == (src / "questions.parquet").read_bytes()
    assert paths["table_2024_06_24_live"].read_text() == "edited\n"   # never replaced
    (src / "table_2024_06_24_live.csv").write_bytes((src / "table_2024_06_24.csv").read_bytes())
    rc, out = _quiet(ad.stage_restore)
    assert rc == 0 and ad.files_not_as_fetched(fetch) == []
    assert _quiet(ad.stage_restore)[1].startswith("nothing to restore")


def test_the_whole_run_on_a_synthetic_parquet_and_a_strangers_recompute(tmp_path, monkeypatch):
    """fetch, score and read end to end on synthetic files through the
    runner's own code and the vendored scorers; then the documented
    recompute from a clone with no data/."""
    pytest.importorskip("numpy")
    _world(tmp_path, monkeypatch, parquet=True)
    assert _quiet(ad.stage_fetch, ad.DEFAULT_DATA_DIR)[0] == 0
    rc, out = _quiet(ad.stage_score)
    assert rc == 0, out
    score = json.loads(ad.SCORE_OUT.read_text())
    assert score["g1"]["passed"] and score["answers_read"] == 14 and score["answers_scored"] == 12
    assert score["answer_integrity"]["duplicates_by_model"] == {"model-a": 1}
    assert score["answer_integrity"]["unknown_question_by_model"] == {"model-a": 1}
    assert score["g2"]["texts_equal"] and score["g2"]["tags_match_disclosure"]
    assert {c["status"] for c in score["cells"]} == {ad.REPRODUCED} and len(score["cells"]) == 6
    assert score["fetch_sha256"] == ad.sha256_file(ad.FETCH_OUT)
    b = next(c for c in score["cells"] if c["model"] == "model-b")
    assert b["delta"] == pytest.approx(200 / 3)  # two json-fenced correct answers flip 0 -> 1
    assert _quiet(ad.stage_score)[0] == 2  # made once
    rc, out = _quiet(ad.stage_read)
    assert rc == 0, out
    reading = json.loads(ad.READ_OUT.read_text())
    assert reading["words"] == "MOVES A NUMBER" and reading["ad2"]["models_moved_up"] == ["model-b"]
    assert reading["score_sha256"] == ad.sha256_file(ad.SCORE_OUT)
    assert _quiet(ad.stage_read)[0] == 2
    # a stranger's clone: the banked artifacts, no data/
    shutil.rmtree(ad.DEFAULT_DATA_DIR)
    n = len(_log_lines())
    rc, out = _quiet(ad.stage_score, tmp_path / "mine.json")
    assert rc == 2 and "fetch --out" not in out and "restore" in out   # a clean refusal
    assert len(_log_lines()) == n
    f, s, r = tmp_path / "s" / "fetch.json", tmp_path / "s" / "score.json", tmp_path / "s" / "reading.json"
    assert _quiet(ad.main, ["fetch", "--out", str(f), "--data-dir", str(tmp_path / "s" / "data")])[0] == 0
    assert _quiet(ad.main, ["score", "--out", str(s), "--fetch", str(f)])[0] == 0
    assert _quiet(ad.main, ["read", "--out", str(r), "--fetch", str(f), "--score", str(s)])[0] == 0
    mine = json.loads(r.read_text())
    assert mine["words"] == reading["words"]
    assert mine["licensed_sentences"] == reading["licensed_sentences"]
    assert {l.get("out") for l in _log_lines() if l["stage"] == "fetch"} >= {str(f.resolve())}


def test_a_recompute_waits_for_the_banked_artifact(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch)
    problems = ad.preconditions("fetch", tmp_path / "r.json", data_dir=tmp_path / "data" / "r")
    assert problems == ["the banked fetch.json does not exist yet; --out recomputes after it"]
    assert _quiet(ad.stage_fetch, ad.DEFAULT_DATA_DIR)[0] == 0
    assert any("the banked score.json does not exist yet" in p
               for p in ad.preconditions("score", tmp_path / "s.json"))
    assert ad.preconditions("fetch", tmp_path / "r.json", data_dir=tmp_path / "data" / "r") == []


def test_a_score_that_stops_after_it_started_leaves_its_attempt_and_why(tmp_path, monkeypatch):
    """The start clock is logged before any work; files that change between
    the precondition check and the work stop it, and the outcome says so."""
    _world(tmp_path, monkeypatch)
    assert _quiet(ad.stage_fetch, ad.DEFAULT_DATA_DIR)[0] == 0
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: True)
    calls = []
    monkeypatch.setattr(ad, "files_not_as_fetched", lambda fetch: calls.append(1) or (["questions"] if len(calls) > 1 else []))
    with pytest.raises(SystemExit, match="changed on disk since fetch"):
        _quiet(ad.stage_score)
    score_lines = [l for l in _log_lines() if l["stage"] == "score"]
    assert "started_utc" in score_lines[0] and score_lines[1]["files"] == ["questions"]
    assert not ad.SCORE_OUT.exists()


def test_read_never_overwrites_a_reading_banked_after_its_checks(tmp_path, monkeypatch):
    """Two readers race past the precondition check: the second one's
    write fails and the first one's bytes stay."""
    _world(tmp_path, monkeypatch)
    ad.write_once(ad.FETCH_OUT, {"complete": True, "g0": {"passed": True}, "remote": []})
    ad.write_once(ad.SCORE_OUT, {"complete": True, "fetch_sha256": ad.sha256_file(ad.FETCH_OUT)})
    ad.READ_OUT.write_text("first\n")
    monkeypatch.setattr(ad, "preconditions", lambda *a, **k: [])
    monkeypatch.setattr(ad, "reading_from", lambda *a, **k: {"published_first": [], "licensed_sentences": {}})
    with pytest.raises(SystemExit, match="already banked"):
        _quiet(ad.stage_read)
    assert ad.READ_OUT.read_text() == "first\n"


def test_score_withholds_after_a_failed_g0(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch)
    ad.write_once(ad.FETCH_OUT, {"complete": True, "g0": {"passed": False}, "remote": []})
    assert any("G0 did not pass" in p for p in ad.preconditions("score"))
    assert _quiet(ad.stage_score)[0] == 2 and not ad.SCORE_OUT.exists()


# -- the protocol and the freeze tool ------------------------------------------------------------


def _changed_after_the_reading(text: str) -> set:
    """Pinned files a dated erratum after the banked reading says changed.

    Before the reading every pinned file must match its frozen digest. Once
    the reading is banked a pinned file may change, but only if a dated
    erratum appended after the reading names it; recomputation then happens
    at the commit that banks the reading (the erratum says which)."""
    if not (REPO / "experiments" / "ad_reading_2026-10-21.json").exists():
        return set()
    marker = "## Erratum \u2014 2026-10-22, after the reading"
    if marker not in text:
        return set()
    tail = text[text.index(marker):]
    return {rel for rel in ("tools/fencecheck.py",) if f"`{rel}`" in tail}



def test_the_protocol_pins_every_file_the_reading_runs_and_prepares(tmp_path, monkeypatch):
    text = ad.PROTOCOL_PATH.read_text(encoding="utf-8")
    required = ad.required_files(MANIFEST)
    if ad.stamp_of(text) is not None:
        quoted = dict(ad.QUOTED_DIGEST_RE.findall(text))
        changed = _changed_after_the_reading(text)
        for rel in required:
            if rel in changed:
                assert rel in quoted, rel
                continue
            assert quoted.get(rel) == _sha((REPO / rel).read_bytes()), rel
        return
    lines = text.splitlines()
    placeholder_lines = [l for l in lines if re.search(r"[A-Z_]+_SHA_PLACEHOLDER", l)]
    assert len(placeholder_lines) == len(required), "one placeholder line per pinned file"
    for rel in required:
        hits = [l for l in placeholder_lines if f"`{rel}` sha256 `" in l]
        assert len(hits) == 1, rel
        assert len(re.findall(r"_SHA_PLACEHOLDER", hits[0])) == 1, rel
    assert "snapshots_ADDENDUM_AD_PROTOCOL_STAMP_PLACEHOLDER.md" in text
    assert "ADDENDUM_AD_PROTOCOL.md.STAMP_PLACEHOLDER.ots" in text
    # dry-run the freeze tool's prepare in a committed copy (prepare refuses
    # a file no commit holds): every placeholder fills, and the runner's own
    # regexes find the stamp and every digest
    root = tmp_path / "repo"
    for rel in required + ["docs/research/ADDENDUM_AD_PROTOCOL.md"]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, root / rel)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "draft")
    monkeypatch.setattr(fp, "REPO", root)
    protocol = root / "docs" / "research" / "ADDENDUM_AD_PROTOCOL.md"
    stamp = fp.prepare(protocol, required, "2026-10-22T0000Z")
    frozen = protocol.read_text(encoding="utf-8")
    assert "PLACEHOLDER" not in frozen and "DRAFT" not in frozen
    assert ad.stamp_of(frozen) == stamp
    quoted = dict(ad.QUOTED_DIGEST_RE.findall(frozen))
    for rel in required:
        assert quoted[rel] == _sha((REPO / rel).read_bytes()), rel
    assert fp.snapshot_path(protocol, stamp).name == f"snapshots_ADDENDUM_AD_PROTOCOL_{stamp}.md"
    # the anchor names the proof and the snapshot the way Z's and AA's do
    assert f"ADDENDUM_AD_PROTOCOL.md.{stamp}.ots" in frozen
    assert f"snapshots_ADDENDUM_AD_PROTOCOL_{stamp}.md" in frozen
    assert fp.snapshot_path(protocol, stamp).read_bytes() == protocol.read_bytes()


def test_every_literal_digest_the_protocol_quotes_matches_its_file():
    """Digests written into the draft by hand (not placeholders) must be
    right now, or `freeze_protocol.py stamp` refuses at the freeze."""
    text = ad.PROTOCOL_PATH.read_text(encoding="utf-8")
    changed = _changed_after_the_reading(text)
    for rel, digest in ad.QUOTED_DIGEST_RE.findall(text):
        if rel in changed:
            continue
        assert _sha((REPO / rel).read_bytes()) == digest, rel


def test_the_documented_prepare_command_pins_exactly_the_files_the_reading_runs():
    doc = ad.__doc__
    command = doc[doc.index("freeze_protocol.py prepare"):].split("\n    #", 1)[0]
    assert "docs/research/ADDENDUM_AD_PROTOCOL.md" in command.split("\n", 1)[0]
    codes = re.findall(r"--code (\S+)", command)
    assert len(codes) == len(set(codes))
    assert sorted(codes) == ad.required_files(MANIFEST)


def test_prepare_refuses_a_pinned_file_that_is_not_committed(tmp_path, monkeypatch):
    """A digest of bytes no commit holds cannot be checked by anyone else."""
    if ad.stamp_of(ad.PROTOCOL_PATH.read_text(encoding="utf-8")) is not None:
        pytest.skip("the protocol is frozen")
    required = ad.required_files(MANIFEST)
    root = tmp_path / "repo"
    for rel in required + ["docs/research/ADDENDUM_AD_PROTOCOL.md"]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, root / rel)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "draft")
    (root / required[0]).write_bytes((root / required[0]).read_bytes() + b"\n")
    monkeypatch.setattr(fp, "REPO", root)
    protocol = root / "docs" / "research" / "ADDENDUM_AD_PROTOCOL.md"
    before = protocol.read_bytes()
    with pytest.raises(SystemExit):
        fp.prepare(protocol, required, "2026-10-22T0000Z")
    assert protocol.read_bytes() == before
