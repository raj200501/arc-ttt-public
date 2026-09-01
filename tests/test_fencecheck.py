"""The shipped tool is the product. Test it like one.

`tools/fencecheck.py` is the artifact this project hands to a stranger and
asks them to run. It had **no tests, no CI step, and no caller** anywhere
in a repository carrying 333 of them -- `grep -rn fencecheck` outside the
tool itself returned nothing. An outside reviewer put that second on his
list of what would have to change, and he was right: a repository whose
entire argument is "our verification actually catches things" shipping an
unverified verifier is the wrong shape, whatever the rest of the suite
says.

These cover the behaviours a first-time user actually hits -- the fence
shapes, the suppression marker, the narrowing that keeps false positives
down, the exit codes a CI step would branch on -- plus the two defects
the same review found by running it.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "fencecheck.py"

pytestmark = pytest.mark.skipif(not TOOL.exists(),
                                reason="the tool is not in this cut")


def _load():
    spec = importlib.util.spec_from_file_location("_fencecheck", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fc = _load()


# ------------------------------------------------------------- strip_fence
@pytest.mark.parametrize("raw,expected,was_fenced", [
    ('{"a": 1}', '{"a": 1}', False),
    ('```json\n{"a": 1}\n```', '{"a": 1}', True),
    ('```\n{"a": 1}\n```', '{"a": 1}', True),
    ('  ```json\n{"a": 1}\n```  ', '{"a": 1}', True),
    # Unterminated: the model ran out of tokens mid-fence. The opening
    # fence still has to come off, or the recovery under-counts exactly
    # the outputs that were truncated -- and truncation correlates with
    # the long documents, so the miss would not be random.
    ('```json\n{"a": 1}', '{"a": 1}', True),
    # A fence around something that is not JSON stays not-JSON. Stripping
    # is not allowed to invent a parse.
    ('```\nnot json at all\n```', 'not json at all', True),
])
def test_strip_fence_shapes(raw: str, expected: str, was_fenced: bool) -> None:
    text, fenced = fc.strip_fence(raw)
    assert (text, fenced) == (expected, was_fenced)


def test_stripping_never_turns_invalid_into_valid_by_accident() -> None:
    """The tool's whole claim is that a fence hides a CORRECT answer.

    If stripping could rescue genuinely malformed output, every "recovered
    by stripping" count would be inflated by the model's real mistakes,
    and the headline finding would be measuring the wrong thing.
    """
    text, _ = fc.strip_fence('```json\n{"a": 1,\n```')
    assert not fc._parses(text)


def test_a_bare_fence_with_no_newline_is_still_handled() -> None:
    text, fenced = fc.strip_fence('```{"a": 1}```')
    assert fenced
    assert json.loads(text) == {"a": 1}


# -------------------------------------------------------------- scanning
# The shape the tool exists to find, and only this shape: a parse of a
# model's text whose failure is SWALLOWED into a zero. That last part is
# the whole point and is easy to get wrong -- a bare `json.loads` that
# raises is a loud failure and the tool deliberately does not report it,
# because a crash gets noticed and a silent 0.0 does not. Writing these
# fixtures without the try/except produced three failing tests and the
# tool was right in all three.
FENCE_BAIT = '''
def grade(completion):
    """Score a model completion."""
    try:
        obj = json.loads(completion)
    except ValueError:
        return 0.0
    return obj["answer"]
'''

SUPPRESSED = '''
def grade(completion):
    """Score a model completion."""
    try:
        obj = json.loads(completion)  # fencecheck: ignore -- handled upstream
    except ValueError:
        return 0.0
    return obj["answer"]
'''

NOT_MODEL_OUTPUT = '''
def load_config(path, completion=None):
    """Mentions a completion, but parses a file."""
    try:
        with open(path) as handle:
            return json.loads(handle.read())
    except ValueError:
        return None
'''

# The same exposed parse, but the failure reaches the caller. Loud, so
# not a finding.
RAISES_INSTEAD = '''
def grade(completion):
    """Score a model completion."""
    obj = json.loads(completion)
    return obj["answer"]
'''


def test_it_finds_an_exposed_parse_of_model_output() -> None:
    findings = fc.scan_source(FENCE_BAIT, "grade.py")
    assert len(findings) == 1, findings
    assert findings[0]["line"] == 5
    assert findings[0]["call"] == "json.loads"


def test_a_parse_that_raises_is_not_a_finding() -> None:
    """The narrowing that makes the census number mean something.

    A fence that causes a crash gets found by whoever runs the eval. A
    fence that causes a silent 0.0 does not, and the entire claim is
    about the second kind. Reporting both would inflate every count here
    with failures that are not fail-open at all.
    """
    assert fc.scan_source(RAISES_INSTEAD, "grade.py") == []


def test_the_ignore_marker_suppresses() -> None:
    """A tool with no escape hatch gets deleted after its first false
    positive, so the escape hatch is load-bearing and has to work."""
    assert fc.scan_source(SUPPRESSED, "grade.py") == []


def test_reading_a_file_is_not_model_output() -> None:
    """The narrowing that keeps this usable.

    Almost every Python file parses JSON somewhere. Only the ones parsing
    a MODEL's text can lose a score to a fence, and a scanner that cannot
    tell the difference reports every repository as broken.
    """
    assert fc.scan_source(NOT_MODEL_OUTPUT, "config.py") == []


def test_a_file_with_no_model_vocabulary_is_skipped_entirely() -> None:
    assert fc.scan_source("import json\njson.loads(x)\n", "x.py") == []


def test_syntax_errors_do_not_crash_the_scan() -> None:
    """Scanning somebody else's repository means meeting Python 2, vendored
    junk and half-written files. Any of them crashing the run makes the
    tool unusable on exactly the large corpora it is sold for."""
    assert fc.scan_source("def broken(:\n", "broken.py") == []


# ----------------------------------------------------------------- CLI
def _run(*args: str, cwd: pathlib.Path | None = None):
    return subprocess.run([sys.executable, str(TOOL), *args],
                          capture_output=True, text=True,
                          cwd=str(cwd or REPO), timeout=300)


def test_version_runs_with_no_arguments_and_no_install() -> None:
    result = _run("--version")
    assert result.returncode == 0
    assert "fencecheck" in (result.stdout + result.stderr)


def test_scan_exits_nonzero_on_a_finding(tmp_path: pathlib.Path) -> None:
    """The exit code is the contract a CI step branches on."""
    (tmp_path / "grade.py").write_text(FENCE_BAIT, encoding="utf-8")
    result = _run("scan", str(tmp_path))
    assert result.returncode == 1, result.stdout


def test_scan_exits_zero_on_a_clean_tree(tmp_path: pathlib.Path) -> None:
    (tmp_path / "config.py").write_text(NOT_MODEL_OUTPUT, encoding="utf-8")
    result = _run("scan", str(tmp_path))
    assert result.returncode == 0, result.stdout


def test_scanning_a_single_file_names_that_file(
        tmp_path: pathlib.Path) -> None:
    """Found by a reviewer running it: `scan <file>` printed the path as
    `.`, because the path was made relative to a root that WAS the file.
    A finding you cannot locate is not a finding, and this is the first
    command anyone tries on a tool they have just downloaded."""
    target = tmp_path / "grade.py"
    target.write_text(FENCE_BAIT, encoding="utf-8")
    findings, files = fc.scan_path(target)
    assert files == 1
    assert findings, "the single-file path found nothing at all"
    assert findings[0]["file"].endswith("grade.py"), findings[0]["file"]
    assert findings[0]["file"] != "."


# ---------------------------------------------------------------- score
def test_score_reports_the_recovery(tmp_path: pathlib.Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("\n".join(json.dumps(row) for row in [
        {"prediction": '```json\n{"a": 1}\n```'},
        {"prediction": '```json\n{"b": 2}\n```'},
        {"prediction": '{"c": 3}'},
        {"prediction": 'genuinely malformed ['},
    ]) + "\n", encoding="utf-8")

    report = fc.score_file(predictions)
    assert report["outputs"] == 4
    assert report["fenced"] == 2
    assert report["parse_as_written"] == 1
    assert report["parse_after_stripping"] == 3
    # Only the two fenced-but-correct ones. The malformed row must not be
    # counted as recovered, or the tool overstates its own finding.
    assert report["recovered_by_stripping"] == 2


def test_score_reads_a_json_array_as_well_as_jsonl(
        tmp_path: pathlib.Path) -> None:
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps(
        [{"output": '```json\n{"a": 1}\n```'}]), encoding="utf-8")
    assert fc.score_file(predictions)["recovered_by_stripping"] == 1


def test_score_exits_nonzero_when_a_fence_cost_a_score(
        tmp_path: pathlib.Path) -> None:
    predictions = tmp_path / "p.jsonl"
    predictions.write_text(
        json.dumps({"prediction": '```json\n{"a": 1}\n```'}) + "\n",
        encoding="utf-8")
    assert _run("score", str(predictions)).returncode == 1


def test_score_exits_zero_when_nothing_was_lost(
        tmp_path: pathlib.Path) -> None:
    predictions = tmp_path / "p.jsonl"
    predictions.write_text(json.dumps({"prediction": '{"a": 1}'}) + "\n",
                           encoding="utf-8")
    assert _run("score", str(predictions)).returncode == 0


# ------------------------------------------------------------ the promise
def test_the_tool_is_importable_without_any_third_party_package(
        tmp_path: pathlib.Path) -> None:
    """"One file, no install" is the pitch. This is the test of it.

    Run with an import path carrying nothing but the standard library, so
    a dependency that crept in through a transitive import fails here
    rather than on a stranger's machine thirty seconds after they
    downloaded it.
    """
    result = subprocess.run(
        [sys.executable, str(TOOL), "--version"],
        capture_output=True, text=True, timeout=120,
        cwd=str(tmp_path),
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
             "PYTHONPATH": "", "PYTHONNOUSERSITE": "1"})
    assert result.returncode == 0, result.stderr[-1500:]


# ------------------------------------------------- the two-tree boundary
def test_a_clone_context_count_is_never_rewritten_to_the_source_count():
    """REGRESSION. The currency gate rewrote "192 banked artifacts" to the
    source tree's 198 inside a sentence reading "312 tests green in the
    clone, 192 banked artifacts" — a paragraph describing the public
    clone in every word without once using a phrase the export-context
    pattern knew. That is the exact defect the pattern was built against,
    committed by the defence itself, on the day it was built.

    This does not belong in this file by subject, but it pins the same
    lesson the file exists for: an instrument that cannot fail its own
    test is decoration.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_stc", REPO / "scripts" / "sync_test_counts.py")
    stc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stc)

    paragraph = (
        "**UNBLOCKED — the link is live and checked from a cold\n"
        "clone.** `HEAD` `dc80f1d`, 312 tests (306 pass, 6 skip) green "
        "in the clone, 192\nbanked artifacts, zero leak-gate hits.")
    idx = paragraph.index("192")
    assert stc.describes_the_export(paragraph, idx), (
        "a sentence about the public clone is not recognised as being "
        "about the other tree, so its counts will be rewritten to this "
        "tree's numbers — the cross-tree defect, again")

    source_sentence = (
        "437 commits as of 2026-08-25, 357 offline tests, 198 banked "
        "experiment artifacts, and a leaderboard entry along the way. "
        "All of it is public and all of it runs on a cold clone.")
    idx = source_sentence.index("198")
    assert not stc.describes_the_export(source_sentence, idx), (
        "the cold-clone vocabulary leaked across a sentence boundary "
        "into source-tree counts — the fix overshot")


def test_the_reviewer_bait_variable_names_are_in_scope() -> None:
    """REGRESSION. A reviewer wrote a synthetic fail-open grader using the
    variable name `model_text` and the scan reported nothing exposed. The
    vocabulary knew `model_output` but not `model_text`; it also lacked
    `sampled`, which is how openai/evals itself names the completion in
    the basic elsuites. A precision-first design earns its false
    negatives one at a time — it does not get to keep the ones a user has
    already tripped over.
    """
    bait = '''
def grade(model_text):
    try:
        obj = json.loads(model_text)
    except ValueError:
        return 0.0
    return obj["answer"]
'''
    assert len(fc.scan_source(bait, "grade.py")) == 1, "model_text missed"
    evals_shaped = bait.replace("model_text", "sampled")
    assert len(fc.scan_source(evals_shaped, "g.py")) == 1, "sampled missed"


def test_context_vocabulary_survives_hard_wrapping() -> None:
    """REGRESSION. "in the clone" wrapped as "in\\n   the clone" defeated
    the export-context vocabulary, so a clone sentence was classified as
    source-tree and its artifact count was silently rewritten to the
    wrong tree's number — three more times, after the vocabulary was
    added to prevent exactly that. Context tests now run over a
    whitespace-collapsed window."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_stc2", REPO / "scripts" / "sync_test_counts.py")
    stc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stc)
    text = ("state. The suite runs green in\n"
            "   the clone — 312 collected, both\n"
            "   measurements — and the tree holds 192 banked artifacts.\n")
    assert stc.describes_the_export(text, text.index("192"))


def test_the_auditors_five_gate_exploits_stay_dead() -> None:
    """REGRESSION, end-to-end through fix() and scan() — not classifiers.

    An audit demonstrated five live wrong-referent paths in one pass:
    a greedy "says N of" pattern fabricating counts; window-vocabulary
    subset context stealing an export count and a suite total; a scoped
    commit count flagged because its "as of" wrapped across a
    blockquote; and a wrapped bare running total invisible to the ban.
    Each is pinned against the REWRITE/SCAN path itself, because the
    previous regression tests pinned only the classifiers and would
    have passed while fix() ignored them.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_cg", REPO / "scripts" / "currency_gate.py")
    cg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cg)

    fabricate = "By round 2 the ledger says 2 of 4 arms were accepted."
    assert cg.fix(fabricate)[0] == fabricate

    steal_export = ("Checked from a cold clone, the harness keeps its "
                    "312 tests green.")
    assert "312" in cg.fix(steal_export)[0]

    steal_suite = ("Before shipping fencecheck we reran the whole tree "
                   "and its 360 tests.")
    assert "360" in cg.fix(steal_suite)[0]

    scoped_wrapped = "The log shows 437 commits as\n> of 2026-08-25."
    assert not cg.scan(scoped_wrapped)["banned"]

    bare_wrapped = "The tree carries 512\n> commits and keeps growing."
    assert cg.scan(bare_wrapped)["banned"]


def test_stripping_returns_the_exact_unrepaired_body() -> None:
    """Strengthens the never-invent-a-parse pin: the earlier version
    passed under any behavior, identity included, because its fixture
    parsed under none. This pins the exact output."""
    text, fenced = fc.strip_fence('```json\n{"a": 1,\n```')
    assert fenced and text == '{"a": 1,'


def test_a_prose_prefixed_fence_is_out_of_scope_and_that_is_deliberate():
    """The leading-fence-only rule, pinned as a DOCUMENTED scope.

    A real banked prediction (k=3 arm, h-3305) wraps correct JSON in a
    fence behind a prose preamble; strip_fence does not count it,
    because the tool's question is 'would stripping one enclosing fence
    rescue this output' — and it would not, the prose remains. The
    fence-rate undercount this produces is disclosed as a dated erratum
    on Addendum R. If this behavior ever changes, that erratum and the
    frozen reading's zero margin must be revisited in the same commit.
    """
    prosey = 'Here is the corrected version:\n\n```\n{"a": 1}\n```'
    text, fenced = fc.strip_fence(prosey)
    assert not fenced
    assert text == prosey.strip()


def test_score_refuses_a_single_json_document(tmp_path) -> None:
    """R2's round-3 find, pinned: fed a banked EXPERIMENT ARTIFACT (one
    pretty-printed JSON object whose lines parse as nothing), score used
    to fall through to the line scorer, score the artifact's own lines,
    and print 'Nothing is being lost to a fence in this file' with exit
    0 — a clean bill for a file whose nested predictions were fenced.
    A fail-open in the fail-open detector. It now refuses loudly."""
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({
        "what": "an experiment artifact, not an outputs file",
        "predictions": {"d1": '```json\n{"a": 1}\n```'}}, indent=2),
        encoding="utf-8")
    report = fc.score_file(artifact)
    assert "refused" in report and "single JSON document" in report["refused"]


def test_score_refuses_mixed_parseable_and_unparseable_lines(tmp_path) -> None:
    """Half-JSONL is neither JSONL nor plain text: scoring only the
    readable subset would publish a rate about a file it did not read."""
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text('{"raw": "{}"}\nnot json at all\n', encoding="utf-8")
    report = fc.score_file(mixed)
    assert "refused" in report


def test_score_accepts_a_single_output_record(tmp_path) -> None:
    """A whole-file dict that IS an output record (carries a raw-text
    key) stays scoreable — the refusal is scoped to non-record files."""
    record = tmp_path / "one_record.json"
    record.write_text(json.dumps({"raw": '```json\n{"a": 1}\n```'}),
                      encoding="utf-8")
    report = fc.score_file(record)
    assert report["outputs"] == 1 and report["recovered_by_stripping"] == 1
