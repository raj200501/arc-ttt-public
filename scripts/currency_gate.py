#!/usr/bin/env python3
"""Some quantities rot. This owns every one of them, or refuses out loud.

    python3 scripts/currency_gate.py            # report, change nothing
    python3 scripts/currency_gate.py --fix      # rewrite what it owns

`scripts/reconcile_outbound.py` proves a figure EXISTS in some artifact.
That is not the same as the figure being CURRENT, and the difference is
not academic: last week's artifact is still on disk, so a count that was
right on Tuesday reconciles forever. `tests/test_reconcile_application.py`
already says this in a docstring -- and then tests it for exactly one
quantity, the correction count.

The application draft was found carrying **three different values for one
quantity in one document**: "181 banked experiment artifacts" in three
places, "192 banked artifacts" in a fourth, while the coverage map on
disk said 187 and the directory held 195. Every one of them passed the
reconciliation gate, because every one of them was true of some artifact
at some point. And the drift was not uniformly flattering -- the live
figure is 41 of 193 primary-verifiable (21%) against a quoted 33 of 181
(18%), so the stale copy was UNDERSTATING the repository. Staleness is
not bias; it is noise, and noise in a document whose entire pitch is that
its numbers hold is the expensive kind.

Two syncers already existed, for tests and for corrections. Both were
written after that same quantity drifted, one of them six times. Nobody
wrote the third, so the third drifted. Writing a fourth bespoke syncer
after the next one drifts is the losing move; this is the registry, and
adding a rotting quantity to it is one entry.

## Two kinds of quantity, and only one of them is quotable

**SYNCED** -- a live value with a stable meaning. Copy may quote it; this
rewrites it when it moves.

**BANNED** -- a value that changes faster than any document can track it.
The commit count rots on *every commit*, including the commit that syncs
it, so a synced commit count is stale before it is pushed. There is no
correct absolute value to write, and the fix is not a faster syncer: it
is to quote the invariant instead. "four in five commits are agent-written"
survives every commit that preserves the ratio. This gate therefore
refuses an absolute commit count in outbound copy rather than updating
it, which is the only version of this check that terminates.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from sync_test_counts import (  # noqa: E402  (one shared rule, not two)
    documents, describes_the_export, is_subset_claim,
    _sentence_around as _shared_sentence_around)


def _context_of(text: str, index: int) -> str:
    """Which tree or referent a count-claim describes.

    'export' and 'subset' both exist because this gate and its sibling
    each rewrote a number across one of these boundaries within a single
    day -- the export's 192 became the source's 198, and the component's
    23 became the suite's 358. Every SYNCED entry now declares the one
    context it owns, and a claim in any other context is invisible to
    it, so the wrong-referent rewrite is structurally unavailable rather
    than merely guarded against.
    """
    # Subset is checked FIRST. A sentence can be about the public export
    # AND contain a component's count -- "fencecheck.py hardened with 25
    # tests, staged at PR #1" -- and with export checked first, the
    # component count was rewritten to the export suite's 312. Fourth
    # wrong-referent rewrite of the week, caught in the gate's own --fix
    # output. The narrower referent wins because a component phrase
    # inside an export sentence still names the component; the reverse
    # is never true.
    if is_subset_claim(text, index):
        return "subset"
    if describes_the_export(text, index):
        return "export"
    return "source"

COVERAGE = REPO / "experiments" / "verification_coverage.json"
EXPORT = REPO / "experiments" / "public_export_counts.json"


# --------------------------------------------------------------------------
# Live values
# --------------------------------------------------------------------------

def export_counts() -> tuple[int, int]:
    """(tests, artifacts) in the built public export, as last measured.

    A different tree with different numbers. Read from the banked
    measurement rather than recomputed, because the export is built into
    a scratch directory that may not exist right now -- and reporting a
    fresh-looking number that was never taken is the failure this whole
    file is about.
    """
    record = json.loads(EXPORT.read_text(encoding="utf-8"))
    return (int(record["public_export_tests"]),
            int(record["public_export_artifacts"]))


def live_artifact_counts() -> tuple[int, int]:
    """(total artifacts, primary-verifiable) from the coverage map.

    Read from the map rather than counted here, so this agrees with the
    document that publishes the breakdown instead of offering a second
    opinion. But the map is itself a banked artifact and can go stale, so
    `stale_coverage_map()` checks it against the directory before any
    figure is copied out of it -- syncing copy to a stale map is the same
    laundering defect one layer down.
    """
    record = json.loads(COVERAGE.read_text(encoding="utf-8"))
    return int(record["total_artifacts"]), int(record["primary_verifiable"])


def stale_coverage_map() -> str | None:
    """Refuse to quote a map that no longer describes the directory."""
    if not COVERAGE.exists():
        return "experiments/verification_coverage.json is missing"
    record = json.loads(COVERAGE.read_text(encoding="utf-8"))
    banked = int(record["total_artifacts"])
    # The map excludes gate exhaust -- the files verifiers write -- and
    # banks which ones, so this reads the rule rather than re-stating it.
    # A duplicated exclusion tuple here would go stale the first time the
    # map's rule changed, and would go stale silently, which is the exact
    # defect this whole gate exists to catch.
    exhaust = set(record.get("excluded_as_gate_exhaust", ()))
    on_disk = len([p for p in (REPO / "experiments").glob("*.json")
                   if p.name not in exhaust])
    if banked != on_disk:
        return (f"the coverage map counts {banked} artifacts but "
                f"{on_disk} are on disk — regenerate it with "
                "`PYTHONPATH=src python3 scripts/verification_coverage.py` "
                "before syncing any figure out of it")
    return None


def commit_count() -> int:
    result = subprocess.run(["git", "rev-list", "--count", "HEAD"],
                            capture_output=True, text=True, cwd=REPO)
    return int(result.stdout.strip() or 0)


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------
#
# Each SYNCED entry: patterns capturing the number in group 1, plus a
# SUSPECT pattern that matches ANY number near the quantity's noun. The
# suspect scan never rewrites. It exists because both prior syncers were
# extended by hand after each miss -- three times for the test count, a
# third phrasing for corrections -- and a phrasing nobody thought of
# cannot be enumerated in advance. A loud "I do not understand this
# claim" beats a silent miss.

# Whitespace, OR a line break inside a markdown blockquote. Every answer
# in the application is hard-wrapped inside `> `, so "181 banked
# experiment\n> artifacts" is one claim written across two lines with a
# quote marker in the middle -- and `\s+` does not match `>`. The first
# run of this gate rewrote four artifact counts and silently walked past
# a fifth for exactly that reason, which is the same shape that once made
# the reconciliation gate report a clean pass while extracting nothing
# from the answer a reviewer is most likely to check.
SOFT = r"(?:[ \t]*\n[ \t]*>?[ \t]*|\s+)"

SYNCED = {
    "artifacts": {
        "patterns": (
            re.compile(rf"(\d+)({SOFT}banked{SOFT}experiment{SOFT}artifacts)"),
            re.compile(rf"(\d+)({SOFT}banked{SOFT}artifacts)"),
        ),
        # Deliberately requires the word "banked". Its first run flagged
        # five hits of "30 artifacts" in two k=30 run notes -- per-tenant
        # receipts inside one experiment, a different quantity that
        # happens to share a noun. A check that cries wolf trains the
        # reader to ignore it, which is the failure this repository has
        # already corrected once; so the scan is narrowed to the phrase
        # outbound copy actually uses, and the cost of that narrowing is
        # stated here rather than hidden: a claim written as "193
        # experiment artifacts", with no "banked", is invisible to it.
        "suspect": re.compile(
            rf"(?<![\d.])(\d{{2,4}})(?:{SOFT}[a-z]+){{0,2}}{SOFT}banked"
            rf"{SOFT}(?:[a-z]+{SOFT})?artifacts\b"),
        "fixer": "scripts/verification_coverage.py, then --fix here",
    },
    "primary_verifiable": {
        "patterns": (
            re.compile(r"(\d+)(\s+of\s+\d+\s+banked artifacts as\s*\n?>?\s*"
                       r"primary-verifiable)"),
            re.compile(r"(?<=says )(\d+)(\s+of)"),
        ),
        "suspect": None,   # covered by the artifacts scan and the ratio check
        "fixer": "scripts/verification_coverage.py, then --fix here",
    },
    # Same nouns, different tree. These two entries only apply inside a
    # sentence that names the public export, and the two above only apply
    # outside one -- because "312 tests" and "333 tests" are both true
    # and a gate that cannot tell them apart will helpfully make one of
    # them false, which is precisely what happened.
    "export_tests": {
        "patterns": (
            re.compile(r"(\d+)(\s+offline tests)"),
            re.compile(r"(\d+)(\s+tests?\s+green)"),
            re.compile(r"(\d+)(-tests?\b)"),
            re.compile(r"(\d+)(-test\s+suite)"),
            re.compile(r"(\d+)(\s*\n?>?\s*tests?(?=[,.;)]|\s*$))",
                       re.MULTILINE),
        ),
        "suspect": re.compile(r"(?<![\d.])(\d{2,4})(?:\s+[a-z]+)?\s+tests\b"),
        "only_in_export_context": True,
        "fixer": "scripts/export_counts.py --tree <built export>",
    },
    "tool_tests": {
        "patterns": (
            re.compile(r"(?<=its own )(\d+)(\s+tests)"),
            re.compile(r"(?<=its )(\d+)(\s+tests)"),
            re.compile(r"(\d+)(\s+tests of its own)"),
        ),
        "suspect": None,
        "context": "subset",
        "fixer": "measured live: pytest --collect-only tests/test_fencecheck.py",
    },
    "export_artifacts": {
        "patterns": (
            re.compile(rf"(\d+)({SOFT}banked{SOFT}experiment{SOFT}artifacts)"),
            re.compile(rf"(\d+)({SOFT}banked{SOFT}artifacts)"),
        ),
        "suspect": None,
        "only_in_export_context": True,
        "fixer": "scripts/export_counts.py --tree <built export>",
    },
}

# Quantities that must not appear as a CURRENT TOTAL.
#
# The ban is on the running total, not on the number. "76 commits in the
# first 33 hours" and "76 commits as of this writing" do not rot -- they
# are measurements of a closed window, which is precisely the shape this
# ban tells you to write instead. The first version of this check flagged
# all three anyway, and the two it was wrong about were the two written
# correctly. So a count carrying a scope in its own sentence is allowed,
# and only a bare running total is refused.
SCOPED = re.compile(
    r"as of\b|in the first\b|over the first\b|at that point\b|"
    r"by 20\d\d-\d\d-\d\d\b|as at\b|initial commit\b|"
    r"first \d+\s*(?:hours?|days?)\b", re.IGNORECASE)

BANNED = {
    "commits": {
        "suspect": re.compile(r"(?<![\d.])(\d{2,4})\s+commits\b"),
        "why": ("a running commit count rots on every commit, including "
                "the one that syncs it. Either scope it to a closed "
                "window (\"76 commits in the first 33 hours\") or quote "
                "the invariant (\"four in five commits are "
                "agent-written\") — both survive the next commit."),
    },
}


# The clause a match sits in, for judging whether a count is scoped to a
# closed window. Imported rather than reimplemented: the export-context
# test uses the same window, and two copies of "where does this sentence
# end" would drift apart on the first fix to either one -- which is the
# same class of defect as two copies of a count.
_sentence_around = _shared_sentence_around

# A dated row in a corrections ledger is a RECORD of a past value and is
# supposed to disagree with today's. Excluding by document rather than by
# guessing at line shape, because the whole page is historical.
HISTORICAL = ("CORRECTIONS.md",)


def scan(text: str) -> dict:
    """Everything this gate has an opinion about, for one document."""
    want = _targets()

    stale, unowned, banned = [], [], []
    for name, spec in SYNCED.items():
        target = want[name]
        wants = spec.get("context",
                         "export" if spec.get("only_in_export_context")
                         else "source")
        owned_spans = []
        for pattern in spec["patterns"]:
            for match in pattern.finditer(text):
                if _context_of(text, match.start()) != wants:
                    continue
                owned_spans.append(match.span(1))
                if int(match.group(1)) != target:
                    stale.append((name, int(match.group(1)), target,
                                  text[:match.start()].count("\n") + 1))
        suspect = spec["suspect"]
        if suspect is None:
            continue
        for match in suspect.finditer(text):
            if _context_of(text, match.start()) != wants:
                continue
            if int(match.group(1)) == target:
                continue
            if any(s <= match.start(1) < e for s, e in owned_spans):
                continue
            unowned.append((name, " ".join(match.group(0).split()),
                            text[:match.start()].count("\n") + 1))

    for name, spec in BANNED.items():
        for match in spec["suspect"].finditer(text):
            if SCOPED.search(_sentence_around(text, match.start())):
                continue
            banned.append((name, " ".join(match.group(0).split()),
                           text[:match.start()].count("\n") + 1))

    return {"stale": stale, "unowned": unowned, "banned": banned}


def tool_test_count() -> int:
    """The fencecheck test file's own collection count, measured.

    Measured per run rather than stored, because a stored copy of this
    number is what went stale as 23 the day the file grew to 24 -- and a
    sibling fixer then "corrected" it to the whole suite's 358.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_fencecheck.py", "-q",
         "--collect-only", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=REPO, timeout=300)
    match = re.search(r"(\d+)\s+tests? collected", result.stdout)
    if not match:
        raise SystemExit("could not collect tests/test_fencecheck.py; a "
                         "subset count with no measurement behind it is "
                         "exactly what this gate exists to refuse")
    return int(match.group(1))


def _targets() -> dict[str, int]:
    total, primary = live_artifact_counts()
    export_tests, export_artifacts = export_counts()
    return {"artifacts": total, "primary_verifiable": primary,
            "export_tests": export_tests,
            "export_artifacts": export_artifacts,
            "tool_tests": tool_test_count()}


def fix(text: str) -> tuple[str, list[str]]:
    want = _targets()
    changed: list[str] = []
    for name, spec in SYNCED.items():
        target = str(want[name])
        wants = spec.get("context",
                         "export" if spec.get("only_in_export_context")
                         else "source")

        def replace(match: re.Match, target: str = target, name: str = name,
                    wants: str = wants) -> str:
            # `text` is captured from the enclosing scope deliberately:
            # the scope test must run against the ORIGINAL document, so a
            # rewrite earlier in the same pass cannot shift the sentence
            # boundaries a later decision depends on.
            if _context_of(text, match.start()) != wants:
                return match.group(0)
            if match.group(1) != target:
                changed.append(f"{name}: {match.group(1)} -> {target}")
            return target + match.group(2)

        for pattern in spec["patterns"]:
            text = pattern.sub(replace, text)
    return text, changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true",
                        help="rewrite the counts this gate owns")
    args = parser.parse_args()

    problem = stale_coverage_map()
    if problem:
        print(f"REFUSING: {problem}")
        return 1

    total, primary = live_artifact_counts()
    print(f"live: {total} banked artifacts, {primary} primary-verifiable "
          f"({round(100 * primary / total)}%), {commit_count()} commits "
          "(never quotable)\n")

    stale_total = unowned_total = banned_total = 0
    for path in documents():
        if path.name in HISTORICAL:
            continue
        text = path.read_text(encoding="utf-8")
        if args.fix:
            new_text, changed = fix(text)
            if changed:
                path.write_text(new_text, encoding="utf-8")
                for line in changed:
                    print(f"  fixed {path.relative_to(REPO)}: {line}")
                text = new_text

        found = scan(text)
        rel = path.relative_to(REPO)
        for name, claimed, target, line in found["stale"]:
            stale_total += 1
            print(f"  STALE   {rel}:{line}  {name} says {claimed}, "
                  f"live is {target}")
        for name, snippet, line in found["unowned"]:
            unowned_total += 1
            print(f"  UNOWNED {rel}:{line}  {name}: {snippet!r} — this gate "
                  "cannot rewrite this shape, so it did NOT update it")
        for name, snippet, line in found["banned"]:
            banned_total += 1
            print(f"  BANNED  {rel}:{line}  {snippet!r} — "
                  f"{BANNED[name]['why']}")

    if not (stale_total or unowned_total or banned_total):
        print("every rotting quantity in outbound copy is current")
        return 0
    print(f"\n{stale_total} stale, {unowned_total} unrecognised, "
          f"{banned_total} banned.")
    if unowned_total:
        print("An unrecognised shape is NOT a pass. Rephrase it to a shape "
              "the registry owns, or add the shape deliberately.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
