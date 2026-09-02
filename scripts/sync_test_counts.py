#!/usr/bin/env python3
"""Rewrite every documented test count to the count the suite really has.

    python3 scripts/sync_test_counts.py            # fix in place
    python3 scripts/sync_test_counts.py --check    # report, change nothing

This module owns the patterns; `tests/test_doc_counts_agree.py` imports
them, so the fixer and the checker cannot disagree about what a claim
looks like. That is the whole point of the file.

The number has now drifted six times (83 -> 128 -> 137 -> 144 -> 151 ->
167 -> ...), twice within hours of being "re-synced by hand", and two
outside readers caught it by running `pytest -q`, which is the one
command a repo like this invites. Each hand-sync was a `sed` written
from the phrasings someone remembered:

  * the first missed the hyphenated form entirely, so "167-test suite"
    survived in our presentation and talking-point documents -- the
    numbers said out loud in a room;
  * the second missed "192-test\\n   suite", wrapped across a line.

A person re-deriving the match rule each time will keep missing a form.
So there is one rule, in one place, used by both sides.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Each pattern captures the number in group 1. Whitespace between the
# count and its noun is \s+ so a claim wrapped across lines still counts.
CLAIM_PATTERNS = (
    re.compile(r"(\d+)\s+offline tests"),
    re.compile(r"(\d+)\s+tests?\s+green"),
    re.compile(r"Test suite:\s*(\d+)\s+green"),
    re.compile(r"(\d+)-tests?\b"),
    re.compile(r"(\d+)-test\s+suite"),
    # "a public harness with 303\n  tests, 179 banked artifacts" -- a bare
    # count, no "offline", no "green". Missed on 2026-08-25, the THIRD
    # phrasing to escape this list. Adding it is necessary and is not
    # sufficient; see SUSPECT below, which exists because adding one
    # pattern per miss is the losing move this module was written about.
    re.compile(r"(\d+)\s+tests?(?=[,.;)]|\s*$)", re.MULTILINE),
)

# ANY number sitting near the word "test". Never used to rewrite -- only
# to ask whether a claim shape has escaped CLAIM_PATTERNS again.
#
# The docstring above says a person re-deriving the match rule each time
# will keep missing a form, and then the list was extended by hand three
# times anyway. A pattern list cannot close this on its own: the failure
# is always a phrasing nobody thought of, and by construction those
# cannot be enumerated. So the fixer now reports what it can SEE but
# cannot MATCH, and does not call such a run clean. A loud "I do not
# understand this claim" beats a silent miss, and it is the only version
# of this check that does not depend on having imagined every sentence.
#
# Deliberately narrow in two ways, because a check that cries wolf trains
# the reader to ignore it -- the failure mode this repository has already
# corrected once. It requires the PLURAL "tests", which excludes "100
# test receipts" and "10 ranked attempts per test"; and it allows at most
# one intervening adjective, which excludes "128 for the test". Its first
# run over the tree produced ten hits, nine of them noise, and narrowing
# it to these two rules left exactly one: a genuinely stale "83 tests" in
# a strategy document that every previous pattern had missed.
SUSPECT = re.compile(r"(?<![\d.])(\d{2,4})(?:\s+[a-z]+)?\s+tests\b")

# Sentences about the PUBLIC EXPORT, whose test count is a different
# number about a different tree.
#
# This module rewrites every documented count to the count THIS tree
# collects, and it had no idea that some sentences are not about this
# tree. `export_public.sh` drops two test files, so the export collects
# 312 where the source collects 333 -- and the application draft said the
# public repo had 333, because the source count grew and this fixer
# faithfully propagated it into a sentence it should never have touched.
#
# That is worse than the drift it was written to stop. Drift leaves a
# number stale; this OVERWROTE A CORRECT NUMBER WITH A WRONG ONE, in the
# flattering direction, in the outbound document, automatically. It was
# found by an outside reviewer who cloned the public repo and ran
# `pytest --collect-only`, which is exactly the invitation this project
# issues on purpose.
#
# So the fixer refuses these sentences rather than rewriting them. A
# refusal is loud and leaves the human to check; a wrong rewrite is
# silent and reads as verified.
# "cold clone" / "in the clone" joined the vocabulary after the gate
# rewrote "192 banked artifacts" to the source tree's 198 in a sentence
# reading "312 tests green in the clone, 192 banked artifacts" -- the
# paragraph described the public clone in every word without once using
# a phrase this pattern knew. The same defect, inside the defence built
# against it, on the day it was built. The verb forms ("a competent team
# clones this stack") stay excluded on purpose: those sentences are
# about competitors, and their figures are the source tree's.
EXPORT_CONTEXT = re.compile(
    r"public (?:repo|repository|export|cut|harness|mirror)|"
    r"arc-ttt-public|exported tree|the export\b|pubcut|"
    r"cold clone|in the clone|the clone holds|checked from a clone",
    re.IGNORECASE)


def _sentence_around(text: str, index: int) -> str:
    """The clause a claim sits in, for deciding whose tree it describes.

    Bounded at BOTH ends by sentence punctuation, blank lines and list
    item boundaries. Getting the end wrong is not cosmetic: the first
    version looked only for ". " and so ran straight past a sentence that
    ended ".\\n", swallowing the following bullet — which mentioned "the
    public repository" and made a claim about the source tree look like a
    claim about the export. A window that leaks into the next sentence
    reads a cue that was never in scope, and the whole disambiguation
    rests on this window being right.
    """
    starts = [text.rfind(s, 0, index)
              for s in (". ", ".\n", "? ", "?\n", "! ", "!\n",
                        "\n\n", "\n- ", "\n* ", "\n> ")]
    start = max(starts + [0])
    ends = [e for e in (text.find(s, index)
                        for s in (". ", ".\n", "? ", "?\n", "! ", "!\n",
                                  "\n\n", "\n- ", "\n* "))
            if e != -1]
    end = min(ends) if ends else len(text)
    return text[start:end]


def _normalized_window(text: str, index: int) -> str:
    """The claim's sentence with whitespace collapsed, for context tests.

    The vocabulary said "in the clone"; the document wrapped it as
    "in\n   the clone", the phrase did not match, a clone sentence was
    classified as source-tree, and its artifact count was silently
    rewritten to the wrong tree's number THREE MORE TIMES after the
    vocabulary was added to prevent exactly that. Hard wrapping has now
    defeated the count patterns, the export vocabulary, and nothing says
    it will stop there -- so every context test runs over a
    whitespace-collapsed copy of the window, once, here.
    """
    # Bare ">" tokens are markdown blockquote markers, not words: left
    # in, a wrapped "as\n> of 2026-08-25" normalizes to "as > of" and
    # every phrase pattern still misses. Dropped here, once.
    return " ".join(w for w in _sentence_around(text, index).split()
                    if w != ">")


def describes_the_export(text: str, index: int) -> bool:
    return bool(EXPORT_CONTEXT.search(_normalized_window(text, index)))


# A count of a COMPONENT's tests — "the tool carries its own 24 tests" —
# is not a claim about the suite, and rewriting it to the suite total is
# not a sync, it is a fabrication. This module did exactly that: it
# turned "its 23 tests" (the fencecheck test file) into "its 358 tests"
# (the whole suite), which is false, inflated, and was headed for the
# outbound application copy. Third instance in one day of the same class:
# the module assumed every "N tests" refers to one referent, and every
# count with a different referent — the export's, the clone's, a
# component's — gets overwritten with the wrong number. Subset claims
# are excluded here and owned by scripts/currency_gate.py, which
# MEASURES the component (pytest --collect-only on the one file) instead
# of assuming.
# ADJACENCY, not window vocabulary. The window version matched the bare
# word "fencecheck" or "its N tests" ANYWHERE in the sentence, so
# "checked from a cold clone, the harness keeps its 312 tests green"
# classified as component and the export's 312 was rewritten to the
# component file's 25 -- and a suite total went the same way. An
# auditor demonstrated both live. A count is a component's count only
# when the possessive is ATTACHED to it: "its 24 tests", "its own 24
# tests", "24 tests of its own", or the count directly preceded or
# followed by the component's name.
# Bare "its N tests" is NOT owned: the antecedent of "its" can be the
# tree, the export, or the tool, and an auditor demonstrated both wrong
# bindings live ("the harness keeps its 312 tests green" -> component;
# "the whole tree and its 359 tests" -> component). Only the
# unambiguous forms bind: "its OWN N tests", "fencecheck's N tests",
# "N tests of its own". Bare possessives fall through to the suspect
# scans and get refused loudly instead of rewritten wrongly.
_SUBSET_BEFORE = re.compile(
    r"(?:\bits\s+own\s+|\bfencecheck(?:\.py)?(?:'s)?\s+(?:own\s+)?)$",
    re.IGNORECASE)
_SUBSET_AFTER = re.compile(
    r"^\s*tests?\s+of\s+its\s+own\b", re.IGNORECASE)


def is_subset_claim(text: str, index: int) -> bool:
    lead = " ".join(text[max(0, index - 48):index].split())
    if _SUBSET_BEFORE.search(lead + (" " if lead and text[index - 1].isspace() else "")):
        return True
    tail_start = index
    while tail_start < len(text) and (text[tail_start].isdigit()):
        tail_start += 1
    tail = " ".join(text[tail_start:tail_start + 48].split())
    return bool(_SUBSET_AFTER.search(" " + tail))

ROOT_DOCS = ("README.md", "EVIDENCE.md", "ROADMAP.md", "paper/DRAFT.md")
# Scratch notes and dated postmortems describe a past state.
SKIP_SUBSTRINGS = ("AUDIT_RESPONSE", "MONDAY_BRIEF", "OVERNIGHT",
                   "snapshots_", "POSTMORTEM")


# A document that declares itself a RECORD is describing a past state on
# purpose, and its figures are supposed to disagree with today's.
#
# Every marker must be a phrase the document says about ITSELF, near the
# top, where a reader sees it. That is the whole constraint: opting out
# has to be visible IN the document, not hidden in a skip list, or a
# stale claim gets exempted by a name nobody reading it can see.
#
# `NOT FOR DISTRIBUTION` and `ARCHIVED` were added when `currency_gate.py`
# flagged an already-sent package and an explicitly archived FAQ. Neither
# should be rewritten, and the reason is stronger than convenience: this
# repository's rule is that outbound copy cannot be corrected after it
# ships, only errata'd. Editing a sent document so a gate goes green
# would be falsifying the record to pass a check about honesty.
RECORD_MARKERS = ("SUPERSEDED", "ARCHIVED", "NOT FOR DISTRIBUTION",
                  "DO NOT ANSWER FROM THIS FILE")


def is_superseded(text: str) -> bool:
    """True if the document declares itself a record of a past state."""
    head = text[:600].upper()
    return any(marker in head for marker in RECORD_MARKERS)


def documents() -> list[Path]:
    """Every doc a reader is invited to check.

    Swept rather than listed, so a new document cannot opt out of the
    check by not being on someone's list.
    """
    paths = [REPO / name for name in ROOT_DOCS]
    docs_dir = REPO / "docs"
    if docs_dir.is_dir():
        paths += sorted(docs_dir.rglob("*.md"))
    return [p for p in paths
            if p.exists()
            and not any(s in p.name for s in SKIP_SUBSTRINGS)
            and not is_superseded(p.read_text(encoding="utf-8"))]


def collected_test_count() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--collect-only",
         "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=REPO, timeout=900,
    )
    match = re.search(r"(\d+)\s+tests? collected", result.stdout)
    if not match:
        sys.exit(f"could not read a collection count:\n{result.stdout[-2000:]}")
    return int(match.group(1))


def stale_claims(path: Path, actual: int) -> list[tuple[int, int]]:
    """(line number, claimed count) for every claim that disagrees."""
    text = path.read_text(encoding="utf-8")
    found = []
    for pattern in CLAIM_PATTERNS:
        for match in pattern.finditer(text):
            claimed = int(match.group(1))
            if claimed != actual \
                    and not describes_the_export(text, match.start()) \
                    and not is_subset_claim(text, match.start()):
                found.append((text[:match.start()].count("\n") + 1, claimed))
    return found


def export_claims(path: Path) -> list[tuple[int, str]]:
    """Test counts this module refuses to touch, because they describe
    the public export rather than this tree. Reported, never rewritten."""
    text = path.read_text(encoding="utf-8")
    out = []
    for pattern in CLAIM_PATTERNS:
        for match in pattern.finditer(text):
            if describes_the_export(text, match.start()) \
                or is_subset_claim(text, match.start()):
                out.append((text[:match.start()].count("\n") + 1,
                            " ".join(match.group(0).split())))
    return sorted(set(out))


def uncovered_claims(path: Path, actual: int) -> list[tuple[int, str]]:
    """Text that looks like a test-count claim but no pattern matches.

    Reported, never rewritten. Rewriting on a shape the module does not
    understand is how a fixer corrupts prose; reporting it is how the
    next escaped phrasing gets noticed in one run instead of three.
    """
    text = path.read_text(encoding="utf-8")
    covered = set()
    for pattern in CLAIM_PATTERNS:
        for match in pattern.finditer(text):
            covered.update(range(match.start(), match.end()))
    out = []
    for match in SUSPECT.finditer(text):
        if int(match.group(1)) == actual:
            continue          # already correct, whatever shape it is in
        if any(i in covered for i in range(match.start(), match.end())):
            continue          # a known shape; stale_claims owns it
        if describes_the_export(text, match.start()):
            # Not an escaped phrasing -- a count about the OTHER tree,
            # which this module refuses to rewrite by design and reports
            # separately. Without this it lands in the "cannot MATCH"
            # list too, and a deliberate refusal reads as a defect.
            continue
        line = text[:match.start()].count("\n") + 1
        out.append((line, " ".join(match.group(0).split())))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="report drift and exit nonzero; change nothing")
    args = parser.parse_args()

    actual = collected_test_count()
    print(f"suite collects {actual} tests")
    drifted = 0

    uncovered: list[str] = []
    refused: list[str] = []
    for path in documents():
        name = path.relative_to(REPO)
        for line, snippet in export_claims(path):
            refused.append(f"  {name}:{line} {snippet!r}")
        for line, snippet in uncovered_claims(path, actual):
            uncovered.append(f"  {name}:{line} {snippet!r}")
        stale = stale_claims(path, actual)
        if not stale:
            continue
        drifted += len(stale)
        for line, claimed in stale:
            print(f"  {name}:{line} claims {claimed}")
        if args.check:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in CLAIM_PATTERNS:
            text = pattern.sub(
                lambda m: m.group(0)
                if describes_the_export(text, m.start())
                or is_subset_claim(text, m.start())
                else m.group(0).replace(m.group(1), str(actual), 1), text)
        path.write_text(text, encoding="utf-8")

    if uncovered:
        # Re-check after rewriting: a shape that only looked uncovered
        # because a neighbouring claim was stale is not a real escape.
        uncovered = []
        for path in documents():
            name = path.relative_to(REPO)
            for line, snippet in uncovered_claims(path, actual):
                uncovered.append(f"  {name}:{line} {snippet!r}")

    if refused:
        print(f"\n{len(refused)} count(s) left alone because the sentence "
              f"describes the PUBLIC EXPORT, not this tree. This module "
              f"once rewrote one of these to this tree's number and turned "
              f"a true statement into a false one that flattered us — an "
              f"outside reviewer caught it by cloning the public repo. "
              f"Verify each against a built export, by hand:")
        for row in refused:
            print(row)

    if uncovered:
        print(f"\n{len(uncovered)} claim shape(s) this module can SEE but "
              f"cannot MATCH — it will never rewrite these, and a phrasing "
              f"that escapes the patterns is how the count drifted three "
              f"times before:")
        for row in uncovered:
            print(row)
        print("Either rephrase the claim into a known shape, or add the "
              "shape to CLAIM_PATTERNS deliberately.")

    if not drifted and not uncovered:
        print("every documented count matches the suite")
        return 0
    if uncovered:
        return 1
    if args.check:
        print(f"\n{drifted} stale claim(s). Run without --check to fix.")
        return 1
    print(f"\nrewrote {drifted} stale claim(s) to {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
