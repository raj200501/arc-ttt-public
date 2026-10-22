#!/usr/bin/env python3
"""Every number in the outbound copy must exist in an artifact. Mechanically.

This repository's entire pitch is that claims reconcile to artifacts. Until
now that was enforced on the pages (`verify_verdict.py`, the readers, the
doc-count guard) and NOT on the outbound copy -- the one document a
stranger reads first, and the only one nobody could check automatically.

An outside reader auditing the emails against the repo found the cost of
that gap: the closing line promised "two commands recompute every number
above", which was false; a claim was stated more strongly in the email
than in our own reader; and several figures cited results that were not
in the public tree at all. Every one of those is a number that appears in
the copy and cannot be traced to an artifact -- which is exactly what a
machine can check and a proofreader reliably will not.

So this extracts every DISTINCTIVE number from each email body and looks
for it in the evidence: the banked experiment artifacts first, then
VERDICT.md and EVIDENCE.md. A number found nowhere is reported as
UNRECONCILED and the script exits non-zero. A figure whose own clause
cites an artifact is checked against that artifact only. A count smaller
than every integer the evidence leaves out cannot fail, so it is reported
UNCHECKABLE: the script exits zero on it but does not call it reconciled.

How weak that is, measured (simulated review round 10,
2026-10-22; `scripts/reconcile_false_pass_probe.py`, banked in
`experiments/reconcile_false_pass_2026-10-22.json`): on 200 made-up
figures per shape with no citation in the clause, the gate passes 178
signed deltas, 24 four-place decimals and all 200 random counts between
1 and 300
(172 of them as uncheckable). With a citation in the clause it passes 1
delta and no decimals. The evidence holds every integer from 1 to 247.

What counts as distinctive, and why the filter is deliberately narrow:
bare small integers ("20 demonstrations", "30 documents") match
everything and would drown the signal, so the gate keys on the shapes
that carry claims -- decimals, money, percentages, W/L/T records,
p-values, and multipliers. A number this misses is a number a reader was
unlikely to be able to check either.

**This is a necessary condition, not a sufficient one.** It proves a
figure appears somewhere in the evidence. It does NOT prove the sentence
around it is true, that the figure describes the same arm, or that the
comparison is fair -- three failures this project has actually committed
and corrected. Read it as a spell-checker for claims, and keep reading
the prose yourself.

    python3 scripts/reconcile_outbound.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = REPO / "docs" / "strategy" / "SEND_PACKAGE_2026-08-20.md"
# An application form is outbound copy in every sense that matters
# here: a stranger reads it, it points at the public repository, and its
# reader can check every figure in it against that repository in ten
# minutes. It was drafted under the same rule as the emails and it is
# gated by the same script, because a rule that applies only to the
# document you remembered to gate is not a rule.
APPLICATION = REPO / "docs" / "strategy" / "APPLICATION_DRAFT.md"
EVIDENCE_PAGES = ["VERDICT.md", "EVIDENCE.md", "README.md", "CHALLENGES.md",
                  "CORRECTIONS.md"]
# The outbound strategy documents beyond the emails and the application.
# They sat outside this gate until simulated review round 9 (2026-09-23)
# pointed out that the copy read first was the copy nobody checked, on the
# day one of them carried a false sentence. Their names live in a private
# list (the documents are not in the public tree); checked section by
# section (`## ` headings).
_LIST = REPO / "docs" / "strategy" / "OUTBOUND_DOCS.txt"
OUTBOUND_STRATEGY_DOCS = ([REPO / "docs" / "strategy" / n for n in _LIST.read_text().split()]
                 if _LIST.exists() else [])

# Number shapes that carry a claim. Bare integers are excluded on purpose:
# see the module docstring.
PATTERNS = {
    "decimal": r"(?<![\d.])\d?\.\d{3,4}(?![\d])",
    "money": r"\$\d+(?:,\d{3})*(?:\.\d+)?",
    "percent": r"\d+(?:\.\d+)?%",
    "record": r"\d+W/\d+L(?:/\d+T)?",
    "pvalue": r"p\s*=\s*[\d.]+e?-?\d*",
    "multiplier": r"\d+(?:\.\d+)?x\b",
    # A sign only counts at a word boundary. Without the lookbehind,
    # the range "0.53-0.94" reads as the signed number "-0.94" and the
    # gate cries wolf -- which trains a reader to ignore it, the exact
    # failure the web-form length warning exists to avoid.
    "signed": r"(?<![\d.])[+-]\d+\.\d+",
    # A COUNTED CLAIM: a bare integer is meaningless on its own, which is
    # why the shapes above skip it -- but "293 tests" or "56 corrections"
    # is a specific, checkable assertion and a reader WILL check it. The
    # gate first ran with these excluded and reported a clean pass over a
    # traction paragraph whose every figure was of exactly this shape:
    # nine claims a reader could verify, two of them checked. Keyed on a
    # closed noun list so it stays narrow, and the noun is stripped
    # before the numeric match so it reconciles like any other quantity.
    # Up to two adjectives are allowed between the number and the noun,
    # because copy is written as "293 offline tests green" and "56 dated
    # corrections", not as bare "293 tests". Requiring adjacency made the
    # pattern match nothing at all in the paragraph it was written for,
    # which is a gate that reports OK because it looked for the wrong
    # thing -- the exact failure this file's docstring is about.
    # Thousands separators only: "1,950 outputs" is one count, but "§1-2,
    # artifacts named there" is not the count "2," -- which `[\d,]*`
    # matched until the strategy documents came under this gate (2026-09-23).
    "counted": r"\b\d+(?:,\d{3})*(?:\s+[a-z]+){0,2}\s+"
               r"(?:tests?|artifacts?|corrections?|documents?|entries"
               r"|receipts?|tenants?|demonstrations?|design partners?"
               r"|packages?|sites?|instances?|candidates?|rejections?|commits?)\b",
    # "6 of 35": a share written as two counts. Never extracted until
    # simulated review round 10 (2026-10-22), so every such figure passed
    # by omission. Both counts must trace; see `_verdict`.
    "fraction": r"\b\d+(?:,\d{3})* of \d+(?:,\d{3})*\b(?![.,]\d)",
}
# Everything after the number is stripped from a "counted" token before it
# is matched numerically.
COUNT_NOUNS = re.compile(r"(?<=\d)(?:\s+[a-z]+)+$")
# Figures that are definitional, external, or arithmetic rather than
# measured -- each with the reason it needs no artifact.
EXEMPT = {
    "$1M": "the raise amount, not a measurement",
    "$0.30": "external list price, quoted with its date",
    "$2.50": "external list price, quoted with its date",
    "$0.29": "external instance quote, labelled not-a-measurement",
}
# Third-party facts about the PROGRAM being applied to, not claims about
# this project's evidence. Each is exempt only because the application
# text itself already says it could not be verified from primary source
# and tells the reader to check the live site -- which is a stronger
# treatment than reconciliation would give it, not a weaker one. If any
# of these ever appears in copy WITHOUT that caveat, delete it from here
# and let the gate fail.
APPLICATION_EXEMPT = {
    "$500K": "third-party report of a third party's published terms; "
             "sources conflict and the text says so. The primary source "
             "was egress-blocked from this machine and COULD NOT BE "
             "VERIFIED.",
    "5%": "same -- reported equity share, sources conflict, flagged "
          "unverified in the text",
    "7%": "same -- reported equity share, sources conflict, flagged "
          "unverified in the text",
    "18%": "arithmetic on a banked figure (33 of 179 artifacts), not an "
           "independent claim; the fraction it rounds is checked",
}
# Third-party results quoted in the outbound documents, cited as theirs.
OUTBOUND_STRATEGY_EXEMPT = {
    "27.64%": "NVIDIA's (NVARC) published ARC Prize 2025 public score, cited "
              "as theirs on the slide; not a claim about this project",
    "24.03%": "same -- NVARC's published private score, cited as theirs",
    # The pattern reads "$40M" as "$40", which would otherwise reconcile
    # against any unrelated 40 in an artifact: a pass that proves nothing.
    # Listed here instead, so the record says what it is.
    "$40": "the $40M round-size threshold in the comparison sample of "
           "docs/strategy/FDM1_EVIDENCE_MEMO_2026-10-21.md (section 5), "
           "quoted as that memo's finding about other companies' rounds; "
           "not a claim about this project and not a valuation",
}


def _sections(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for heading in ("## SHORT BODY", "## TECHNICAL BODY", "## GENERALIST BODY"):
        start = text.find(heading)
        if start < 0:
            continue
        end = text.find("\n## ", start + 10)
        body = text[start:end if end > 0 else len(text)]
        name = heading.replace("## ", "").replace(" BODY", "")
        if name == "SHORT":
            # the rendered email is the block quote between the rules
            match = re.search(r"\n---\n\n(> .*?)\n\n---", body, re.DOTALL)
            body = match.group(1) if match else body
        out[name] = body
    return out


def _unwrap(body: str) -> str:
    """Blockquote markers off, hard-wrapped lines joined.

    Copy here is written wrapped at ~72 columns inside `> ` quotes, so
    "56 corrections" is stored as "56\\n> corrections" and a pattern that
    spans the number and its noun sees nothing. The counted-claim pattern
    silently matched ZERO of the wrapped claims until this existed --
    which is the same failure as a gate that cannot fail, arriving by a
    different route.
    """
    lines = [re.sub(r"^\s*>\s?", "", line) for line in body.split("\n")]
    return re.sub(r"[ \t]+", " ", " ".join(lines))


def _claims(body: str, exempt: dict[str, str] | None = None
            ) -> list[tuple[str, str]]:
    body = _unwrap(body)
    exempt = {**EXEMPT, **(exempt or {})}
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for kind, pattern in PATTERNS.items():
        for match in re.finditer(pattern, body):
            token = match.group(0)
            if token in seen or token in exempt:
                continue
            seen.add(token)
            found.append((kind, token))
    return found


# A counted claim whose own clause cites a banked artifact -- "237 banked
# artifacts in 48 days (`experiments/project_timeline.json`)" -- is checked
# against THAT artifact, not against every number in the evidence. Slide
# copy said 236 while the artifact it cited said 237, and the gate passed
# it because an unrelated 236 (the public export's count) existed
# elsewhere: the "matched an unrelated number" class booked on 2026-08-25,
# again (CORRECTIONS 2026-10-21). The citation is the first backticked
# `experiments/<name>.json` after the claim, before the clause ends and
# before the next counted claim starts, so a claim without its own
# citation never borrows a neighbour's. No citation, or a cited file that
# is not on disk: the claim is checked against all the evidence, as before.
CITATION = re.compile(r"`experiments/([A-Za-z0-9_.\-]+\.json)`")
# A slide's "Sources:" line names several files for the whole slide, so it
# is not any one claim's own citation; it ends the search like a clause end.
CLAUSE_END = re.compile(r"\.(?:\s|$)|;\s|\n\n|\bSources?:")
CITATION_REACH = 240
# Sections headed "## Update — YYYY-MM-DD" are dated records: their counts
# are what the cited artifact said on that date, and the artifact has
# moved since. They are checked against all the evidence, as before.
DATED_PART = re.compile(r"^Update [—–-] \d{4}-\d{2}-\d{2}\b")


def _cited_claims(body: str, exempt: dict[str, str] | None = None
                  ) -> list[tuple[str, str, str | None]]:
    """`_claims`, plus the artifact each counted claim's own clause cites."""
    body = _unwrap(body)
    exempt = {**EXEMPT, **(exempt or {})}
    counted_starts = [m.start() for m in re.finditer(PATTERNS["counted"], body)]
    found: list[tuple[str, str, str | None]] = []
    seen: set[tuple[str, str | None]] = set()
    for kind, pattern in PATTERNS.items():
        for match in re.finditer(pattern, body):
            token = match.group(0)
            if token in exempt:
                continue
            # Every shape takes the citation in its own clause, not only
            # counts (simulated review round 10, 2026-10-22).
            limit = min([s for s in counted_starts if s > match.start()]
                        + [len(body), match.end() + CITATION_REACH])
            stop = CLAUSE_END.search(body, match.end())
            if stop:
                limit = min(limit, stop.start())
            cite = CITATION.search(body, match.end(), limit)
            cited = cite.group(1) if cite else None
            if (token, cited) in seen or (cited is None and any(
                    token == t for t, _ in seen)):
                continue
            seen.add((token, cited))
            found.append((kind, token, cited))
    return found


def _cited_text(name: str | None, cache: dict) -> str | None:
    """The cited artifact's text, or None to fall back."""
    if (name is None or name == "outbound_reconciliation.json"
            or name.startswith("reconcile_false_pass")):
        return None
    key = "text:" + name
    if key not in cache:
        path = REPO / "experiments" / name
        cache[key] = (_dash_normalise(path.read_text(encoding="utf-8"))
                      if path.is_file() else None)
    return cache[key]


def _cited_numbers(name: str | None, cache: dict) -> list[float] | None:
    """Every number in the cited artifact, or None to fall back."""
    if name is None or name == "outbound_reconciliation.json":
        return None
    if name not in cache:
        text = _cited_text(name, cache)
        cache[name] = _haystack_numbers(text) if text is not None else None
    return cache[name]


_FLOORS: dict[tuple[int, int], int] = {}


def _count_floor(numbers: list[float]) -> int:
    """The smallest positive integer absent from `numbers`.

    Every count below it is present, so a count below it CANNOT fail a
    match against these numbers. Simulated review round 10 (2026-10-22)
    found the whole evidence holds every integer from 1 to 247: the gate
    could not fail on "12 design partners". Such a count is reported as
    uncheckable, not reconciled.
    """
    key = (id(numbers), len(numbers))
    if key not in _FLOORS:
        present = {int(v) for v in numbers if v >= 1 and v == int(v)}
        floor = 1
        while floor in present:
            floor += 1
        _FLOORS[key] = floor
    return _FLOORS[key]


def _verdict(kind: str, token: str, haystack: str, numbers: list[float],
             cited_numbers: list[float] | None = None,
             cited_text: str | None = None) -> str:
    """"reconciled", "uncheckable" or "missing".

    A cited claim is checked against the cited artifact only, whatever its
    shape. A count (or either side of "N of M") smaller than every integer
    the reference set leaves out could not have failed, so it is
    "uncheckable": the gate exits zero on it but does not vouch for it.
    """
    reference = cited_numbers if cited_numbers is not None else numbers
    if kind in ("counted", "fraction"):
        if kind == "counted":
            parts = [COUNT_NOUNS.sub("", token)]
        else:
            parts = [p.strip() for p in token.split(" of ")]
        floor = _count_floor(reference)
        values = [float(p.replace(",", "")) for p in parts]
        if not all(_reconciles("counted", p, haystack, numbers, cited_numbers)
                   for p in parts):
            return "missing"
        if any(v < floor for v in values):
            return "uncheckable"
        return "reconciled"
    if kind == "percent" and cited_numbers is not None:
        text = cited_text or ""
        literal = re.search(r"(?<![\d.])" + re.escape(token), text)
        return ("reconciled" if literal or _reconciles(
            "decimal", token.rstrip("%"), haystack, numbers, cited_numbers)
                else "missing")
    return ("reconciled" if _reconciles(kind, token, haystack, numbers,
                                        cited_numbers) else "missing")


def _check(body: str, exempt: dict[str, str] | None, haystack: str,
           numbers: list[float], cache: dict, use_citations: bool = True,
           uncheckable: list[str] | None = None
           ) -> tuple[list[str], list[dict]]:
    """Reconciled tokens and missing items. Tokens the gate cannot test
    go to `uncheckable` (when given), never to `reconciled`."""
    reconciled, missing = [], []
    for kind, token, cited in _cited_claims(body, exempt):
        own = _cited_numbers(cited, cache) if use_citations else None
        text = _cited_text(cited, cache) if use_citations else None
        verdict = _verdict(kind, token, haystack, numbers, own, text)
        if verdict == "reconciled":
            reconciled.append(token)
        elif verdict == "uncheckable":
            if uncheckable is not None:
                uncheckable.append(token)
        else:
            item = {"kind": kind, "token": token}
            if own is not None:
                item["checked_against"] = f"experiments/{cited}"
            missing.append(item)
    return reconciled, missing


# First-person claims of agent work. The fence lane (the tool, all three
# census runs, the hand-adjudication of 35 sites, the rescoring) is
# agent-authored, so "I read", "I measured" or "hand-checked" beside it
# is a claim `git log` disproves. The figures in such a sentence all
# reconcile; only the subject is false, which is why the figure check
# above passed it twice (CORRECTIONS 2026-08-25 and 2026-10-22).
FIRST_PERSON_WORK = re.compile(
    r"\bI\s+(?:read|measured|hand-checked|hand-verified|found|adjudicated)\b"
    r"|\b[Hh]and-checked\b|\b[Mm]y\s+benchmark\b")
# "Attached to fence-lane work": the verb counts when its paragraph is
# about that work. "hand-checked" and "my benchmark" count anywhere.
FENCE_LANE = re.compile(
    r"fence|census|candidate site|packages|scorer|benchmark|fencecheck"
    r"|\bevals\b|json|0\.0000|0\.8958", re.IGNORECASE)
ALWAYS = re.compile(r"\b[Hh]and-checked\b|\b[Mm]y\s+benchmark\b")
# The disclosure section quotes the withdrawn claim on purpose.
DISCLOSURE_HEADING = "## HOW THE WORK IS MADE"


def first_person_work(text: str) -> list[dict]:
    """Every first-person work verb beside fence-lane work, with context.

    Paragraphs are blank-line separated; blockquote markers and hard
    wraps are removed first, so "I\\n> measured" is still seen.
    """
    out: list[dict] = []
    in_disclosure = False
    for paragraph in re.split(r"\n\s*\n", text):
        stripped = paragraph.lstrip()
        if stripped.startswith("## "):
            in_disclosure = stripped.startswith(DISCLOSURE_HEADING)
        if in_disclosure:
            continue
        flat = _unwrap(paragraph)
        for match in FIRST_PERSON_WORK.finditer(flat):
            if ALWAYS.fullmatch(match.group(0)) or FENCE_LANE.search(flat):
                start = max(0, match.start() - 60)
                out.append({"phrase": match.group(0),
                            "context": flat[start:match.end() + 60].strip()})
    return out


def _application_sections(text: str) -> dict[str, str]:
    """Every `###` block of the application, keyed by its question.

    The whole file is checked, not just the drafted answers: the
    commentary around them carries figures too, and a number that is
    wrong in the note explaining an answer is a number that will end up
    in the answer.
    """
    out: dict[str, str] = {}
    parts = re.split(r"\n### ", "\n" + text)
    for part in parts[1:]:
        heading, _, body = part.partition("\n")
        out[heading.strip()[:60]] = body
    # Anything before the first `###` -- the strategy note and the
    # blockers -- is checked as one more body rather than skipped.
    preamble = parts[0]
    if preamble.strip():
        out["(preamble: strategy note and blockers)"] = preamble
    return out


def _dash_normalise(text: str) -> str:
    """ASCII-fold the minus signs typography introduces.

    Pages are typeset with U+2212 MINUS and en-dashes; the email is typed
    with a hyphen. `-22.4` and `\u221222.4` are the same claim, and a gate
    that cannot see that reports a false failure on correct copy.
    """
    return (text.replace("\u2212", "-").replace("\u2013", "-")
                .replace("\u2014", "-"))


def _haystack() -> str:
    parts = []
    for name in EVIDENCE_PAGES:
        path = REPO / name
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    for path in sorted((REPO / "experiments").glob("*.json")):
        # THE GATE MUST NOT READ ITS OWN OUTPUT AS EVIDENCE. This artifact
        # records every token the gate could not reconcile -- so with it
        # in the haystack, running the gate twice launders any invented
        # figure into "reconciled" on the second pass. It is a verifier
        # that cannot fail, arriving through its own exhaust. Found by
        # hand-checking a figure the gate had just passed: "179 banked
        # artifacts" traced only to this file and to an unrelated 179
        # elsewhere.
        if path.name == "outbound_reconciliation.json":
            continue
        # Nor the false-pass probe's output: it holds fabricated figures
        # on purpose, and with it in the evidence they would reconcile.
        if path.name.startswith("reconcile_false_pass"):
            continue
        parts.append(path.read_text(encoding="utf-8"))
    return _dash_normalise("\n".join(parts))


def _haystack_numbers(haystack: str) -> list[float]:
    """Every number in the evidence, once, as floats."""
    values = []
    for token in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", haystack):
        try:
            values.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return values


def _reconciles(kind: str, token: str, haystack: str,
                numbers: list[float],
                cited_numbers: list[float] | None = None) -> bool:
    """Does this claim trace to a figure in the evidence?

    Matching is NUMERIC and precision-aware, not string-rounded. The first
    version of this gate built string variants by rounding -- so the
    invented figure `0.7391` produced the variant `0.7`, which matches
    almost any page, and the mutation test caught the gate passing a
    number that exists nowhere. A gate with a false-NEGATIVE machine
    inside it is worse than no gate, because it licenses the copy.

    Tolerance is half a unit in the claim's own last decimal place, so
    `+40.4` legitimately matches a banked `0.4035` (40.35 vs 40.4) while
    `0.7391` matches nothing unless something within 0.00005 exists.
    Scale conversion is tried both ways because this project writes F1 as
    points on the pages and as a fraction in the artifacts.
    """
    if kind in ("record", "pvalue", "percent"):
        # literal forms, not quantities -- but "5%" is not inside "25%"
        return re.search(r"(?<![\d.])" + re.escape(token), haystack) is not None
    if kind == "counted":
        token = COUNT_NOUNS.sub("", token)
    bare = token.lstrip("+").rstrip("x").replace("$", "").replace(",", "")
    try:
        value = float(bare)
    except ValueError:
        return token in haystack
    decimals = len(bare.split(".")[1]) if "." in bare else 0
    tolerance = 0.5 * (10 ** -decimals)
    # Scale conversion exists because this project writes F1 as points on
    # the pages and as a fraction in the artifacts. A COUNT has no such
    # duality, and allowing it would let "179 artifacts" reconcile
    # against an unrelated 1.79 -- a false positive on exactly the claim
    # shape a reader is most likely to check by hand. Counts must match
    # exactly, at their own scale.
    scales = (1.0,) if kind == "counted" else (1.0, 100.0, 0.01)
    if cited_numbers is not None:
        # the artifact the claim's own sentence names, and only that one
        numbers = cited_numbers
    for candidate in numbers:
        for scale in scales:
            if abs(candidate * scale - value) <= tolerance:
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(
        REPO / "experiments" / "outbound_reconciliation.json"))
    args = parser.parse_args()

    if not PACKAGE.exists():
        print(f"no outbound package at {PACKAGE}")
        return 0
    bodies = _sections(_dash_normalise(
        PACKAGE.read_text(encoding="utf-8")))
    haystack = _haystack()
    numbers = _haystack_numbers(haystack)
    cache: dict = {}

    report: dict[str, dict] = {}
    unreconciled_total = 0
    uncheckable_total = 0
    for name, body in bodies.items():
        unsure: list[str] = []
        reconciled, missing = _check(body, None, haystack, numbers, cache,
                                     uncheckable=unsure)
        unreconciled_total += len(missing)
        uncheckable_total += len(unsure)
        report[name] = {
            "claims_checked": len(reconciled) + len(missing),
            "reconciled": len(reconciled),
            "unreconciled": missing,
            "uncheckable": unsure,
        }

    application: dict[str, dict] = {}
    if APPLICATION.exists():
        for name, body in _application_sections(_dash_normalise(
                APPLICATION.read_text(encoding="utf-8"))).items():
            unsure = []
            reconciled, missing = _check(body, APPLICATION_EXEMPT, haystack,
                                         numbers, cache, uncheckable=unsure)
            unreconciled_total += len(missing)
            uncheckable_total += len(unsure)
            application[name] = {
                "claims_checked": len(reconciled) + len(missing),
                "reconciled": len(reconciled),
                "unreconciled": missing,
                "uncheckable": unsure,
            }

    first_person: list[dict] = []
    if APPLICATION.exists():
        first_person = first_person_work(
            APPLICATION.read_text(encoding="utf-8"))

    strategy_docs: dict[str, dict] = {}
    for doc in OUTBOUND_STRATEGY_DOCS:
        if not doc.exists():
            continue
        text = _dash_normalise(doc.read_text(encoding="utf-8"))
        for part in re.split(r"\n## ", "\n" + text):
            heading = part.split("\n", 1)[0].strip()[:60] or "(preamble)"
            unsure = []
            reconciled, missing = _check(
                part, OUTBOUND_STRATEGY_EXEMPT, haystack, numbers, cache,
                use_citations=not DATED_PART.match(heading),
                uncheckable=unsure)
            if not reconciled and not missing and not unsure:
                continue
            unreconciled_total += len(missing)
            uncheckable_total += len(unsure)
            strategy_docs[f"{doc.name} / {heading}"] = {
                "claims_checked": len(reconciled) + len(missing),
                "reconciled": len(reconciled),
                "unreconciled": missing,
                "uncheckable": unsure,
            }

    record = {
        "what": "Every distinctive number in the outbound copy, checked "
                "against the banked artifacts and the evidence pages.",
        "status": "NECESSARY, NOT SUFFICIENT. It proves a figure appears "
                  "somewhere in the evidence. It does not prove the "
                  "sentence around it is true, that it describes the same "
                  "arm, or that the comparison is fair -- three failures "
                  "this project has committed and corrected. Read the prose. "
                  "Measured on random fabricated figures: see "
                  "experiments/reconcile_false_pass_2026-10-22.json.",
        "uncheckable_means": "a count, or a side of 'N of M', smaller than "
                             "every integer its reference set leaves out. It "
                             "could not fail a match, so the gate exits zero "
                             "on it but does not count it as reconciled.",
        "pages_searched": EVIDENCE_PAGES + ["experiments/*.json"],
        "exempt": EXEMPT,
        "bodies": report,
        "application_exempt": APPLICATION_EXEMPT,
        "application_answers": application,
        "application_source": str(APPLICATION.relative_to(REPO))
                              if APPLICATION.exists() else None,
        "application_first_person_work": first_person,
        "outbound_strategy_exempt": OUTBOUND_STRATEGY_EXEMPT,
        "outbound_strategy_documents": strategy_docs,
        "unreconciled_total": unreconciled_total,
        "uncheckable_total": uncheckable_total,
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")

    for name, block in report.items():
        status = "OK" if not block["unreconciled"] else "UNRECONCILED"
        print(f"{name:12s} {block['reconciled']:3d}/{block['claims_checked']:3d} "
              f"reconciled  [{status}]")
        for item in block["unreconciled"]:
            print(f"    {item['kind']:11s} {item['token']}")
        if block["uncheckable"]:
            print(f"    uncheckable: {', '.join(block['uncheckable'])}")
    if application:
        print(f"\n-- {APPLICATION.name}")
        for name, block in application.items():
            status = "OK" if not block["unreconciled"] else "UNRECONCILED"
            print(f"{block['reconciled']:3d}/{block['claims_checked']:3d} "
                  f"[{status:12s}] {name}")
            for item in block["unreconciled"]:
                print(f"    {item['kind']:11s} {item['token']}")
            if block["uncheckable"]:
                print(f"    uncheckable: {', '.join(block['uncheckable'])}")
    if strategy_docs:
        print("\n-- outbound strategy documents")
        for name, block in strategy_docs.items():
            status = "OK" if not block["unreconciled"] else "UNRECONCILED"
            print(f"{block['reconciled']:3d}/{block['claims_checked']:3d} "
                  f"[{status:12s}] {name}")
            for item in block["unreconciled"]:
                print(f"    {item['kind']:11s} {item['token']}")
            if block["uncheckable"]:
                print(f"    uncheckable: {', '.join(block['uncheckable'])}")
    if first_person:
        print(f"\n-- {APPLICATION.name}: first-person claims of agent work")
        for item in first_person:
            print(f"    {item['phrase']!r}: ...{item['context']}...")
    print(f"\n{uncheckable_total} figure(s) uncheckable: small counts that "
          "every reference set contains, so no match could fail. Not "
          "reconciled; check them by hand.")
    print(f"\nbanked: {args.out}")
    if first_person:
        print(f"\n{len(first_person)} first-person claim(s) of agent work in "
              "the application. The fence lane is agent-authored; name the "
              "agent organisation as the subject.", file=sys.stderr)
        return 1
    if unreconciled_total:
        print(f"\n{unreconciled_total} figure(s) in the outbound copy trace to "
              "no artifact. Either bank the artifact or cut the number.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
