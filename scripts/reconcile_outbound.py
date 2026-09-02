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
UNRECONCILED and the script exits non-zero.

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
    "counted": r"\b\d[\d,]*(?:\s+[a-z]+){0,2}\s+"
               r"(?:tests?|artifacts?|corrections?|documents?|entries"
               r"|receipts?|tenants?|demonstrations?|design partners?"
               r"|packages?|sites?|instances?|candidates?|rejections?|commits?)\b",
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
                numbers: list[float]) -> bool:
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
        return token in haystack  # these are literal forms, not quantities
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

    report: dict[str, dict] = {}
    unreconciled_total = 0
    for name, body in bodies.items():
        reconciled, missing = [], []
        for kind, token in _claims(body):
            if _reconciles(kind, token, haystack, numbers):
                reconciled.append(token)
            else:
                missing.append({"kind": kind, "token": token})
        unreconciled_total += len(missing)
        report[name] = {
            "claims_checked": len(reconciled) + len(missing),
            "reconciled": len(reconciled),
            "unreconciled": missing,
        }

    application: dict[str, dict] = {}
    if APPLICATION.exists():
        for name, body in _application_sections(_dash_normalise(
                APPLICATION.read_text(encoding="utf-8"))).items():
            reconciled, missing = [], []
            for kind, token in _claims(body, APPLICATION_EXEMPT):
                if _reconciles(kind, token, haystack, numbers):
                    reconciled.append(token)
                else:
                    missing.append({"kind": kind, "token": token})
            unreconciled_total += len(missing)
            application[name] = {
                "claims_checked": len(reconciled) + len(missing),
                "reconciled": len(reconciled),
                "unreconciled": missing,
            }

    record = {
        "what": "Every distinctive number in the outbound copy, checked "
                "against the banked artifacts and the evidence pages.",
        "status": "NECESSARY, NOT SUFFICIENT. It proves a figure appears "
                  "somewhere in the evidence. It does not prove the "
                  "sentence around it is true, that it describes the same "
                  "arm, or that the comparison is fair -- three failures "
                  "this project has committed and corrected. Read the prose.",
        "pages_searched": EVIDENCE_PAGES + ["experiments/*.json"],
        "exempt": EXEMPT,
        "bodies": report,
        "application_exempt": APPLICATION_EXEMPT,
        "application_answers": application,
        "application_source": str(APPLICATION.relative_to(REPO))
                              if APPLICATION.exists() else None,
        "unreconciled_total": unreconciled_total,
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")

    for name, block in report.items():
        status = "OK" if not block["unreconciled"] else "UNRECONCILED"
        print(f"{name:12s} {block['reconciled']:3d}/{block['claims_checked']:3d} "
              f"reconciled  [{status}]")
        for item in block["unreconciled"]:
            print(f"    {item['kind']:11s} {item['token']}")
    if application:
        print(f"\n-- {APPLICATION.name}")
        for name, block in application.items():
            status = "OK" if not block["unreconciled"] else "UNRECONCILED"
            print(f"{block['reconciled']:3d}/{block['claims_checked']:3d} "
                  f"[{status:12s}] {name}")
            for item in block["unreconciled"]:
                print(f"    {item['kind']:11s} {item['token']}")
    print(f"\nbanked: {args.out}")
    if unreconciled_total:
        print(f"\n{unreconciled_total} figure(s) in the outbound copy trace to "
              "no artifact. Either bank the artifact or cut the number.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
