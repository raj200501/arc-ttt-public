"""Every document that points a reader at the results-against-thesis
artifact must state the count that artifact enumerates.

Round 7 of the review simulation caught the application draft saying
"seven results" beside a link to an artifact that enumerated twelve, and
a second sentence saying "nine" beside the same link. A count that lives
in prose beside a link rots the moment the artifact grows; the artifact
is the referent, so the prose is checked against it here.

Rule: in each paragraph that names `results_against_thesis_<date>.json`,
either the artifact's total appears (as a word or a numeral), or every
per-thesis count appears (the split form, "nine ... three"). The linked
date must be the newest artifact of that name -- an old enumeration is a
stale count with extra steps.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# The application draft is not exported to the public tree; README and
# EVIDENCE are, and are checked wherever this test runs.
DOCS = [p for p in (
    REPO / "docs" / "strategy" / "APPLICATION_DRAFT.md",
    REPO / "README.md",
    REPO / "EVIDENCE.md",
) if p.exists()]
WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
    12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
    20: "twenty",
}
LINK = re.compile(r"results_against_thesis_(\d{4}-\d{2}-\d{2})\.json")


def _newest_artifact() -> Path:
    found = sorted((REPO / "experiments").glob("results_against_thesis_*.json"))
    assert found, "no results_against_thesis artifact banked"
    return found[-1]


def _counts(artifact: Path) -> tuple[int, list[int]]:
    data = json.loads(artifact.read_text())
    per_thesis = [len(t["results"]) for t in data["theses"]]
    assert data["count"] == sum(per_thesis), "artifact's own count disagrees with its enumeration"
    return data["count"], per_thesis


def _forms(n: int) -> list[str]:
    return [rf"\b{n}\b", rf"\b{WORDS[n]}\b"] if n in WORDS else [rf"\b{n}\b"]


def _mentions(n: int, paragraph: str) -> bool:
    return any(re.search(f, paragraph, re.IGNORECASE) for f in _forms(n))


def _paragraphs(text: str) -> list[str]:
    # blank line separates paragraphs; blockquote markers and line wraps are noise
    return [re.sub(r"[\s>]+", " ", p) for p in re.split(r"\n\s*\n", text)]


def test_every_link_to_the_results_artifact_carries_its_count() -> None:
    artifact = _newest_artifact()
    total, per_thesis = _counts(artifact)
    newest_date = LINK.search(artifact.name).group(1)
    problems: list[str] = []
    seen = 0
    for doc in DOCS:
        for paragraph in _paragraphs(doc.read_text()):
            dates = LINK.findall(paragraph)
            if not dates:
                continue
            seen += 1
            for date in dates:
                if date != newest_date:
                    problems.append(f"{doc.name}: links results_against_thesis_{date}.json; newest is {newest_date}")
            whole = _mentions(total, paragraph)
            split = all(_mentions(k, paragraph) for k in per_thesis)
            if not (whole or split):
                problems.append(
                    f"{doc.name}: paragraph links the artifact ({total} results) without stating "
                    f"{total} or the split {per_thesis}: {paragraph[:120]!r}")
    assert seen >= len(DOCS), "the artifact is expected to be linked from every document checked"
    assert not problems, "\n".join(problems)


def test_no_document_states_a_different_total_beside_the_artifact() -> None:
    """A wrong count beside the link is worse than no count: 'seven results
    against my own theses' shipped once."""
    total, _ = _counts(_newest_artifact())
    pattern = re.compile(r"\b(\w+|\d+)\s+(?:preregistered\s+)?results against (?:my|its) own thes[ei]s", re.IGNORECASE)
    wrong: list[str] = []
    for doc in DOCS:
        for m in pattern.finditer(re.sub(r"[\s>]+", " ", doc.read_text())):
            said = m.group(1).lower()
            n = int(said) if said.isdigit() else next((k for k, w in WORDS.items() if w == said), None)
            if n is not None and n != total:
                wrong.append(f"{doc.name}: {m.group(0)!r} (artifact: {total})")
    assert not wrong, "\n".join(wrong)
