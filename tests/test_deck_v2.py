"""DECK_V2.md and its rendered HTML: what makes the slides checkable.

`tests/test_reconcile_outbound.py` checks that every distinctive figure on
the slides traces to an artifact. These tests hold the rest of the
contract the slide file states in its preamble:

* the HTML is rendered from the Markdown and never edited by hand, so a
  number cannot reach the HTML without passing the gate on the Markdown;
* the HTML makes no network request and its figures are the committed
  ones;
* every path a Sources line cites exists;
* the counts that move (the corrections ledger) are current, and the
  authorship counts are the ones in the banked ledger they cite, scoped
  to the commit it names;
* the founder's placeholders stay unfilled, the Ask names no amount, and
  no use-of-funds row finances the closed adaptation program;
* the superseded slide document says it is superseded, on its first line.

Skipped where docs/strategy is absent: the slide documents are internal
and not part of the public export.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
STRATEGY = REPO / "docs" / "strategy"
SOURCE = STRATEGY / "DECK_V2.md"
HTML = STRATEGY / "deck_v2.html"
RENDERER = STRATEGY / "render_deck_v2.py"

pytestmark = pytest.mark.skipif(
    not SOURCE.exists(), reason="DECK_V2.md is internal and not in this cut")

SUPERSEDED_LINE = ("SUPERSEDED on 2026-10-21 by DECK_V2.md / deck_v2.html; "
                   "kept as the record of what was sent.")


def _renderer():
    spec = importlib.util.spec_from_file_location("_render_deck_v2", RENDERER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _slides() -> dict[int, str]:
    text = SOURCE.read_text(encoding="utf-8")
    parts = re.split(r"^## Slide (\d+): .*$", text, flags=re.MULTILINE)
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}


def test_the_html_is_the_rendered_markdown() -> None:
    """A hand edit to the HTML would put a number outside the gate."""
    rendered = _renderer().render(SOURCE.read_text(encoding="utf-8"))
    assert HTML.read_text(encoding="utf-8") == rendered, (
        "deck_v2.html differs from a fresh render of DECK_V2.md; re-run "
        "python3 docs/strategy/render_deck_v2.py and never edit the HTML")


def test_ten_to_twelve_slides_numbered_in_order() -> None:
    numbers = sorted(_slides())
    assert numbers == list(range(1, len(numbers) + 1))
    assert 10 <= len(numbers) <= 12
    assert HTML.read_text(encoding="utf-8").count('<section class="slide') == len(numbers)


def test_the_html_makes_no_network_request() -> None:
    html = HTML.read_text(encoding="utf-8")
    assert not re.search(r"""(?:src|href)\s*=\s*["']?(?:https?:)?//""", html)
    assert "@import" not in html
    assert not re.search(r"url\(\s*['\"]?(?:https?:)?//", html)
    assert "<link" not in html.lower()
    assert "<img" not in html.lower()


def test_every_figure_is_a_committed_figure() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    cited = re.findall(r"^!\[[^\]]*\]\(([^)\s]+)\)$", text, flags=re.MULTILINE)
    assert cited, "no figure on any slide"
    for rel in cited:
        path = (STRATEGY / rel).resolve()
        assert path.parent == REPO / "figures" and path.suffix == ".svg", rel
        assert path.exists(), rel
    html = HTML.read_text(encoding="utf-8")
    assert html.count('class="figsvg"') == len(cited)
    # ids are prefixed per figure, so no two inlined figures share one
    ids = re.findall(r'\bid="([^"]+)"', html)
    assert len(ids) == len(set(ids)), "two inlined figures share an element id"


def _tracked() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z", "--", ".", ":(exclude)work"],
        capture_output=True, text=True, check=True).stdout
    return [p for p in out.split("\0") if p]


def test_every_cited_source_exists() -> None:
    """A footnote that names a file nobody can open is not a source."""
    tracked = _tracked()
    missing = []
    for number, body in _slides().items():
        sources = [b for b in re.split(r"\n\s*\n", body) if b.startswith("Sources:")]
        assert sources, f"slide {number} has no Sources line"
        for token in re.findall(r"`([^`]+)`", " ".join(sources)):
            if "/" not in token and not re.search(r"\.(?:md|json|py|txt|svg)$", token):
                continue  # a SHA or a branch name, not a path
            suffix = token[4:] if token.startswith(".../") else token
            if not any(p == suffix or p.endswith("/" + suffix) for p in tracked):
                missing.append(f"slide {number}: {token}")
    assert not missing, f"cited sources that do not exist: {missing}"


def test_the_corrections_counts_are_current(tmp_path: pathlib.Path) -> None:
    """Reconciling is not being current: last week's count still reconciles."""
    out = tmp_path / "ledger.json"
    subprocess.run([sys.executable, str(REPO / "scripts" / "self_correction_ledger.py"),
                    "--out", str(out)], capture_output=True, text=True,
                   cwd=REPO, timeout=120, check=True)
    ledger = json.loads(out.read_text(encoding="utf-8"))
    slide = " ".join(_slides()[7].split())
    total, outward = ledger["total_dated_corrections"], ledger["outward_facing"]
    expected = [f"{total} dated corrections", f"> {total} |",
                f"{outward} of them reached, or would have reached", f"> {outward} |",
                f"from {ledger['first']} to {ledger['latest']}"]
    by_section = ledger["by_section"]
    expected += [f"{by_section['Claims withdrawn']} claims withdrawn",
                 f"{by_section['Overstatements corrected']} overstatements corrected",
                 f"{by_section['Numbers corrected']} numbers corrected",
                 f"{by_section['Defects found in our own verification path']} defects"]
    stale = [e for e in expected if e not in slide]
    assert not stale, (f"slide 7 no longer matches the ledger: {stale}. Update "
                       "DECK_V2.md, then re-render the HTML")


def test_the_authorship_counts_are_the_banked_ledger() -> None:
    """Slide 11 quotes a banked run of scripts/authorship_ledger.py, scoped
    to the commit it names. A running commit count rots on every commit
    (scripts/currency_gate.py bans it), so the slide carries a dated one."""
    ledger = json.loads((REPO / "experiments" / "authorship_ledger_2026-09-16.json")
                        .read_text(encoding="utf-8"))
    slide = " ".join(_slides()[11].split())
    head = ledger["head"][:7]
    total, agent, human = (ledger["total_commits"], ledger["agent_commits"],
                           ledger["human_commits"])
    assert f"As of `{head}` on {ledger['date']}, {agent} of the {total} commits" in slide
    assert f"agent share {ledger['agent_share']}" in slide
    assert f"The other {human} carried no agent mark as of that commit" in slide
    assert f"> {agent} of {total} |" in slide and f"> {human} |" in slide


def test_placeholders_stay_unfilled_and_the_ask_names_no_amount() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    assert text.count("[FOUNDER:") >= 12, (
        "a founder placeholder was filled in by someone other than the founder")
    ask = _slides()[12]
    assert ask.count("[FOUNDER: amount]") >= 7
    assert not re.search(r"\$\s?\d", ask), "the Ask names an amount; that is the founder's"


def test_no_use_of_funds_row_finances_the_closed_thesis() -> None:
    """Round 9: the old Ask financed the thesis the gates had closed."""
    ask = _slides()[12]
    rows = [line for line in ask.split("\n")
            if line.startswith("|") and not line.startswith("|---")][1:]
    assert len(rows) >= 4
    closed = re.compile(r"adapt|lora|rung|base model|fine-tun|per-tenant", re.I)
    funded = [r for r in rows if closed.search(r)]
    assert not funded, f"a use-of-funds row finances the closed program: {funded}"
    assert "Not funded: anything for the adaptation program" in ask


def test_no_em_dashes_in_the_slides() -> None:
    assert "—" not in SOURCE.read_text(encoding="utf-8")


def test_the_superseded_documents_say_so_on_line_one() -> None:
    for name in ("DE" + "CK.md", "deck.html"):
        first = (STRATEGY / name).read_text(encoding="utf-8").split("\n", 1)[0]
        assert first == SUPERSEDED_LINE, name
