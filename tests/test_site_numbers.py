"""docs/site/index.html: every number on the page appears in the file it links to.

The page is the short public description of fencecheck. Its rule is the
repository's rule: a number appears only if a banked file holds it, and
the page quotes, it does not compute. This test holds the page to that.

For every token in the page's visible text that contains a digit:

* it must sit inside an element that names its source: the nearest
  enclosing <a> whose href is a blob link into the public repository, or
  the nearest enclosing element with data-src, which must itself contain
  a link to that same file. The token must appear in that file. A bare
  number or a date needs its own <a>: data-src covers only names, such
  as a model name, file:line or a command, inside a quoted block;
* it must sit inside an element whose data-quote names one of the four
  documents the page may quote from, and appear in that document too.

"Appear" means: an integer appears as a whole number, not inside a
longer one; a decimal appears as written, or as a number in the file
with the same value at the precision shown (0.6350 on the page matches
0.635 in a JSON artifact); anything else with a digit in it (a date, a
model name, file:line, a URL) appears verbatim, ignoring case and
treating - and _ as the same character, so the label micro-F1 matches
the key micro_f1 in an artifact. A token that is part of
the path its own link points to (a file name shown as link text) needs
nothing more, and a raw.githubusercontent.com URL must name a file that
exists in this tree.

Limits, stated so this test is not read as more than it is: it checks
presence, not meaning. It cannot tell whether 64 in an artifact is the
count the sentence is about, and a small integer appears in almost any
file. Blocks marked data-verbatim (the CI snippets, the rebuild command)
are checked character for character, which is the strong form.

Also checked: every blob link names a file in this tree, the page makes
no external request (no script, stylesheet, image, font, frame or
url()), and, where the private leak gate exists, it passes on the page.
Standard library only; no model libraries.
"""

from __future__ import annotations

import re
import subprocess
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "docs" / "site" / "index.html"
REPO_URL = "https://github.com/raj200501/arc-ttt-public"
BLOB = REPO_URL + "/blob/main/"
RAW = "https://raw.githubusercontent.com/raj200501/arc-ttt-public/main/"
QUOTABLE = {"README.md", "VERDICT.md", "CORRECTIONS.md", "tools/README.md"}
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "source", "track", "wbr"}

TOKEN = re.compile(r"[A-Za-z0-9_.\-/:@+%,]*\d[A-Za-z0-9_.\-/:@+%,]*")
NUMBER = re.compile(r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
DECIMAL_IN_FILE = re.compile(r"(?<![\d.])\d+\.\d+(?![\d.])")
EXTERNAL = re.compile(
    r"<script|<link\b|<img\b|<iframe|<object|<embed|@import|@font-face|url\(|"
    r"(?<![\w-])src\s*=", re.I)


def blob_path(href: str) -> str | None:
    if href.startswith(BLOB):
        return href[len(BLOB):].split("#", 1)[0]
    return None


class _Page(HTMLParser):
    """Collects digit tokens with their source and quote, and verbatim blocks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, dict, dict]] = []
        self.tokens: list[tuple[str, tuple[str, str] | None, str | None]] = []
        self.verbatim: list[tuple[str, str]] = []
        self.hrefs: list[str] = []
        self.unlinked_src: list[str] = []
        self.quotes: list[str] = []
        self.srcs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        a = {k: (v or "") for k, v in attrs}
        if "data-quote" in a:
            self.quotes.append(a["data-quote"])
        if "data-src" in a:
            self.srcs.append(a["data-src"])
        if tag == "a" and "href" in a:
            self.hrefs.append(a["href"])
            path = blob_path(a["href"])
            for _, attrs_up, rec in self.stack:
                if path is not None and attrs_up.get("data-src") == path:
                    rec["linked"] = True
        if tag in VOID:
            return
        self.stack.append((tag, a, {"text": [], "linked": False}))

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID:
            return
        while self.stack:
            name, a, rec = self.stack.pop()
            if "data-verbatim" in a:
                self.verbatim.append((a["data-verbatim"], "".join(rec["text"])))
            if "data-src" in a and not rec["linked"]:
                self.unlinked_src.append(a["data-src"])
            if name == tag:
                break

    def handle_data(self, data: str) -> None:
        if any(name in ("style", "script") for name, _, _ in self.stack):
            return
        for _, _, rec in self.stack:
            rec["text"].append(data)
        for match in TOKEN.finditer(data):
            token = match.group(0).strip(".,:;")
            if any(ch.isdigit() for ch in token):
                self.tokens.append((token, self._source(), self._quote()))

    def _source(self) -> tuple[str, str] | None:
        for name, a, _ in reversed(self.stack):
            if name == "a" and blob_path(a.get("href", "")) is not None:
                return ("link", blob_path(a["href"]) or "")
            if "data-src" in a:
                return ("src", a["data-src"])
        return None

    def _quote(self) -> str | None:
        for _, a, _ in reversed(self.stack):
            if "data-quote" in a:
                return a["data-quote"]
        return None


def appears(token: str, text: str) -> bool:
    """True if the token is present in text, in the sense the docstring gives."""
    if NUMBER.fullmatch(token):
        core = token.rstrip("%")
        for form in {core, core.replace(",", "")}:
            if re.search(r"(?<![\d.,])" + re.escape(form) + r"(?!\d|[.,]\d)", text):
                return True
        if "." in core:
            plain = core.replace(",", "")
            want = Decimal(plain)
            half = Decimal(5).scaleb(-(len(plain.split(".")[1]) + 1))
            return any(abs(Decimal(m.group(0)) - want) <= half
                       for m in DECIMAL_IN_FILE.finditer(text))
        return False
    return _name(token) in _name(text)


def _name(text: str) -> str:
    return text.lower().replace("-", "_")


_cache: dict[str, str] = {}


def _read(rel: str) -> str:
    if rel not in _cache:
        _cache[rel] = (REPO / rel).read_text(encoding="utf-8")
    return _cache[rel]


def problems(html: str) -> list[str]:
    """Every way the page breaks its own rule; empty when it holds."""
    page = _Page()
    page.feed(html)
    page.close()
    found: list[str] = []

    for href in page.hrefs:
        path = blob_path(href)
        if path is not None:
            if not (REPO / path).is_file():
                found.append(f"link to a file not in this tree: {path}")
        elif not (href.startswith("#") or href.rstrip("/") == REPO_URL):
            found.append(f"link outside the public repository: {href}")
    for rel in page.srcs:
        if not (REPO / rel).is_file():
            found.append(f"data-src names a file not in this tree: {rel}")
    for rel in page.unlinked_src:
        found.append(f"data-src={rel} has no link to {rel} inside it")
    for rel in page.quotes:
        if rel not in QUOTABLE:
            found.append(f"data-quote={rel} is not one of {sorted(QUOTABLE)}")

    for token, source, quote in page.tokens:
        if token.startswith(RAW):
            if not (REPO / token[len(RAW):]).is_file():
                found.append(f"{token}: no such file in this tree")
            continue
        if source is None:
            found.append(f"{token!r} has no source link")
            continue
        kind, rel = source
        if kind == "link" and token in rel:
            continue  # a file name shown as its own link text
        if kind != "link" and (NUMBER.fullmatch(token) or DATE.fullmatch(token)):
            found.append(f"{token!r} is a number with no link of its own (only data-src={rel})")
        if not (REPO / rel).is_file():
            continue  # already reported above
        if not appears(token, _read(rel)):
            found.append(f"{token!r} does not appear in {rel}, the file it links to")
        if quote is None:
            found.append(f"{token!r} is not inside a data-quote element")
        elif quote in QUOTABLE and not appears(token, _read(quote)):
            found.append(f"{token!r} does not appear in {quote}, the document it quotes")

    for rel, text in page.verbatim:
        if rel not in QUOTABLE:
            found.append(f"data-verbatim={rel} is not a quotable document")
        elif text.strip("\n") not in _read(rel):
            found.append(f"a data-verbatim block is not character for character in {rel}:\n"
                         + text)

    if EXTERNAL.search(html):
        found.append(f"external request or script: {EXTERNAL.search(html).group(0)!r}")
    return found


def test_every_number_appears_in_the_file_it_links_to() -> None:
    found = problems(PAGE.read_text(encoding="utf-8"))
    assert not found, "docs/site/index.html breaks its own rule:\n  " + "\n  ".join(found)


def test_the_check_sees_the_headline_numbers() -> None:
    """A checker that finds no tokens passes every page; make sure it finds these."""
    page = _Page()
    page.feed(PAGE.read_text(encoding="utf-8"))
    tokens = {token for token, _, _ in page.tokens}
    assert {"0.0000", "0.8958", "30", "64", "63", "0.2152"} <= tokens
    assert len(page.verbatim) >= 3


@pytest.mark.parametrize("old, new", [
    ("0.8958</a>", "0.8959</a>"),                      # a number its artifact does not hold
    (">63</a>", ">62</a>"),                            # a census count changed
    ("actions/checkout@v4", "actions/checkout@v3"),    # a CI snippet edited
    ("<footer>", '<footer><p data-quote="README.md">17 packages</p>'),  # no source
    ("<footer>", '<footer><p><a href="' + BLOB + 'README.md">0.4035</a></p>'),  # no quote
    ("<footer>", '<footer><p data-quote="README.md" data-src="README.md">0.4035 <a href="'
     + BLOB + 'README.md">README.md</a></p>'),  # a number not linked itself
])
def test_a_planted_error_is_caught(old: str, new: str) -> None:
    html = PAGE.read_text(encoding="utf-8")
    assert old in html
    assert problems(html.replace(old, new, 1)), f"planting {new!r} went unnoticed"


def test_a_value_matches_at_the_precision_shown() -> None:
    assert appears("0.6350", '"mean_fence_stripped": 0.635,')
    assert appears("0.0000", '"mean_as_is": 0.0,')
    assert not appears("0.8958", '"mean_fence_stripped": 0.9,')
    assert not appears("30", "300 documents and 2.30 seconds")
    assert appears("1,823", "1,823 outputs")


GATE = REPO / "scripts" / "leak_gate.sh"


@pytest.mark.skipif(not GATE.exists(), reason="private-only gate script")
def test_the_leak_gate_passes_on_the_site() -> None:
    result = subprocess.run(["bash", str(GATE), str(PAGE.parent)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
