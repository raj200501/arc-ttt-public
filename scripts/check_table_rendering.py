#!/usr/bin/env python3
"""Does every published word actually reach the page?

`VERDICT.md` is the map of this project and its central promise is that
every result appears in the words frozen before it ran. That promise is
about what a reader *sees*, and a markdown table silently discards any
cell past its header's column count.

Four rows were doing exactly that. **2,621 characters of Addendum L's
amendment — its reasoning and its frozen readings — did not render on
GitHub at all**, and neither did 1,981 of Addendum N's, because a decided
row had been written onto the same line as the tail of the
preregistration row it replaced. The text was in the file, in git, in
every diff. It was not on the page. No test could have caught it because
every test read the file.

This is the check that reads the table the way the renderer does. It
also catches the reverse defect — a row with FEWER cells than its header,
which shifts every following column and puts an artifact path under
"command".

    PYTHONPATH=src python3 scripts/check_table_rendering.py
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# An unescaped pipe starts a new cell; `\|` is a literal bar.
CELL_SPLIT = re.compile(r"(?<!\\)\|")
SEPARATOR = re.compile(r"^[\s:\-]+$")

DEFAULT_PAGES = ("VERDICT.md", "README.md", "EVIDENCE.md", "CORRECTIONS.md",
                 "CHALLENGES.md", "RESULTS.md")


def cells(line: str) -> list[str]:
    parts = CELL_SPLIT.split(line)
    # A table row starts with a pipe, so the first fragment is empty and
    # is not a cell. The TRAILING pipe is optional in markdown, so only
    # drop the last fragment when it is actually empty -- dropping it
    # unconditionally under-counted a row that omits the closing pipe,
    # and this checker reported such a row as having one cell when it
    # had two. A checker that miscounts is a checker that will be
    # ignored.
    if parts and not parts[0].strip():
        parts = parts[1:]
    if len(parts) > 1 and parts[-1] == "":
        parts = parts[:-1]
    return parts


def problems(path: pathlib.Path) -> list[str]:
    found: list[str] = []
    header: int | None = None
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"),
                                  start=1):
        if not line.lstrip().startswith("|"):
            header = None
            continue
        row = cells(line)
        if header is None:
            header = len(row)
            continue
        if SEPARATOR.match("".join(row)):
            continue
        if len(row) > header:
            dropped = sum(len(c) for c in row[header:])
            found.append(
                f"{path.name}:{number} has {len(row)} cells against a "
                f"{header}-column header — {dropped} characters are DROPPED "
                f"by the renderer and never reach a reader. First hidden "
                f"cell: {row[header].strip()[:80]!r}")
        elif len(row) < header:
            found.append(
                f"{path.name}:{number} has {len(row)} cells against a "
                f"{header}-column header — the remaining columns shift, so "
                f"content renders under the wrong heading")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pages", nargs="*", default=None)
    args = parser.parse_args()

    pages = [pathlib.Path(p) for p in args.pages] if args.pages else [
        REPO / name for name in DEFAULT_PAGES]
    found: list[str] = []
    checked = 0
    for path in pages:
        if not path.exists():
            continue
        checked += 1
        found.extend(problems(path))

    if not checked:
        raise SystemExit("no pages to check — refusing to report success")
    if not found:
        print(f"{checked} page(s): every table row renders every cell")
        return 0
    print(f"{len(found)} table row(s) lose content when rendered:\n",
          file=sys.stderr)
    for row in found:
        print(f"  {row}", file=sys.stderr)
    print("\nA row with extra cells is usually two rows written onto one "
          "line. Split them, or escape an inline bar as \\| so the prose "
          "stays in one cell.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
