"""Published text that does not reach the page is not published.

Every other check in this repository reads the file. A markdown renderer
does not: it discards any table cell past the header's column count. Four
rows in `VERDICT.md` were losing content that way, including **2,621
characters of Addendum L's amendment — its reasoning and its frozen
readings** — and 1,981 of Addendum N's. The text was in the file, in git,
and in every diff. It was never on the page.

That is a direct hit on this project's central promise, which is not
"the frozen words exist" but "the frozen words are what you read".
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "check_table_rendering.py"

pytestmark = pytest.mark.skipif(
    not SCRIPT.exists(), reason="the renderer check is not in this cut")

_spec = importlib.util.spec_from_file_location("check_table_rendering", SCRIPT)
check = importlib.util.module_from_spec(_spec)
sys.modules["check_table_rendering"] = check
_spec.loader.exec_module(check)


def test_every_published_table_row_renders_every_cell() -> None:
    found = []
    for name in check.DEFAULT_PAGES:
        path = REPO / name
        if path.exists():
            found.extend(check.problems(path))
    assert not found, (
        "table rows lose content when rendered:\n  " + "\n  ".join(found)
        + "\n\nRun: python3 scripts/check_table_rendering.py")


def test_it_catches_a_row_with_an_extra_cell(tmp_path: pathlib.Path) -> None:
    """MUTATION TEST: the defect that actually shipped.

    Two rows written onto one line. The renderer shows the first four
    cells and silently drops the rest.
    """
    page = tmp_path / "page.md"
    page.write_text(
        "| claim | verdict | artifact | command |\n"
        "|---|---|---|---|\n"
        "| a | b | c | d | THIS TEXT NEVER RENDERS | e |\n",
        encoding="utf-8")
    found = check.problems(page)
    assert found, "an over-filled row was passed as fine"
    assert "DROPPED" in found[0]
    assert "THIS TEXT NEVER RENDERS" in found[0]


def test_it_catches_a_row_with_a_missing_cell(tmp_path: pathlib.Path) -> None:
    """The reverse defect shifts every following column, so an artifact
    path renders under the heading "command"."""
    page = tmp_path / "page.md"
    page.write_text(
        "| claim | verdict | artifact | command |\n"
        "|---|---|---|---|\n"
        "| a | b |\n",
        encoding="utf-8")
    found = check.problems(page)
    assert found and "shift" in found[0]


def test_an_escaped_bar_does_not_start_a_new_cell(
        tmp_path: pathlib.Path) -> None:
    r"""`\|` is a literal bar in prose, not a column boundary.

    Escaping is the fix applied to two rows whose verdict prose contained
    an inline bar, so the checker must agree with the renderer about it
    or it will demand a "fix" that breaks correct rows.
    """
    page = tmp_path / "page.md"
    page.write_text(
        "| claim | verdict | artifact | command |\n"
        "|---|---|---|---|\n"
        "| a | b \\| still b | c | d |\n",
        encoding="utf-8")
    assert check.problems(page) == []


def test_a_row_without_a_trailing_pipe_is_counted_correctly(
        tmp_path: pathlib.Path) -> None:
    """The closing pipe is optional in markdown.

    The first version of this checker dropped the last fragment
    unconditionally and reported a complete two-cell row as having one
    cell. A checker that miscounts gets ignored, which is the same
    outcome as not having one.
    """
    page = tmp_path / "page.md"
    page.write_text(
        "| claim | verdict |\n"
        "|---|---|\n"
        "| a | b\n",
        encoding="utf-8")
    assert check.problems(page) == []


def test_it_refuses_to_report_success_over_nothing(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A check that finds no pages must not print a clean bill of health."""
    monkeypatch.setattr(sys, "argv",
                        ["check_table_rendering.py", str(tmp_path / "no.md")])
    with pytest.raises(SystemExit) as caught:
        check.main()
    assert "refusing to report success" in str(caught.value)
