"""Every figure must still be the picture of the artifact it names.

`scripts/make_figures.py` writes `figures/<name>.json` beside each
figure: the artifacts it read, their sha256, and the exact values it
drew. These tests re-read the artifacts and re-derive the values with
the same stdlib extractors. If a hash or a value has moved, the figure
is stale and says something the artifact no longer says; the fix is to
re-run `python3 scripts/make_figures.py`, never to edit the JSON.

No matplotlib is needed here. Without a `figures/` directory (for
example in a tree that does not ship the figures) the figure checks skip.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "make_figures.py"
FIGURES = REPO / "figures"

_spec = importlib.util.spec_from_file_location("make_figures", SCRIPT)
assert _spec and _spec.loader
make_figures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_figures)


def _records() -> list[tuple[Path, dict]]:
    if not FIGURES.is_dir():
        pytest.skip("no figures/ directory in this tree")
    paths = sorted(FIGURES.glob("*.json"))
    if not paths:
        pytest.skip("figures/ holds no figure records")
    return [(p, json.loads(p.read_text(encoding="utf-8"))) for p in paths]


def test_every_figure_still_matches_its_artifacts() -> None:
    stale = []
    for path, record in _records():
        name = record["figure"]
        assert path.stem == name, f"{path.name} records figure {name!r}"
        assert name in make_figures.EXTRACTORS, f"{name}: no extractor in make_figures.py"
        # A changed hash alone is not staleness: anchored protocols gain dated
        # errata and artifacts gain fields. The figure is stale only when the
        # values it drew are no longer what its artifacts say.
        for art in record["artifacts"]:
            assert (REPO / art["path"]).exists(), f"{name}: {art['path']} is gone"
        values = make_figures.canonical(make_figures.EXTRACTORS[name]())
        if values != record["values"]:
            stale.append(f"{name}: the artifact values no longer equal the plotted values")
    assert not stale, ("stale figures; re-run python3 scripts/make_figures.py:\n  "
                       + "\n  ".join(stale))


def test_every_figure_record_names_files_that_exist() -> None:
    missing = [f for _, record in _records() for f in record["files"]
               if not (REPO / f).is_file()]
    assert not missing, f"figure files named in a record but absent: {missing}"


def test_the_prose_check_refuses_a_wrong_number(tmp_path, monkeypatch) -> None:
    (tmp_path / "doc.md").write_text("The mean was **-7.3** F1 at\n0.5B.", encoding="utf-8")
    monkeypatch.setattr(make_figures, "REPO", tmp_path)
    claim = {"doc": "doc.md", "quote": "The mean was **-7.3** F1 at 0.5B."}
    make_figures.check_prose([{**claim, "numbers": [("-7.3", -7.27, "mean")]}])
    with pytest.raises(SystemExit, match="prose says '-7.3'"):
        make_figures.check_prose([{**claim, "numbers": [("-7.3", -7.36, "mean")]}])
    with pytest.raises(SystemExit, match="quote no longer present"):
        make_figures.check_prose([{"doc": "doc.md", "quote": "The mean was -7.4",
                                   "numbers": []}])
    # Private pitch copy: checked where it exists, never quoted in a record,
    # counted (not failed) where the public export has removed it.
    private = make_figures.PRIVATE_PREFIX + "notes.md"
    out = make_figures.check_prose([{"doc": private, "quote": "x", "numbers": []}])
    assert out["private_quotes"] == {"checked": 0, "absent_from_this_tree": 1}
    (tmp_path / make_figures.PRIVATE_PREFIX).mkdir(parents=True)
    (tmp_path / private).write_text("The mean was -7.3 F1.", encoding="utf-8")
    out = make_figures.check_prose([{"doc": private, "quote": "The mean was -7.3",
                                     "numbers": [("-7.3", -7.27, "mean")]}])
    assert out == {"public": [], "private_quotes": {"checked": 1, "absent_from_this_tree": 0}}
    with pytest.raises(SystemExit, match="prose says"):
        make_figures.check_prose([{"doc": private, "quote": "The mean was -7.3",
                                   "numbers": [("-7.3", -7.4, "mean")]}])


def test_display_rounding_matches_the_prose() -> None:
    # 0.1125 x 100 formats as '11.2' with Python floats; the prose says 11.3.
    assert make_figures.fmt(0.1125, 100, 1) == "11.3"
    assert make_figures.signed(-0.0727, 100) == "−7.3"
    assert make_figures.signed(0.05, 100, 0) == "+5"
