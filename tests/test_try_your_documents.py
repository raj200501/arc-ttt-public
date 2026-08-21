"""The 'run it on your own documents' path, minus the model.

Everything a reader could run verified OUR numbers on OUR data. This
script is the first one that points the machinery at theirs, so its
arithmetic and its honesty text are worth pinning: a paired delta it
prints could otherwise be mistaken for a gate result, and it is not one
-- not blind, no preregistered bar, one corpus.

The adaptation arms need torch and minutes of compute; the split, the
scoring and the report do not, and those are what is tested here.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "try_your_documents.py"


def _module():
    spec = importlib.util.spec_from_file_location("try_your_documents", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                    encoding="utf-8")
    return path


def test_a_perfect_arm_scores_one_and_a_wrong_arm_does_not(tmp_path) -> None:
    module = _module()
    gold = _write(tmp_path / "gold.jsonl", [
        {"id": "a", "gold": {"vendor": "Acme", "total": "10.00"}},
        {"id": "b", "gold": {"vendor": "Globex", "total": "20.00"}},
    ])
    perfect = _write(tmp_path / "perfect.jsonl", [
        {"id": "a", "prediction": {"vendor": "Acme", "total": "10.00"}},
        {"id": "b", "prediction": {"vendor": "Globex", "total": "20.00"}},
    ])
    wrong = _write(tmp_path / "wrong.jsonl", [
        {"id": "a", "prediction": {"vendor": "WRONG", "total": "10.00"}},
        {"id": "b", "prediction": {"vendor": "WRONG", "total": "WRONG"}},
    ])
    assert module.score_arm(perfect, gold) == {"a": 1.0, "b": 1.0}
    scored = module.score_arm(wrong, gold)
    assert 0.0 <= scored["b"] < scored["a"] < 1.0


def test_a_missing_prediction_scores_zero_rather_than_being_skipped(
        tmp_path) -> None:
    """Dropping a hard document must not quietly raise the mean."""
    module = _module()
    gold = _write(tmp_path / "gold.jsonl", [
        {"id": "a", "gold": {"vendor": "Acme"}},
        {"id": "b", "gold": {"vendor": "Globex"}},
    ])
    partial = _write(tmp_path / "pred.jsonl",
                     [{"id": "a", "prediction": {"vendor": "Acme"}}])
    scored = module.score_arm(partial, gold)
    assert scored == {"a": 1.0, "b": 0.0}


def test_sign_test_drops_ties_and_matches_a_known_p_value() -> None:
    module = _module()
    wins, losses, ties, p = module.sign_test([0.1, 0.2, 0.0, -0.1, 0.0])
    assert (wins, losses, ties) == (2, 1, 2)
    # one-sided P(X >= 2) with n=3 fair coins = (3 + 1) / 8
    assert p == pytest.approx(0.5)

    _, _, _, all_wins = module.sign_test([0.1] * 10)
    assert all_wins == pytest.approx(1 / 1024)


def test_sign_test_survives_an_all_tie_arm() -> None:
    module = _module()
    assert module.sign_test([0.0, 0.0])[3] == 1.0


def test_the_script_says_what_it_is_not(tmp_path) -> None:
    """The disclaimers are load-bearing, so they are pinned like a number.

    A paired delta printed by a script that reads your gold, fixes no bar
    in advance, and can be re-run until the number is nice is NOT the
    thing VERDICT.md's rows are. If that ever stops being said out loud,
    this script starts manufacturing numbers people mistake for gates.
    """
    text = SCRIPT.read_text(encoding="utf-8")
    for claim in ("It is NOT blind", "NO preregistered bar",
                  "a data point, not a benchmark"):
        assert claim in text, f"the honesty text lost: {claim!r}"
    assert "CHALLENGE_TERMS.md" in text, (
        "the script must point at the version that IS evidence")


def test_a_negative_result_is_reported_not_buried() -> None:
    """If adaptation loses on the reader's data, the script says so."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "mean_adapted - mean_base <= 0" in text, (
        "no branch handles a delta that goes against us")
    assert "we will publish" in text
    assert "than after a purchase" in text


def test_a_saturated_baseline_tells_the_reader_they_do_not_need_this() -> None:
    """The most useful answer for some readers is 'this is not for you'.

    Found by running the script: on easy documents both arms scored
    1.0000, and the only branch that fired was the generic 'delta <= 0'
    message. A tie at the ceiling is not a loss -- it means plain
    prompting already solves their extraction and there is no headroom
    to sell into. Saying anything else there would be selling.
    """
    text = SCRIPT.read_text(encoding="utf-8")
    assert "mean_base >= 0.98" in text, (
        "no branch distinguishes a saturated baseline from a real loss")
    assert "nothing" in text and "for you to buy" in text


def test_help_runs_without_torch_installed() -> None:
    """--help must work on a machine that cannot load the model."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, cwd=REPO, timeout=120,
        env={"PYTHONPATH": str(REPO / "tests" / "_no_torch"), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert "--docs" in result.stdout
