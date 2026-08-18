"""Foundation tests: loader validation, scoring, augmentation round-trips."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arcttt.augment import (
    DIHEDRAL_SWEEP,
    IDENTITY,
    Augmentation,
    deterministic_palettes,
    rotate90,
)
from arcttt.tasks import Grid, TaskFormatError, load_task, score_attempts, to_grid

GRID: Grid = ((1, 2, 3), (4, 5, 6))


def write_task(path: Path, payload: dict) -> Path:
    file = path / "t001.json"
    file.write_text(json.dumps(payload))
    return file


def valid_payload() -> dict:
    return {
        "train": [{"input": [[1, 2], [3, 4]], "output": [[1]]}],
        "test": [{"input": [[5, 6]], "output": [[2]]}],
    }


def test_load_task_roundtrips_valid_schema(tmp_path: Path) -> None:
    task = load_task(write_task(tmp_path, valid_payload()))
    assert task.task_id == "t001"
    assert task.train[0].output == ((1,),)
    assert task.test[0].input == ((5, 6),)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda p: p.pop("test"),
        lambda p: p["train"][0].pop("output"),
        lambda p: p["train"][0]["input"].append([1]),  # ragged row width
        lambda p: p["train"][0]["input"][0].__setitem__(0, 11),  # bad color
        lambda p: p.__setitem__("train", []),
    ),
)
def test_load_task_fails_closed_on_malformed_input(tmp_path: Path, mutate) -> None:
    payload = valid_payload()
    mutate(payload)
    with pytest.raises(TaskFormatError):
        load_task(write_task(tmp_path, payload))


def test_grid_size_cap_enforced() -> None:
    with pytest.raises(TaskFormatError, match="exceeds"):
        to_grid([[0] * 31])


def test_scoring_is_exact_match_within_two_attempts() -> None:
    solution: Grid = ((1, 2), (3, 4))
    wrong: Grid = ((0, 0), (0, 0))
    assert score_attempts([solution], solution)
    assert score_attempts([wrong, solution], solution)
    assert not score_attempts([wrong, wrong, solution], solution)  # third attempt ignored
    assert not score_attempts([], solution)


def test_rotate90_matches_known_result() -> None:
    assert rotate90(GRID) == ((4, 1), (5, 2), (6, 3))


def test_dihedral_sweep_has_eight_distinct_elements() -> None:
    images = {augmentation.apply(GRID) for augmentation in DIHEDRAL_SWEEP}
    assert len(DIHEDRAL_SWEEP) == 8
    assert len(images) == 8  # non-square grid: all eight images distinct


@pytest.mark.parametrize("augmentation", DIHEDRAL_SWEEP)
def test_dihedral_round_trip_identity(augmentation: Augmentation) -> None:
    assert augmentation.invert(augmentation.apply(GRID)) == GRID


def test_palette_round_trip_and_background_fixed() -> None:
    for palette in deterministic_palettes(seed=7, count=5):
        assert palette[0] == 0
        augmentation = Augmentation(palette=palette)
        augmentation.validate()
        assert augmentation.invert(augmentation.apply(GRID)) == GRID


def test_palettes_are_seed_deterministic() -> None:
    assert deterministic_palettes(3, 4) == deterministic_palettes(3, 4)
    assert deterministic_palettes(3, 4) != deterministic_palettes(4, 4)


def test_invalid_augmentation_rejected() -> None:
    with pytest.raises(ValueError):
        Augmentation(palette=(0,) * 10).validate()


def test_task_level_augmentation_preserves_structure(tmp_path: Path) -> None:
    task = load_task(write_task(tmp_path, valid_payload()))
    transformed = DIHEDRAL_SWEEP[3].apply_task(task)
    assert transformed.task_id == task.task_id
    assert len(transformed.train) == len(task.train)
    restored = DIHEDRAL_SWEEP[3].apply_task(transformed)  # not inverse; just structural
    assert restored.train[0].output is not None
    assert IDENTITY.apply_task(task).train[0].input == task.train[0].input


def test_expanded_sweep_crosses_dihedral_with_palettes() -> None:
    from arcttt.augment import expanded_sweep

    sweep = expanded_sweep(seed=7, palettes_per_element=2)
    # 8 dihedral elements x (identity palette + 2 seeded permutations)
    assert len(sweep) == 24
    for augmentation in sweep:
        augmentation.validate()
        assert augmentation.invert(augmentation.apply(GRID)) == GRID
    # every dihedral element keeps one identity-palette representative
    identity_palette = tuple(range(10))
    identity_frames = [a for a in sweep if a.palette == identity_palette]
    assert len(identity_frames) == 8
    assert len(set(sweep)) == len(sweep)  # no duplicate augmentations
    # deterministic under the same seed
    assert expanded_sweep(seed=7, palettes_per_element=2) == sweep


def test_example_shuffle_is_deterministic_and_preserves_pairs() -> None:
    from arcttt.serialize import ttt_training_examples
    from arcttt.tasks import Pair, Task

    grids = [((i,),) for i in range(1, 5)]
    task = Task(
        task_id="t",
        train=tuple(Pair(input=g, output=g) for g in grids),
        test=(Pair(input=((0,),), output=None),),
    )
    unshuffled = ttt_training_examples(task)
    shuffled = ttt_training_examples(task, shuffle_seed=3)
    assert ttt_training_examples(task, shuffle_seed=3) == shuffled  # deterministic
    assert len(shuffled) == len(unshuffled) == 4
    for base, permuted in zip(unshuffled, shuffled):
        # same held-out target (final user/assistant turns)
        assert base[-2:] == permuted[-2:]
        # context is a permutation of the same user/assistant pairs
        base_pairs = {(base[i], base[i + 1]) for i in range(0, len(base) - 2, 2)}
        perm_pairs = {
            (permuted[i], permuted[i + 1]) for i in range(0, len(permuted) - 2, 2)
        }
        assert base_pairs == perm_pairs
    assert shuffled != unshuffled  # at least one context order actually moved
