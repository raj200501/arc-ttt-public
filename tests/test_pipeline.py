"""Serializer and voting-pipeline tests — offline, no model required."""

from __future__ import annotations

import math

import pytest

from arcttt.augment import DIHEDRAL_SWEEP, IDENTITY, Augmentation
from arcttt.serialize import (
    ChatTurn,
    grid_to_text,
    task_to_messages,
    text_to_grid,
    ttt_training_examples,
)
from arcttt.tasks import Grid, Pair, Task, TaskFormatError
from arcttt.vote import Candidate, pool_predictions, rescore_candidates, select_attempts

GRID: Grid = ((1, 0), (2, 3))


def make_task(demos: int = 3) -> Task:
    pairs = tuple(
        Pair(input=((i, i), (i, i)), output=((i + 1, i + 1), (i + 1, i + 1)))
        for i in range(demos)
    )
    return Task(task_id="t", train=pairs, test=(Pair(input=GRID, output=None),))


def test_grid_text_round_trip() -> None:
    assert text_to_grid(grid_to_text(GRID)) == GRID
    assert grid_to_text(GRID) == "10\n23"


def test_text_to_grid_rejects_junk() -> None:
    with pytest.raises(TaskFormatError):
        text_to_grid("12\nab")


def test_task_to_messages_shape_and_final_turn() -> None:
    task = make_task(2)
    turns = task_to_messages(task)
    assert len(turns) == 5  # 2 demos * 2 turns + test input
    assert [t.role for t in turns] == ["user", "assistant", "user", "assistant", "user"]
    assert turns[-1] == ChatTurn("user", grid_to_text(GRID))
    with pytest.raises(TaskFormatError):
        task_to_messages(task, test_index=1)


def test_ttt_examples_are_leave_one_out() -> None:
    task = make_task(3)
    examples = ttt_training_examples(task)
    assert len(examples) == 3
    for example in examples:
        assert len(example) == 6  # 2 context demos * 2 + held-out pair * 2
        assert example[-1].role == "assistant"
    held_out_targets = {example[-1].content for example in examples}
    assert len(held_out_targets) == 3  # each demo held out exactly once


def test_pool_predictions_inverts_before_counting() -> None:
    rotated = DIHEDRAL_SWEEP[1]  # rotations=1
    predictions = [
        (IDENTITY, GRID),
        (rotated, rotated.apply(GRID)),  # same answer seen through the augmentation
        (IDENTITY, ((9, 9), (9, 9))),
    ]
    counts = pool_predictions(predictions)
    assert counts[GRID] == 2
    assert counts[((9, 9), (9, 9))] == 1


def test_rescoring_uses_identical_augmentation_set() -> None:
    counts = {GRID: 2}
    seen: list[Augmentation] = []

    def log_probability(grid: Grid, augmentation: Augmentation) -> float:
        seen.append(augmentation)
        return -1.0

    candidates = rescore_candidates(counts, DIHEDRAL_SWEEP, log_probability)
    assert len(seen) == len(DIHEDRAL_SWEEP)
    assert candidates[0].mean_log_probability == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        rescore_candidates(counts, (), log_probability)


def test_selection_count_dominates_probability() -> None:
    common = Candidate(grid=GRID, found_count=3, mean_log_probability=-5.0)
    likely = Candidate(grid=((9,),), found_count=2, mean_log_probability=-0.01)
    rare = Candidate(grid=((8,),), found_count=1, mean_log_probability=0.0)
    attempts = select_attempts([rare, likely, common])
    assert attempts == (GRID, ((9,),))  # count wins; probability only breaks near-ties


def test_selection_probability_breaks_ties() -> None:
    a = Candidate(grid=((1,),), found_count=2, mean_log_probability=-0.1)
    b = Candidate(grid=((2,),), found_count=2, mean_log_probability=-2.0)
    assert select_attempts([b, a]) == (((1,),), ((2,),))
    assert math.exp(-0.1) < 1.0  # sanity: probability term cannot outweigh a count


def test_raw_qwen_format_matches_champion_template() -> None:
    from arcttt.model import turns_to_raw_qwen

    turns = (
        ChatTurn("user", "12\n34"),
        ChatTurn("assistant", "56"),
        ChatTurn("user", "78"),
    )
    text = turns_to_raw_qwen(turns, add_generation_prompt=True)
    assert text == (
        "<|im_start|>user\n12\n34<|im_end|>"
        "<|im_start|>assistant\n56<|im_end|>"
        "<|im_start|>user\n78<|im_end|>"
        "<|im_start|>assistant\n"
    )
    assert "system" not in text
    assert turns_to_raw_qwen(turns[:2], add_generation_prompt=False).endswith("<|im_end|>")


def test_full_permutation_palettes() -> None:
    from arcttt.augment import deterministic_palettes

    fixed = deterministic_palettes(5, 8, fix_background=True)
    assert all(p[0] == 0 for p in fixed)
    free = deterministic_palettes(5, 8, fix_background=False)
    assert any(p[0] != 0 for p in free)  # background moves in champion-style palettes
    assert all(sorted(p) == list(range(10)) for p in free)
