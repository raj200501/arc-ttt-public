"""Grid and task serialization for language-model training and inference.

Clean-room implementation of the digit-serialization scheme described in the
NVARC 2025 paper (grids as digit rows inside chat turns): each grid row is a
line of digits, demonstration pairs become user/assistant turns, and the test
input is the final user turn awaiting an assistant completion. No code from
the NVARC repository (which carries no license) is used.
"""

from __future__ import annotations

from dataclasses import dataclass

from arcttt.tasks import Grid, Task, TaskFormatError, to_grid


def grid_to_text(grid: Grid) -> str:
    return "\n".join("".join(str(cell) for cell in row) for row in grid)


def text_to_grid(text: str) -> Grid:
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if not all(ch.isdigit() for ch in line):
            raise TaskFormatError(f"non-digit character in grid text: {line[:20]!r}")
        rows.append([int(ch) for ch in line])
    return to_grid(rows)


@dataclass(frozen=True)
class ChatTurn:
    role: str  # "user" | "assistant"
    content: str


def task_to_messages(task: Task, test_index: int = 0) -> tuple[ChatTurn, ...]:
    """Serialize demonstrations plus one test input as chat turns.

    The model is expected to complete the final assistant turn with the
    serialized output grid.
    """

    if not 0 <= test_index < len(task.test):
        raise TaskFormatError(f"{task.task_id}: test index {test_index} out of range")
    turns: list[ChatTurn] = []
    for pair in task.train:
        if pair.output is None:
            raise TaskFormatError(f"{task.task_id}: train pair missing output")
        turns.append(ChatTurn("user", grid_to_text(pair.input)))
        turns.append(ChatTurn("assistant", grid_to_text(pair.output)))
    turns.append(ChatTurn("user", grid_to_text(task.test[test_index].input)))
    return tuple(turns)


def ttt_training_examples(
    task: Task, shuffle_seed: int | None = None
) -> tuple[tuple[ChatTurn, ...], ...]:
    """Leave-one-out demonstration examples for per-task adaptation.

    For each demonstration pair, the remaining pairs act as context and the
    held-out pair supplies the supervised completion — the standard per-task
    TTT corpus construction, multiplied later by augmentations. A shuffle seed
    permutes the context-pair order deterministically, so different
    augmentations of the same task present the demonstrations differently.
    """

    import random

    order_rng = random.Random(shuffle_seed) if shuffle_seed is not None else None
    examples = []
    for held_out in range(len(task.train)):
        context = [
            index
            for index, pair in enumerate(task.train)
            if index != held_out and pair.output is not None
        ]
        if order_rng is not None:
            order_rng.shuffle(context)
        turns: list[ChatTurn] = []
        for index in context:
            pair = task.train[index]
            if pair.output is None:  # excluded above; narrows the type
                continue
            turns.append(ChatTurn("user", grid_to_text(pair.input)))
            turns.append(ChatTurn("assistant", grid_to_text(pair.output)))
        target = task.train[held_out]
        if target.output is None:
            continue
        turns.append(ChatTurn("user", grid_to_text(target.input)))
        turns.append(ChatTurn("assistant", grid_to_text(target.output)))
        examples.append(tuple(turns))
    if not examples:
        raise TaskFormatError(f"{task.task_id}: no usable TTT examples")
    return tuple(examples)
