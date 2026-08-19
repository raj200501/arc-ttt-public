"""ARC task data model: loading, validation, and exact-match scoring.

Grids are tuples of tuples of ints (colors 0-9), immutable and hashable so
augmentation dedup and vote counting stay trivial and bug-resistant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

Grid = tuple[tuple[int, ...], ...]

MAX_SIDE = 30
COLORS = 10


class TaskFormatError(ValueError):
    """Raised when a task file violates the ARC schema."""


def to_grid(rows: object) -> Grid:
    if not isinstance(rows, list) or not rows:
        raise TaskFormatError("grid must be a non-empty list of rows")
    width = None
    out: list[tuple[int, ...]] = []
    for row in rows:
        if not isinstance(row, list) or not row:
            raise TaskFormatError("grid rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise TaskFormatError("grid rows must share one width")
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell < COLORS:
                raise TaskFormatError("grid cells must be ints in [0, 10)")
        out.append(tuple(row))
    if len(out) > MAX_SIDE or (width or 0) > MAX_SIDE:
        raise TaskFormatError(f"grid exceeds {MAX_SIDE}x{MAX_SIDE}")
    return tuple(out)


def grid_to_lists(grid: Grid) -> list[list[int]]:
    return [list(row) for row in grid]


@dataclass(frozen=True)
class Pair:
    input: Grid
    output: Grid | None  # None for hidden test outputs


@dataclass(frozen=True)
class Task:
    task_id: str
    train: tuple[Pair, ...]
    test: tuple[Pair, ...]

    def validate(self) -> None:
        if not self.train or not self.test:
            raise TaskFormatError(f"{self.task_id}: needs train and test pairs")
        for pair in self.train:
            if pair.output is None:
                raise TaskFormatError(f"{self.task_id}: train pairs need outputs")


def load_task(path: str | Path) -> Task:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "train" not in raw or "test" not in raw:
        raise TaskFormatError(f"{path.name}: missing train/test keys")

    def pairs(items: object, split: str) -> tuple[Pair, ...]:
        if not isinstance(items, list):
            raise TaskFormatError(f"{path.name}: {split} must be a list")
        built = []
        for item in items:
            if not isinstance(item, dict) or "input" not in item:
                raise TaskFormatError(f"{path.name}: {split} pair needs input")
            output = item.get("output")
            built.append(
                Pair(
                    input=to_grid(item["input"]),
                    output=to_grid(output) if output is not None else None,
                )
            )
        return tuple(built)

    task = Task(task_id=path.stem, train=pairs(raw["train"], "train"), test=pairs(raw["test"], "test"))
    task.validate()
    return task


def load_directory(directory: str | Path) -> dict[str, Task]:
    directory = Path(directory)
    tasks = {}
    for path in sorted(directory.glob("*.json")):
        task = load_task(path)
        tasks[task.task_id] = task
    if not tasks:
        raise TaskFormatError(f"no task files under {directory}")
    return tasks


def score_attempts(attempts: list[Grid], solution: Grid, max_attempts: int = 2) -> bool:
    """ARC scoring: correct if ANY of the first `max_attempts` attempts matches exactly."""

    return any(attempt == solution for attempt in attempts[:max_attempts])
