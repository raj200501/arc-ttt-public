"""End-to-end solver: tasks -> per-test attempts -> submission.json."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from arcttt.augment import DIHEDRAL_SWEEP, Augmentation
from arcttt.model import Predictor
from arcttt.tasks import Grid, Task, grid_to_lists, score_attempts
from arcttt.vote import pool_predictions, rescore_candidates, select_attempts


@dataclass(frozen=True)
class SolveConfig:
    augmentations: tuple[Augmentation, ...] = DIHEDRAL_SWEEP
    samples_per_augmentation: int = 2
    rescore_augmentations: tuple[Augmentation, ...] = DIHEDRAL_SWEEP
    attempts: int = 2
    # TTT can train on a larger augmentation set (e.g. dihedral x color
    # permutations) than the frames predictions are generated in.
    ttt_augmentations: tuple[Augmentation, ...] | None = None


@dataclass
class SolveResult:
    attempts: dict[str, list[dict[str, list[list[int]]]]] = field(default_factory=dict)
    solved: int = 0
    scored: int = 0

    @property
    def accuracy(self) -> float:
        return self.solved / self.scored if self.scored else 0.0


def solve_task(task: Task, predictor: Predictor, config: SolveConfig) -> list[list[Grid]]:
    """Return ranked attempts for every test input of one task."""

    predictor.adapt(task, config.ttt_augmentations or config.augmentations)
    per_test: list[list[Grid]] = []
    for test_index in range(len(task.test)):
        predictions: list[tuple[Augmentation, Grid]] = []
        frame_predictor = getattr(predictor, "predict_frames", None)
        if callable(frame_predictor):
            # Hand all augmentation frames to the predictor at once; with DFS
            # decoding this runs as one lockstep-batched pass, otherwise it
            # falls back to per-frame predict().
            transformed_tasks = [
                augmentation.apply_task(task) for augmentation in config.augmentations
            ]
            frame_grids = frame_predictor(
                transformed_tasks, test_index, config.samples_per_augmentation
            )
            for augmentation, grids in zip(config.augmentations, frame_grids):
                for grid in grids:
                    predictions.append((augmentation, grid))
        else:
            for augmentation in config.augmentations:
                transformed = augmentation.apply_task(task)
                for grid in predictor.predict(
                    transformed, test_index, config.samples_per_augmentation
                ):
                    predictions.append((augmentation, grid))
        counts = pool_predictions(predictions)
        if not counts:
            per_test.append([])
            continue

        # Shared guard for all three scorer branches: the fast paths divide
        # by len(rescore_augmentations) and would otherwise raise an
        # uninformative ZeroDivisionError (the fallback branch raises this
        # same ValueError via rescore_candidates).
        if not config.rescore_augmentations:
            raise ValueError("rescoring needs at least one augmentation")

        pair_scorer = getattr(predictor, "log_probabilities_pairs", None)
        batch_scorer = getattr(predictor, "log_probabilities", None)
        if callable(pair_scorer):
            # All (augmentation, candidate) pairs scored in chunked batch
            # forwards — no per-frame call overhead.
            grids = list(counts)
            pairs = []
            for augmentation in config.rescore_augmentations:
                transformed_task = augmentation.apply_task(task)
                for grid in grids:
                    pairs.append(
                        (transformed_task, test_index, augmentation.apply(grid))
                    )
            flat_scores = pair_scorer(pairs)
            totals = {grid: 0.0 for grid in grids}
            for pair_index, score in enumerate(flat_scores):
                totals[grids[pair_index % len(grids)]] += score
            from arcttt.vote import Candidate

            candidates = tuple(
                Candidate(
                    grid=grid,
                    found_count=counts[grid],
                    mean_log_probability=totals[grid] / len(config.rescore_augmentations),
                )
                for grid in grids
            )
        elif callable(batch_scorer):
            # One padded batch forward per augmentation scores every candidate.
            grids = list(counts)
            totals = {grid: 0.0 for grid in grids}
            for augmentation in config.rescore_augmentations:
                transformed_task = augmentation.apply_task(task)
                rendered = [augmentation.apply(grid) for grid in grids]
                for grid, score in zip(
                    grids, batch_scorer(transformed_task, test_index, rendered)
                ):
                    totals[grid] += score
            from arcttt.vote import Candidate

            candidates = tuple(
                Candidate(
                    grid=grid,
                    found_count=counts[grid],
                    mean_log_probability=totals[grid] / len(config.rescore_augmentations),
                )
                for grid in grids
            )
        else:

            def log_probability(grid: Grid, augmentation: Augmentation) -> float:
                transformed_task = augmentation.apply_task(task)
                return predictor.log_probability(
                    transformed_task, test_index, augmentation.apply(grid)
                )

            candidates = rescore_candidates(
                counts, config.rescore_augmentations, log_probability
            )
        per_test.append(list(select_attempts(candidates, config.attempts)))
    return per_test


def solve_tasks(
    tasks: Sequence[Task], predictor: Predictor, config: SolveConfig
) -> SolveResult:
    result = SolveResult()
    fallback: Grid = ((0,),)
    for task in tasks:
        ranked = solve_task(task, predictor, config)
        entries = []
        for test_index, attempts in enumerate(ranked):
            first = attempts[0] if attempts else fallback
            second = attempts[1] if len(attempts) > 1 else first
            entries.append(
                {"attempt_1": grid_to_lists(first), "attempt_2": grid_to_lists(second)}
            )
            solution = task.test[test_index].output
            if solution is not None:
                result.scored += 1
                if score_attempts([first, second], solution):
                    result.solved += 1
        result.attempts[task.task_id] = entries
    return result


def write_submission(result: SolveResult, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(result.attempts), encoding="utf-8")
    return path
