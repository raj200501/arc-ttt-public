"""End-to-end solver tests with a deterministic oracle predictor (no model)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from arcttt.augment import DIHEDRAL_SWEEP, Augmentation
from arcttt.solve import SolveConfig, solve_tasks, write_submission
from arcttt.tasks import Grid, Pair, Task


def identity_task(task_id: str, solvable: bool) -> Task:
    """Output equals input; the oracle knows this, the broken oracle does not."""

    grid: Grid = ((1, 2), (3, 4))
    return Task(
        task_id=task_id,
        train=(Pair(input=grid, output=grid),),
        test=(Pair(input=grid, output=grid if solvable else ((9,),)),),
    )


class OraclePredictor:
    """Predicts output == input (correct for identity tasks). Tracks adapt calls."""

    def __init__(self) -> None:
        self.adapted: list[str] = []

    def adapt(self, task: Task, augmentations: Sequence[Augmentation]) -> None:
        self.adapted.append(task.task_id)

    def predict(self, task: Task, test_index: int, samples: int) -> list[Grid]:
        return [task.test[test_index].input] * samples

    def log_probability(self, task: Task, test_index: int, output: Grid) -> float:
        return 0.0 if output == task.test[test_index].input else -10.0


def test_end_to_end_solve_and_submission(tmp_path: Path) -> None:
    tasks = [identity_task("solvable", True), identity_task("unsolvable", False)]
    predictor = OraclePredictor()
    config = SolveConfig(samples_per_augmentation=1)
    result = solve_tasks(tasks, predictor, config)

    assert predictor.adapted == ["solvable", "unsolvable"]
    assert result.scored == 2
    assert result.solved == 1  # identity oracle solves the identity task only
    assert result.accuracy == 0.5

    path = write_submission(result, tmp_path / "submission.json")
    payload = json.loads(path.read_text())
    assert set(payload) == {"solvable", "unsolvable"}
    entry = payload["solvable"][0]
    assert set(entry) == {"attempt_1", "attempt_2"}
    assert entry["attempt_1"] == [[1, 2], [3, 4]]


def test_oracle_predictions_survive_augmentation_round_trip() -> None:
    # The oracle answers in each augmented frame; pooling must invert correctly,
    # so the identity task is solved even though every frame differs.
    task = identity_task("t", True)
    result = solve_tasks([task], OraclePredictor(), SolveConfig(samples_per_augmentation=1))
    assert result.solved == 1
    assert len(DIHEDRAL_SWEEP) == 8


def test_ttt_augmentations_decoupled_from_prediction_frames() -> None:
    from arcttt.solve import solve_task

    class RecordingPredictor(OraclePredictor):
        def __init__(self) -> None:
            super().__init__()
            self.adapt_augmentations: Sequence[Augmentation] = ()

        def adapt(self, task: Task, augmentations: Sequence[Augmentation]) -> None:
            super().adapt(task, augmentations)
            self.adapt_augmentations = tuple(augmentations)

    from arcttt.augment import expanded_sweep

    ttt_set = expanded_sweep(seed=1, palettes_per_element=1)
    predictor = RecordingPredictor()
    config = SolveConfig(samples_per_augmentation=1, ttt_augmentations=ttt_set)
    solve_task(identity_task("t", True), predictor, config)
    assert predictor.adapt_augmentations == ttt_set  # trains on the expanded set
    assert config.augmentations == DIHEDRAL_SWEEP  # predicts in dihedral frames


def test_solve_uses_predict_frames_when_predictor_offers_it() -> None:
    from arcttt.solve import solve_task

    class FramePredictor(OraclePredictor):
        def __init__(self) -> None:
            super().__init__()
            self.frame_calls: list[int] = []

        def predict(self, task: Task, test_index: int, samples: int) -> list[Grid]:
            raise AssertionError("batched path must not fall back per frame")

        def predict_frames(
            self, tasks: Sequence[Task], test_index: int, samples: int
        ) -> list[list[Grid]]:
            self.frame_calls.append(len(tasks))
            return [[task.test[test_index].input] * samples for task in tasks]

    task = identity_task("t", True)
    predictor = FramePredictor()
    config = SolveConfig(samples_per_augmentation=1)
    attempts = solve_task(task, predictor, config)

    assert predictor.frame_calls == [len(config.augmentations)]  # one batched call
    # frame answers invert through pooling exactly like the per-frame path
    assert attempts[0][0] == task.test[0].input
