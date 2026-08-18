"""Kaggle notebook entrypoint for the ARC-AGI-2 track.

Paste-able into a Kaggle code notebook (or attach this repo as a dataset and
run it). Offline-safe: expects the base/fine-tuned model attached as a Kaggle
model/dataset, reads the competition's test challenges JSON, writes
``submission.json`` in the required two-attempt schema.

Competition constraints this respects (verified 2026-08-08):
- no internet during scoring; model weights must come from attached datasets;
- <= 12 hours wall clock (T4 smoke measured ~77 s/task at 4 augmentations,
  1 sample, 0.5B model => ~5.1 h for 240 tasks; L4x4 gives real headroom);
- output file `submission.json`: {task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")  # repo attached at the notebook working directory

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from arcttt.augment import DIHEDRAL_SWEEP
from arcttt.model import CausalLMPredictor, TTTConfig
from arcttt.solve import SolveConfig, solve_task
from arcttt.tasks import Grid, Pair, Task, grid_to_lists, to_grid

# --- configuration ---------------------------------------------------------

MODEL_PATH = "/kaggle/input/arcttt-model"  # attached model dataset directory
CHALLENGES = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
OUTPUT = "submission.json"
TIME_BUDGET_SECONDS = int(11.0 * 3600)  # leave margin under the 12 h cap

TTT = TTTConfig(lora_rank=16, lora_alpha=32, epochs=1, max_new_tokens=992,
                max_sequence_tokens=2560)
SOLVE = SolveConfig(
    augmentations=DIHEDRAL_SWEEP[:4],
    samples_per_augmentation=1,
    rescore_augmentations=DIHEDRAL_SWEEP[:4],
)
FALLBACK: Grid = ((0,),)


def load_challenges(path: str) -> list[Task]:
    raw = json.loads(Path(path).read_text())
    tasks = []
    for task_id, payload in sorted(raw.items()):
        train = tuple(
            Pair(input=to_grid(p["input"]), output=to_grid(p["output"]))
            for p in payload["train"]
        )
        test = tuple(Pair(input=to_grid(p["input"]), output=None) for p in payload["test"])
        tasks.append(Task(task_id=task_id, train=train, test=test))
    return tasks


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to(device)
    tasks = load_challenges(CHALLENGES)
    print(f"{len(tasks)} tasks on {device}", flush=True)

    submission: dict[str, list[dict[str, list[list[int]]]]] = {}
    started = time.monotonic()
    for index, task in enumerate(tasks):
        # Always emit fallback entries first so the submission stays valid
        # even if the budget runs out mid-way.
        submission[task.task_id] = [
            {"attempt_1": grid_to_lists(FALLBACK), "attempt_2": grid_to_lists(FALLBACK)}
            for _ in task.test
        ]
        if time.monotonic() - started > TIME_BUDGET_SECONDS:
            continue  # budget exhausted: fallback rows only
        try:
            predictor = CausalLMPredictor(model, tokenizer, TTT, device)
            ranked = solve_task(task, predictor, SOLVE)
            for test_index, attempts in enumerate(ranked):
                if attempts:
                    first = attempts[0]
                    second = attempts[1] if len(attempts) > 1 else first
                    submission[task.task_id][test_index] = {
                        "attempt_1": grid_to_lists(first),
                        "attempt_2": grid_to_lists(second),
                    }
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
        if (index + 1) % 10 == 0:
            elapsed = time.monotonic() - started
            print(f"{index + 1}/{len(tasks)} tasks, {elapsed / 60:.0f} min", flush=True)
        Path(OUTPUT).write_text(json.dumps(submission))  # checkpoint every task

    Path(OUTPUT).write_text(json.dumps(submission))
    print("submission.json written", flush=True)


if __name__ == "__main__":
    main()
