"""Kaggle kernel entry: shard tasks across every visible GPU.

The scoring VM exposes 4x L4; one worker process per GPU roughly quadruples
the per-task time budget versus the single-GPU v4 kernel. Workers checkpoint
their partial submissions atomically after every task, so the parent can
always merge whatever exists when the wall-clock budget expires.
"""

from __future__ import annotations

import glob
import json as _json
import multiprocessing as _mp
import os as _os
import time as _time

import torch as _torch

from arcttt.augment import DIHEDRAL_SWEEP, expanded_sweep
from arcttt.model import CausalLMPredictor, TTTConfig
from arcttt.solve import SolveConfig, solve_task
from arcttt.tasks import Grid, Pair, Task, grid_to_lists, to_grid

WALL_BUDGET_SECONDS = 11.0 * 3600
PER_WORKER_MARGIN = 20 * 60  # stop starting tasks this close to the deadline


def _make_configs() -> tuple[TTTConfig, SolveConfig]:
    ttt = TTTConfig(
        lora_rank=16,
        lora_alpha=32,
        epochs=1,
        max_new_tokens=992,
        max_sequence_tokens=2560,
        raw_qwen_format=True,
        gradient_checkpointing=True,
        use_dfs=True,
        dfs_probability_cutoff=0.1,
        dfs_max_candidates=32,
        shuffle_examples=True,
    )
    solve = SolveConfig(
        augmentations=DIHEDRAL_SWEEP,
        samples_per_augmentation=1,
        rescore_augmentations=DIHEDRAL_SWEEP,
        ttt_augmentations=expanded_sweep(seed=0, palettes_per_element=1),
    )
    return ttt, solve


def _find_model_dir() -> str:
    for config_path in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        directory = _os.path.dirname(config_path)
        if glob.glob(_os.path.join(directory, "*.safetensors")):
            return directory
    raise RuntimeError("no model directory with config.json + safetensors found")


def _load_challenges(path: str) -> list[Task]:
    raw = _json.loads(open(path).read())
    tasks = []
    for task_id, payload in sorted(raw.items()):
        train = tuple(
            Pair(input=to_grid(p["input"]), output=to_grid(p["output"]))
            for p in payload["train"]
        )
        test = tuple(Pair(input=to_grid(p["input"]), output=None) for p in payload["test"])
        tasks.append(Task(task_id=task_id, train=train, test=test))
    return tasks


def _task_cells(task: Task) -> int:
    total = 0
    for pair in list(task.train) + list(task.test):
        total += len(pair.input) * len(pair.input[0])
        if pair.output is not None:
            total += len(pair.output) * len(pair.output[0])
    return total


def _write_atomic(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    open(tmp, "w").write(_json.dumps(payload))
    _os.replace(tmp, path)


def _worker(rank: int, model_dir: str, tasks: list[Task], deadline: float) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _torch.device(f"cuda:{rank}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=_torch.bfloat16).to(
        device
    )
    ttt, solve = _make_configs()
    part_path = f"submission_part_{rank}.json"
    part: dict[str, list[dict[str, list[list[int]]]]] = {}
    for index, task in enumerate(tasks):
        if _time.time() > deadline - PER_WORKER_MARGIN:
            print(f"[w{rank}] deadline margin reached at task {index}", flush=True)
            break
        started = _time.monotonic()
        try:
            predictor = CausalLMPredictor(model, tokenizer, ttt, device)
            ranked = solve_task(task, predictor, solve)
            entries = []
            for attempts in ranked:
                if attempts:
                    first = attempts[0]
                    second = attempts[1] if len(attempts) > 1 else first
                    entries.append(
                        {
                            "attempt_1": grid_to_lists(first),
                            "attempt_2": grid_to_lists(second),
                        }
                    )
                else:
                    entries.append(None)  # keep the parent's fallback for this test
            part[task.task_id] = entries
        except _torch.OutOfMemoryError:
            _torch.cuda.empty_cache()
            print(f"[w{rank}] {task.task_id}: OOM", flush=True)
        except Exception as error:
            print(f"[w{rank}] {task.task_id}: {type(error).__name__}", flush=True)
        _write_atomic(part_path, part)
        if (index + 1) % 5 == 0:
            print(
                f"[w{rank}] {index + 1}/{len(tasks)} "
                f"({_time.monotonic() - started:.0f}s last)",
                flush=True,
            )


def main() -> None:
    started = _time.time()
    deadline = started + WALL_BUDGET_SECONDS
    model_dir = _find_model_dir()
    print("model dir:", model_dir, flush=True)
    matches = glob.glob("/kaggle/input/**/*test_challenges.json", recursive=True)
    if not matches:
        matches = glob.glob("/kaggle/input/**/*evaluation_challenges.json", recursive=True)
    if not matches:
        listing = "\n".join(sorted(glob.glob("/kaggle/input/**", recursive=True))[:80])
        raise RuntimeError("no challenges file under /kaggle/input; tree:\n" + listing)
    print("challenges:", matches[0], flush=True)
    tasks = _load_challenges(matches[0])
    tasks.sort(key=_task_cells)  # smallest first: budget -> most real predictions

    fallback: Grid = ((0,),)
    submission: dict[str, list[dict[str, list[list[int]]]]] = {
        task.task_id: [
            {"attempt_1": grid_to_lists(fallback), "attempt_2": grid_to_lists(fallback)}
            for _ in task.test
        ]
        for task in tasks
    }
    _write_atomic("submission.json", submission)

    workers = max(1, _torch.cuda.device_count())
    print(f"{len(tasks)} tasks | {workers} gpu workers", flush=True)
    if workers == 1:
        _worker(0, model_dir, tasks, deadline)
    else:
        context = _mp.get_context("spawn")
        procs = [
            context.Process(
                target=_worker, args=(rank, model_dir, tasks[rank::workers], deadline)
            )
            for rank in range(workers)
        ]
        for proc in procs:
            proc.start()
        # A worker hung mid-task must not block past the platform's 12h kill,
        # or the merged submission never gets written at all.
        for proc in procs:
            proc.join(timeout=max(60.0, deadline + 10 * 60 - _time.time()))
        for proc in procs:
            if proc.is_alive():
                print(f"terminating straggler worker {proc.pid}", flush=True)
                proc.terminate()
                proc.join(timeout=60)

    for part_path in glob.glob("submission_part_*.json"):
        try:
            part = _json.loads(open(part_path).read())
        except Exception as error:
            print(f"unreadable {part_path}: {type(error).__name__}", flush=True)
            continue
        for task_id, entries in part.items():
            for test_index, entry in enumerate(entries):
                if entry is not None and test_index < len(submission[task_id]):
                    submission[task_id][test_index] = entry
    _write_atomic("submission.json", submission)
    real = sum(
        1
        for entries in submission.values()
        for entry in entries
        if entry["attempt_1"] != grid_to_lists(fallback)
    )
    print(
        f"submission.json complete | {real} real predictions | "
        f"{(_time.time() - started) / 60:.0f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
