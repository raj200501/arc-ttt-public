"""Kaggle kernel entry: shard tasks across every visible GPU.

The run environment is pinned to T4 (machine_shape) — the kernel's own run
produces the scored file, so it must be the environment we validated on.
One worker per visible GPU; workers checkpoint their partial submissions
atomically after every task, so the parent can always merge whatever
exists when the wall-clock budget expires.
"""

from __future__ import annotations

import glob
import json as _json
import multiprocessing as _mp
import os as _os
import time as _time

import torch as _torch

from arcttt.augment import DIHEDRAL_SWEEP
from arcttt.model import CausalLMPredictor, TTTConfig
from arcttt.solve import SolveConfig, solve_task
from arcttt.tasks import Grid, Pair, Task, grid_to_lists, to_grid

WALL_BUDGET_SECONDS = 11.0 * 3600
PER_WORKER_MARGIN = 20 * 60  # stop starting tasks this close to the deadline


def _make_configs() -> tuple[TTTConfig, SolveConfig]:
    # rank/alpha: champion pairing is r=256/alpha=32 rslora; rank here is
    # set from the measured T4 rank sweep (experiments/t4_rank_sweep_*.json).
    # Config set from t4_champ_diag_2026-08-08: TTT sharpens lp(true) at
    # 8 dihedral augs; 16-aug TTT costs 600-950 s/task on T4 (does not fit
    # 240 tasks in 11h) and OOMs 3/8 tasks even at batch 1. Recall, not
    # search depth, is the gap -> cutoff 0.1 + greedy-include. Rank comes
    # from the measured sweep (experiments/t4_rank_sweep_*.json).
    # rank 256 = champion pairing; the T4 sweep measured all ranks at
    # near-identical wall-clock (485-630 s) and peak memory (10.9-12.8 GB
    # of 16), so champion parity costs nothing extra. OOM ladder covers
    # the tail.
    ttt = TTTConfig(
        lora_rank=256,
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
        dfs_time_budget_seconds=120.0,
        ttt_batch_size=1,
    )
    solve = SolveConfig(
        augmentations=DIHEDRAL_SWEEP,
        samples_per_augmentation=1,
        rescore_augmentations=DIHEDRAL_SWEEP,
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


def _bf16_compute_works(device: "_torch.device") -> bool:
    """Functional probe: is_bf16_supported() lies in both directions across
    torch versions (False on T4 where emulated bf16 works; potentially True
    where kernels are missing). Running an actual matmul is ground truth."""

    try:
        a = _torch.randn(8, 8, dtype=_torch.bfloat16, device=device)
        result = float((a @ a).float().sum().item())
        return result == result  # reached => kernels exist (NaN-safe tautology)
    except Exception:
        return False


def _worker(rank: int, model_dir: str, tasks: list[Task], deadline: float) -> None:
    import dataclasses
    import traceback

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _torch.device(f"cuda:{rank}")
    # The kernel's own run produces the scored file (no separate rerun), so
    # the pipeline must actually execute HERE. machine_shape pins a T4; this
    # probe is the belt-and-suspenders: degrade to fp16 inference-only
    # (TTT off — fp16 training without a loss scaler is unsafe) rather than
    # dying, and say so loudly.
    bf16 = _bf16_compute_works(device)
    dtype = _torch.bfloat16 if bf16 else _torch.float16
    if not bf16:
        print(
            f"[w{rank}] WARNING: bf16 compute unavailable on this GPU — "
            "fp16 inference-only mode (TTT disabled). If this appears on a "
            "submitted run, the accelerator pin failed.",
            flush=True,
        )
    print(
        f"[w{rank}] {_torch.cuda.get_device_name(rank)} | bf16={bf16} "
        f"| dtype={dtype}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device)
    ttt, solve = _make_configs()
    if not bf16:  # fp16 training without a loss scaler is unsafe; skip TTT
        ttt = dataclasses.replace(ttt, epochs=0)
    first_error = True
    part_path = f"submission_part_{rank}.json"
    part: dict[str, list[dict[str, list[list[int]]]]] = {}
    for index, task in enumerate(tasks):
        if _time.time() > deadline - PER_WORKER_MARGIN:
            print(f"[w{rank}] deadline margin reached at task {index}", flush=True)
            break
        started = _time.monotonic()
        try:
            # OOM degrade-and-retry ladder: a memory failure costs a retry
            # at a lighter config, not the whole task. Level 1 halves the
            # sequence cap (drops the biggest training examples); level 2
            # additionally skips TTT entirely (inference-only).
            ladder = (
                (ttt, solve),
                (dataclasses.replace(ttt, max_sequence_tokens=1280), solve),
                (
                    dataclasses.replace(
                        ttt, max_sequence_tokens=1280, epochs=0
                    ),
                    solve,
                ),
            )
            ranked = None
            for level, (ttt_try, solve_try) in enumerate(ladder):
                try:
                    predictor = CausalLMPredictor(model, tokenizer, ttt_try, device)
                    ranked = solve_task(task, predictor, solve_try)
                    break
                except _torch.OutOfMemoryError:
                    _torch.cuda.empty_cache()
                    print(
                        f"[w{rank}] {task.task_id}: OOM at level {level}",
                        flush=True,
                    )
            if ranked is None:
                continue  # every ladder level OOMed; fallback rows stand
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
        except Exception as error:
            print(f"[w{rank}] {task.task_id}: {type(error).__name__}", flush=True)
            if first_error:  # one full traceback per worker, not 60
                first_error = False
                traceback.print_exc()
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
