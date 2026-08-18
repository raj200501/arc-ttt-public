"""Kaggle kernel entry v10: same recall bound as v9, more time to use it.

v9 (cutoff 0.02 / 64 candidates) completed all 240 tasks in 520 of the 660
available minutes - no deadline-margin hit, no pacing-guard fire, no
straggler termination (kaggle_v9_run_2026-08-11.json). So the SESSION had
21% wall-clock headroom while the SEARCH was capped at 60 s per predict.
That leaves v9's result confounded: a flat score cannot distinguish "the
NLL bound was never binding" from "the widened frontier was breadth-first
and ran out of its 60 s before reaching the deeper truths".

v10 spends the measured headroom on exactly that ambiguity and changes ONE
knob from v9: dfs_time_budget_seconds 60 -> 90. Bound (0.02 / 3.91 nats),
candidate cap (64), frames, rank, ladder and pacing guards are v9 verbatim.

Budget math: DFS is one component of the 520 min; +50% on the per-predict
search cap projects well inside 660 min, and if the estimate is wrong the
existing pacing ladder degrades the tail rather than losing the file (that
machinery is proven - it simply never had to fire in v9).

Preregistered read, written before the score:
  - v9 flat AND v10 up   -> the frontier was time-starved; buy more search
    time before touching the bound again.
  - v9 flat AND v10 flat -> recall is NOT the binding constraint at this
    config; stop widening search, move to TTT sharpening (epochs 1 -> 2).
  - v9 up AND v10 up     -> keep pushing the same direction (bound + time).
This experiment is informative under every branch, which is why it is the
next submission regardless of which way v9 lands.
"""

from __future__ import annotations

import glob
import json as _json
import multiprocessing as _mp
import os as _os
import time as _time

import torch as _torch

from arcttt.augment import DIHEDRAL_SWEEP, Augmentation
from arcttt.model import CausalLMPredictor, TTTConfig
from arcttt.solve import SolveConfig, solve_task
from arcttt.tasks import Grid, Pair, Task, grid_to_lists, to_grid

WALL_BUDGET_SECONDS = 11.0 * 3600
PER_WORKER_MARGIN = 20 * 60  # stop starting tasks this close to the deadline

# Four maximally spread D4 elements: both parities, all rotation offsets.
AUG_QUARTET: tuple[Augmentation, ...] = (
    Augmentation(rotations=0, flip=False),
    Augmentation(rotations=1, flip=False),
    Augmentation(rotations=2, flip=True),
    Augmentation(rotations=3, flip=True),
)


def _make_configs() -> tuple[TTTConfig, SolveConfig]:
    # rank/alpha: champion pairing r=256/alpha=32 rslora; the T4 sweep
    # measured all ranks at near-identical wall-clock and memory, so
    # champion parity costs nothing. OOM ladder covers the tail — and at
    # 4 lockstep frames the peak that drove v7's 52 level-0 OOMs halves.
    ttt = TTTConfig(
        lora_rank=256,
        lora_alpha=32,
        epochs=1,
        max_new_tokens=992,
        max_sequence_tokens=2560,
        raw_qwen_format=True,
        gradient_checkpointing=True,
        use_dfs=True,
        dfs_probability_cutoff=0.02,
        dfs_max_candidates=64,
        shuffle_examples=True,
        dfs_time_budget_seconds=90.0,
        ttt_batch_size=1,
    )
    solve = SolveConfig(
        augmentations=AUG_QUARTET,
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


def _worker(
    rank: int,
    model_dir: str,
    tasks: list[Task],
    deadline: float,
    big_task_cells: int,
) -> None:
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
    first_alloc_error = True
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
            # Tail cap + pacing: v7 measured 793-1104 s on the largest
            # tasks against a 311 s/task budget. Tasks in the top size
            # quartile start at level 1 (their level-0 attempt was the
            # usual OOM-then-retry anyway); when the remaining wall-clock
            # per remaining task falls below the thresholds, every task
            # degrades a level so the tail gets attempted at all.
            start_level = 1 if _task_cells(task) >= big_task_cells else 0
            remaining = deadline - PER_WORKER_MARGIN - _time.time()
            per_task = remaining / max(1, len(tasks) - index)
            if per_task < 150:
                start_level = 2
            elif per_task < 240:
                start_level = max(start_level, 1)
            if index == 0 and start_level > 0 and _task_cells(task) < big_task_cells:
                # the guard binding on the FIRST task means the whole run is
                # time-starved (e.g. single-GPU fallback) — say so once, loudly
                print(
                    f"[w{rank}] pacing guard active from task 0: "
                    f"{per_task:.0f}s/task budget -> start level {start_level}",
                    flush=True,
                )
            ranked = None
            for level, (ttt_try, solve_try) in enumerate(ladder):
                if level < start_level:
                    continue
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
                except RuntimeError as error:
                    # cuBLAS/cuDNN allocation failures surface as plain
                    # RuntimeError, not OutOfMemoryError — same remedy.
                    # "CUBLAS" also matches non-alloc CUBLAS faults, so the
                    # first full message is logged per worker: a postmortem
                    # must be able to tell alloc pressure from a real bug.
                    message = str(error)
                    if (
                        "out of memory" not in message
                        and "CUBLAS" not in message
                        and "CUDNN" not in message
                    ):
                        raise
                    _torch.cuda.empty_cache()
                    if first_alloc_error:
                        first_alloc_error = False
                        print(f"[w{rank}] first alloc message: {message[:300]}", flush=True)
                    print(
                        f"[w{rank}] {task.task_id}: alloc RuntimeError at "
                        f"level {level}",
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
        if (index + 1) % 40 == 0:  # periodic telemetry so a killed run still reports
            print(
                f"[w{rank}] dfs telemetry @ {index + 1}: "
                f"{_json.dumps(CausalLMPredictor.dfs_telemetry())}",
                flush=True,
            )
        if (index + 1) % 5 == 0:
            print(
                f"[w{rank}] {index + 1}/{len(tasks)} "
                f"({_time.monotonic() - started:.0f}s last)",
                flush=True,
            )
    print(
        f"[w{rank}] FINAL dfs telemetry: "
        f"{_json.dumps(CausalLMPredictor.dfs_telemetry())}",
        flush=True,
    )


def main() -> None:
    started = _time.time()
    deadline = started + WALL_BUDGET_SECONDS
    model_dir = _find_model_dir()
    print("model dir:", model_dir, flush=True)
    # _rebuild_like reconstructs KV caches as full-attention layers; that is
    # only exact when the model uses no sliding-window layers. Qwen3 dense
    # defaults to none — surface the actual config so the assumption is a
    # printed fact, not a guess (see review finding on cache rebuild).
    # Exception-proof: this is a script kernel, so an uncaught exception in
    # main() means "Notebook Threw Exception" and no score even after the
    # fallback file exists — a diagnostic print must never be able to do that.
    try:
        model_config = _json.loads(open(_os.path.join(model_dir, "config.json")).read())
        sliding = model_config.get("use_sliding_window", False)
        layer_types = set(model_config.get("layer_types") or [])
        print(
            f"config: use_sliding_window={sliding} layer_types={sorted(layer_types)}",
            flush=True,
        )
        if sliding or (layer_types - {"full_attention"}):
            print(
                "WARNING: sliding-window attention configured — cache rebuild "
                "in the batched search assumes full-attention layers; results "
                "from compacted searches may be degraded on this model.",
                flush=True,
            )
    except Exception as error:
        print(f"config probe failed (non-fatal): {type(error).__name__}", flush=True)
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

    sizes = sorted(_task_cells(task) for task in tasks)
    big_task_cells = sizes[(3 * len(sizes)) // 4]  # top-quartile threshold

    workers = max(1, _torch.cuda.device_count())
    print(
        f"{len(tasks)} tasks | {workers} gpu workers | "
        f"big-task threshold {big_task_cells} cells",
        flush=True,
    )
    if workers == 1:
        _worker(0, model_dir, tasks, deadline, big_task_cells)
    else:
        context = _mp.get_context("spawn")
        procs = [
            context.Process(
                target=_worker,
                args=(rank, model_dir, tasks[rank::workers], deadline, big_task_cells),
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

    # Final search telemetry per worker is printed inside _worker; the parent
    # cannot see child class state (spawn), so the log is the collection point.
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
