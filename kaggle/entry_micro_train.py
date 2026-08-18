"""Kaggle kernel entry: G7 micro-tier own-model proof — bounded LoRA
continued-training of Qwen2.5-0.5B-Instruct on public ARC TRAINING data.

Not the champion 4B on purpose: the point of the micro tier (GOALS.md G7,
OWN_MODEL_PLAN.md) is an adapter WE own, trained on license-clean data
(ARC-AGI training set, Apache-2.0), on the free weekly Kaggle T4 quota —
promotional credits stay untouched. The run is evidence, not a leaderboard
play: a 0.5B model is expected to solve ~0 eval tasks; the registry number
is the paired before/after held-out eval (solved count AND teacher-forced
mean lp(true), which moves even when solve counts do not — t4-champ-diag
lesson).

Data rules (non-negotiable): training examples come ONLY from
*training_challenges.json + *training_solutions.json. The held-out eval
uses 10 public *evaluation* tasks — never as training input, and never any
test data.

Hardening inherited from v7/v8: machine_shape pins the T4 (v6 postmortem),
functional bf16 probe (is_bf16_supported lies on T4), atomic writes for
every artifact, wall-clock budget with margins so the platform kill never
eats the checkpoint, OOM-skip in the train loop, pure-torch LoRA only
(kaggle-v3 died on a peft ImportError; the image does not ship it).
"""

from __future__ import annotations

import glob
import json as _json
import math as _math
import multiprocessing as _mp
import os as _os
import random as _random
import time as _time
from collections.abc import Iterator, Sequence

import torch as _torch

from arcttt.augment import DIHEDRAL_SWEEP, Augmentation
from arcttt.lora import inject_lora, lora_parameters, remove_lora
from arcttt.model import CausalLMPredictor, TTTConfig
from arcttt.serialize import ChatTurn, grid_to_text
from arcttt.solve import SolveConfig, solve_task
from arcttt.tasks import Pair, Task, score_attempts, to_grid

# -- budgets ----------------------------------------------------------------
WALL_BUDGET_SECONDS = 4.0 * 3600  # total kernel budget (quota math in RUNBOOK)
AFTER_EVAL_RESERVE = 40 * 60  # reserved after training for the paired eval
SAVE_MARGIN = 5 * 60  # stop training this far before the reserve starts

# -- training hyperparameters (micro-tier: bounded, documented, seeded) -----
LORA_RANK = 64  # t4-rank-sweep: r=64 fastest+lightest; adapter is ours either way
LORA_ALPHA = 32  # rslora scaling, champion pairing
LEARNING_RATE = 1e-4
WARMUP_STEPS = 50  # optimizer steps of linear warmup, then constant
GRAD_ACCUM = 8  # micro-batch 1 x 8 accumulation per optimizer step
CHECKPOINT_EVERY = 25  # optimizer steps between atomic checkpoint writes
MAX_SEQ_TOKENS = 2048  # skip longer serialized tasks (logged)
SEED = 0

EVAL_TASK_COUNT = 10  # public eval tasks, sorted by task_id, deterministic
EVAL_MAX_SEQ_TOKENS = 4096
EVAL_DFS_BUDGET_SECONDS = 20.0

# Four maximally spread D4 elements (v8's quartet): both parities, all offsets.
AUG_QUARTET: tuple[Augmentation, ...] = (
    Augmentation(rotations=0, flip=False),
    Augmentation(rotations=1, flip=False),
    Augmentation(rotations=2, flip=True),
    Augmentation(rotations=3, flip=True),
)


# -- shared v8 utilities ----------------------------------------------------


def _find_model_dir() -> str:
    for config_path in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        directory = _os.path.dirname(config_path)
        if glob.glob(_os.path.join(directory, "*.safetensors")):
            return directory
    listing = "\n".join(sorted(glob.glob("/kaggle/input/**", recursive=True))[:80])
    raise RuntimeError(
        "no model directory with config.json + safetensors found; tree:\n" + listing
    )


def _write_atomic(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    open(tmp, "w").write(_json.dumps(payload))
    _os.replace(tmp, path)


def _bf16_compute_works(device: "_torch.device") -> bool:
    """Functional probe: is_bf16_supported() lies in both directions across
    torch versions (False on T4 where emulated bf16 works). Ground truth is
    an actual matmul."""

    try:
        a = _torch.randn(8, 8, dtype=_torch.bfloat16, device=device)
        result = float((a @ a).float().sum().item())
        return result == result
    except Exception:
        return False


def _find_one(patterns: Sequence[str]) -> str:
    for pattern in patterns:
        matches = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
        if matches:
            return matches[0]
    listing = "\n".join(sorted(glob.glob("/kaggle/input/**", recursive=True))[:80])
    raise RuntimeError(f"none of {patterns} under /kaggle/input; tree:\n{listing}")


def _load_split(challenges_path: str, solutions_path: str) -> list[Task]:
    """Tasks with test outputs attached from the paired solutions file."""

    challenges = _json.loads(open(challenges_path).read())
    solutions = _json.loads(open(solutions_path).read())
    tasks: list[Task] = []
    for task_id, payload in sorted(challenges.items()):
        answers = solutions.get(task_id)
        if answers is None:
            continue
        train = tuple(
            Pair(input=to_grid(p["input"]), output=to_grid(p["output"]))
            for p in payload["train"]
        )
        test = tuple(
            Pair(input=to_grid(p["input"]), output=to_grid(answers[i]))
            for i, p in enumerate(payload["test"])
        )
        tasks.append(Task(task_id=task_id, train=train, test=test))
    if not tasks:
        raise RuntimeError(f"no tasks with solutions from {challenges_path}")
    return tasks


# -- training data ----------------------------------------------------------


def task_examples(task: Task) -> list[tuple[ChatTurn, ...]]:
    """One supervised example per test pair: demonstrations as chat turns,
    the test output as the supervised final assistant turn — the exact raw
    serialization the harness trains and decodes (kaggle-v1/v2 lesson: the
    format IS the model interface)."""

    examples = []
    for test_pair in task.test:
        if test_pair.output is None:
            continue
        turns: list[ChatTurn] = []
        for pair in task.train:
            if pair.output is None:
                continue
            turns.append(ChatTurn("user", grid_to_text(pair.input)))
            turns.append(ChatTurn("assistant", grid_to_text(pair.output)))
        turns.append(ChatTurn("user", grid_to_text(test_pair.input)))
        turns.append(ChatTurn("assistant", grid_to_text(test_pair.output)))
        examples.append(tuple(turns))
    return examples


def example_stream(tasks: Sequence[Task], seed: int) -> Iterator[tuple[ChatTurn, ...]]:
    """Infinite deterministic stream: each epoch reshuffles task order and
    advances every task one step around the dihedral sweep, so a budget-cut
    epoch still saw a uniform augmentation mix."""

    epoch = 0
    while True:
        rng = _random.Random(f"{seed}:{epoch}")
        order = list(range(len(tasks)))
        rng.shuffle(order)
        for position, task_index in enumerate(order):
            augmentation = DIHEDRAL_SWEEP[(task_index + epoch) % len(DIHEDRAL_SWEEP)]
            transformed = augmentation.apply_task(tasks[task_index])
            for example in task_examples(transformed):
                yield example
        epoch += 1


# -- adapter checkpointing --------------------------------------------------


def save_adapter(model: _torch.nn.Module, path: str, metadata: dict[str, str]) -> int:
    """Atomically write every lora_a/lora_b tensor; returns tensor count."""

    from safetensors.torch import save_file

    tensors = {
        name: parameter.detach().to(_torch.float32).cpu().contiguous()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }
    if not tensors:
        raise RuntimeError("no LoRA tensors to save")
    tmp = f"{path}.tmp"
    save_file(tensors, tmp, metadata={k: str(v) for k, v in metadata.items()})
    _os.replace(tmp, path)
    return len(tensors)


# -- training loop ----------------------------------------------------------


def train_micro(
    model: object,
    tokenizer: object,
    tasks: Sequence[Task],
    device: _torch.device,
    *,
    deadline: float,
    adapter_path: str,
    log_path: str,
    max_steps: int | None = None,
    grad_accum: int = GRAD_ACCUM,
    checkpoint_every: int = CHECKPOINT_EVERY,
    learning_rate: float = LEARNING_RATE,
    warmup_steps: int = WARMUP_STEPS,
    lora_rank: int = LORA_RANK,
    lora_alpha: int = LORA_ALPHA,
    max_sequence_tokens: int = MAX_SEQ_TOKENS,
    seed: int = SEED,
) -> dict:
    """Budget-driven LoRA continued-training; returns the loss-curve log.

    The loop streams serialized tasks until the deadline (or max_steps for
    the CPU smoke), checkpointing the adapter + log atomically so a platform
    kill costs at most `checkpoint_every` steps of progress.
    """

    _torch.manual_seed(seed)
    encoder = CausalLMPredictor(
        model,
        tokenizer,
        TTTConfig(raw_qwen_format=True, max_sequence_tokens=max_sequence_tokens),
        device,
    )
    remove_lora(model)  # defensive: never stack adapters
    wrapped = inject_lora(model, lora_rank, lora_alpha, use_rslora=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    parameters = lora_parameters(model)
    optimizer = _torch.optim.AdamW(parameters, lr=learning_rate)

    metadata = {
        "format": "arcttt-lora-v1",
        "base": "Qwen2.5-0.5B-Instruct",
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "use_rslora": True,
        "serialization": "raw_qwen",
    }
    log: dict = {
        "kind": "micro_train",
        "config": {
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "use_rslora": True,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "grad_accum": grad_accum,
            "max_sequence_tokens": max_sequence_tokens,
            "seed": seed,
            "wrapped_linears": len(wrapped),
            "train_tasks": len(tasks),
            "augmentations": "dihedral-8 sweep, per-epoch rotation",
        },
        "steps": [],
        "optimizer_steps": 0,
        "micro_batches": 0,
        "tokens": 0,
        "skipped_too_long": 0,
        "skipped_nonfinite": 0,
        "skipped_oom": 0,
    }

    started = _time.monotonic()
    accumulated = 0
    running_loss = 0.0
    step = 0

    def checkpoint() -> None:
        log["train_seconds"] = _time.monotonic() - started
        log["tokens_per_second"] = (
            log["tokens"] / log["train_seconds"] if log["train_seconds"] > 0 else 0.0
        )
        save_adapter(model, adapter_path, {**metadata, "optimizer_steps": step})
        _write_atomic(log_path, log)

    # first artifact exists BEFORE any GPU work: a crash mid-training then
    # still leaves a diagnosable adapter+log pair in /kaggle/working
    checkpoint()

    for turns in example_stream(tasks, seed):
        if _time.time() > deadline:
            print(f"[train] deadline reached at step {step}", flush=True)
            break
        if max_steps is not None and step >= max_steps:
            break
        encoded = encoder._encode(turns, supervise_final=True)
        if encoded is None:
            log["skipped_too_long"] += 1
            continue
        input_ids, labels = encoded
        try:
            loss = model(
                input_ids=input_ids,
                attention_mask=_torch.ones_like(input_ids),
                labels=labels,
            ).loss
            if not bool(_torch.isfinite(loss)):
                log["skipped_nonfinite"] += 1
                optimizer.zero_grad()
                accumulated = 0
                running_loss = 0.0
                continue
            (loss / grad_accum).backward()
        except _torch.cuda.OutOfMemoryError:
            optimizer.zero_grad()
            _torch.cuda.empty_cache()
            log["skipped_oom"] += 1
            accumulated = 0
            running_loss = 0.0
            continue
        except RuntimeError as error:
            # cuBLAS/cuDNN allocation failures surface as plain RuntimeError
            # (v8 postmortem lesson). In a script kernel an uncaught exception
            # forfeits the whole run — /kaggle/working is not published on
            # failure — so alloc pressure must cost one batch, not the run.
            message = str(error)
            if (
                "out of memory" not in message
                and "CUBLAS" not in message
                and "CUDNN" not in message
            ):
                raise
            optimizer.zero_grad()
            _torch.cuda.empty_cache()
            if log.get("skipped_alloc_runtime", 0) == 0:
                print(f"first alloc RuntimeError: {message[:300]}", flush=True)
            log["skipped_alloc_runtime"] = log.get("skipped_alloc_runtime", 0) + 1
            accumulated = 0
            running_loss = 0.0
            continue
        log["micro_batches"] += 1
        log["tokens"] += int(input_ids.numel())
        running_loss += float(loss.item())
        accumulated += 1
        if accumulated < grad_accum:
            continue
        lr_scale = min(1.0, (step + 1) / warmup_steps) if warmup_steps else 1.0
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * lr_scale
        _torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        log["optimizer_steps"] = step
        log["steps"].append(
            {
                "step": step,
                "loss": running_loss / grad_accum,
                "lr": learning_rate * lr_scale,
                "tokens": log["tokens"],
                "seconds": round(_time.monotonic() - started, 1),
            }
        )
        accumulated = 0
        running_loss = 0.0
        if step % checkpoint_every == 0:
            checkpoint()
            recent = log["steps"][-1]
            print(
                f"[train] step {step} | loss {recent['loss']:.4f} | "
                f"{log['tokens_per_second']:.0f} tok/s | "
                f"oom {log['skipped_oom']} long {log['skipped_too_long']}",
                flush=True,
            )

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval()
    checkpoint()
    return log


# -- paired held-out eval ---------------------------------------------------


class FrozenPredictor(CausalLMPredictor):
    """Predictor whose adapt() is a no-op: measures the model AS IS.

    solve_task unconditionally calls adapt(), which would strip and replace
    the continued-training adapter; the paired eval must never do that —
    it compares base weights vs base+our adapter, with zero test-time
    training on either arm."""

    def adapt(self, task: Task, augmentations: Sequence[Augmentation]) -> None:
        self.base_model.eval()


def eval_configs() -> tuple[TTTConfig, SolveConfig]:
    ttt = TTTConfig(
        epochs=0,  # no TTT: the eval isolates the continued-training delta
        max_new_tokens=992,
        max_sequence_tokens=EVAL_MAX_SEQ_TOKENS,
        raw_qwen_format=True,
        use_dfs=True,
        dfs_probability_cutoff=0.1,
        dfs_max_candidates=16,
        dfs_time_budget_seconds=EVAL_DFS_BUDGET_SECONDS,
    )
    solve = SolveConfig(
        augmentations=AUG_QUARTET,
        samples_per_augmentation=1,
        rescore_augmentations=AUG_QUARTET,
    )
    return ttt, solve


def run_eval(
    model: object,
    tokenizer: object,
    device: _torch.device,
    eval_tasks: Sequence[Task],
    label: str,
    deadline: float | None = None,
    out_path: str | None = None,
) -> dict:
    """Small-budget solve of the fixed eval slice + teacher-forced lp(true)."""

    ttt, solve = eval_configs()
    predictor = FrozenPredictor(model, tokenizer, ttt, device)
    report: dict = {"label": label, "tasks": {}, "solved_pairs": 0, "scored_pairs": 0}
    lp_values: list[float] = []
    for task in eval_tasks:
        if deadline is not None and _time.time() > deadline:
            report["tasks"][task.task_id] = {"skipped": "deadline"}
            if out_path is not None:
                _write_atomic(out_path, report)
            continue
        started = _time.monotonic()
        entry: dict = {}
        try:
            ranked = solve_task(task, predictor, solve)
            solved = 0
            for test_index, attempts in enumerate(ranked):
                solution = task.test[test_index].output
                if solution is None:
                    continue
                report["scored_pairs"] += 1
                if attempts and score_attempts(list(attempts), solution):
                    solved += 1
                    report["solved_pairs"] += 1
            entry["solved_pairs"] = solved
            lp_true = [
                predictor.log_probabilities(task, test_index, [pair.output])[0]
                for test_index, pair in enumerate(task.test)
                if pair.output is not None
            ]
            finite = [v for v in lp_true if _math.isfinite(v)]
            entry["lp_true_per_token"] = finite
            lp_values.extend(finite)
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        entry["seconds"] = round(_time.monotonic() - started, 1)
        report["tasks"][task.task_id] = entry
        print(f"[eval:{label}] {task.task_id}: {entry}", flush=True)
        if out_path is not None:  # partial results survive a killed worker
            _write_atomic(out_path, report)
    report["mean_lp_true"] = sum(lp_values) / len(lp_values) if lp_values else None
    if out_path is not None:
        _write_atomic(out_path, report)
    return report


def _before_eval_worker(
    rank: int, model_dir: str, eval_tasks: list[Task], out_path: str, deadline: float
) -> None:
    """Second-T4 worker: baseline eval of the stock model, in parallel with
    training on GPU 0 (single-GPU sessions run this inline instead).

    Writes the report atomically after EVERY task, so a killed or wedged
    worker still leaves the completed tasks on disk for the paired means."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _torch.device(f"cuda:{rank}")
    bf16 = _bf16_compute_works(device)
    dtype = _torch.bfloat16 if bf16 else _torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device)
    model.eval()
    run_eval(
        model, tokenizer, device, eval_tasks, "before",
        deadline=deadline, out_path=out_path,
    )


# -- main -------------------------------------------------------------------


def main() -> None:
    started = _time.time()
    train_deadline = started + WALL_BUDGET_SECONDS - AFTER_EVAL_RESERVE - SAVE_MARGIN
    final_deadline = started + WALL_BUDGET_SECONDS

    if not _torch.cuda.is_available():
        raise RuntimeError("no CUDA device — accelerator pin failed; refusing to burn quota on CPU")

    model_dir = _find_model_dir()
    print("model dir:", model_dir, flush=True)
    train_challenges = _find_one(["*training_challenges.json"])
    train_solutions = _find_one(["*training_solutions.json"])
    eval_challenges = _find_one(["*evaluation_challenges.json"])
    eval_solutions = _find_one(["*evaluation_solutions.json"])
    # printed so the data boundary (training-only optimizer input, held-out
    # evaluation slice) is auditable from the log alone
    print(
        f"data files: train={train_challenges},{train_solutions} "
        f"eval={eval_challenges},{eval_solutions}",
        flush=True,
    )
    train_tasks = _load_split(train_challenges, train_solutions)
    eval_tasks = _load_split(eval_challenges, eval_solutions)
    eval_slice = sorted(eval_tasks, key=lambda task: task.task_id)[:EVAL_TASK_COUNT]
    print(
        f"{len(train_tasks)} training tasks | eval slice: "
        f"{[task.task_id for task in eval_slice]}",
        flush=True,
    )

    device = _torch.device("cuda:0")
    bf16 = _bf16_compute_works(device)
    dtype = _torch.bfloat16 if bf16 else _torch.float32
    if not bf16:
        print(
            "WARNING: bf16 compute unavailable — training in float32 "
            "(fp16 training without a loss scaler is unsafe); expect fewer "
            "steps in the same wall budget. If this appears, the T4 pin failed.",
            flush=True,
        )
    print(
        f"{_torch.cuda.get_device_name(0)} | bf16={bf16} | dtype={dtype} | "
        f"gpus={_torch.cuda.device_count()}",
        flush=True,
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device)

    # Baseline eval: parallel on the second T4 when present, else inline
    # BEFORE any adapter is injected.
    before_proc = None
    before_path = "eval_before.json"
    if _torch.cuda.device_count() >= 2:
        context = _mp.get_context("spawn")
        before_proc = context.Process(
            target=_before_eval_worker,
            args=(1, model_dir, list(eval_slice), before_path, train_deadline),
        )
        before_proc.start()
        print("baseline eval running on cuda:1 in parallel", flush=True)
    else:
        report = run_eval(
            model, tokenizer, device, eval_slice, "before",
            deadline=started + 45 * 60,  # never let the baseline eat the train window
        )
        _write_atomic(before_path, report)

    log = train_micro(
        model,
        tokenizer,
        train_tasks,
        device,
        deadline=train_deadline,
        adapter_path="adapter_micro.safetensors",
        log_path="train_log.json",
    )
    log["dtype"] = str(dtype)
    log["model_dir"] = model_dir

    if before_proc is not None:
        before_proc.join(timeout=max(60.0, train_deadline + 5 * 60 - _time.time()))
        if before_proc.is_alive():
            print("terminating straggler baseline-eval worker", flush=True)
            before_proc.terminate()
            before_proc.join(timeout=60)
        print(f"baseline-eval worker exitcode: {before_proc.exitcode}", flush=True)

    try:
        log["eval_before"] = _json.loads(open(before_path).read())
    except Exception as error:
        log["eval_before"] = {"error": f"unreadable: {type(error).__name__}"}

    # After-arm: the adapter is still injected from training.
    log["eval_after"] = run_eval(
        model, tokenizer, device, eval_slice, "after", deadline=final_deadline
    )
    log["wall_seconds"] = _time.time() - started
    _write_atomic("train_log.json", log)

    before, after = log["eval_before"], log["eval_after"]
    # The headline lp delta is only meaningful over tasks BOTH arms scored:
    # deadline skips can desynchronize the arms, and unpaired means mislead.
    paired_before: list[float] = []
    paired_after: list[float] = []
    for task_id, before_entry in (before.get("tasks") or {}).items():
        after_entry = (after.get("tasks") or {}).get(task_id) or {}
        b_vals = before_entry.get("lp_true_per_token") or []
        a_vals = after_entry.get("lp_true_per_token") or []
        if b_vals and a_vals:
            paired_before.extend(b_vals)
            paired_after.extend(a_vals)
    log["paired_mean_lp_before"] = (
        sum(paired_before) / len(paired_before) if paired_before else None
    )
    log["paired_mean_lp_after"] = (
        sum(paired_after) / len(paired_after) if paired_after else None
    )
    _write_atomic("train_log.json", log)
    print(
        "REGISTRY | micro-train | "
        f"steps {log['optimizer_steps']} | tokens {log['tokens']} | "
        f"{log.get('tokens_per_second', 0):.0f} tok/s | "
        f"before {before.get('solved_pairs')}/{before.get('scored_pairs')} "
        f"lp {before.get('mean_lp_true')} | "
        f"after {after.get('solved_pairs')}/{after.get('scored_pairs')} "
        f"lp {after.get('mean_lp_true')} | "
        f"PAIRED lp {log['paired_mean_lp_before']} -> {log['paired_mean_lp_after']} | "
        f"{log['wall_seconds'] / 60:.0f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
