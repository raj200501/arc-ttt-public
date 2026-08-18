"""Kaggle kernel entry: G-E2 GATING ARMS ONLY (k=10), then exit.

Why this exists, measured 2026-08-11: a RUNNING Kaggle kernel exposes no
partial output and no log - `kaggle kernels output` on a live kernel returns
an empty directory. So a full 18-arm rung locks its results until the whole
session ends (up to 11 h), even though the six k=10 arms that alone decide
G-E2 finish in roughly the first third of it. Waiting 11 h for a number that
existed at 3-4 h costs a day of iteration on the project's critical path.

This build runs ONLY k=10 x seeds {1,2,3} x {adapted, kshot} and exits, so
the session ends when the deciding number exists and the artifacts become
readable immediately. Everything else - models, frozen hyperparameters,
vote/rescore, resume-by-dataset-seeding, per-arm atomic writes - is identical
to entry_cord_scale.py, so arms from either build are interchangeable and
pair cleanly (same environment, same "device" provenance tag).

Run the curve arms (k=5, k=30) afterwards with the full build; they are
reported for curve shape and do not gate.

THIS BUILD is pinned to the 1.5B rung on CPU, for two measured reasons
(2026-08-11):

  * The auto-plan build walks ["0.5b", "1.5b"] and never reaches 1.5b. On
    free Kaggle CPU a 0.5b arm costs 1.3-1.5 h, so the rung's 18 arms need
    ~25 h against a 12 h session wall - the first session burned 10.3 h and
    finished 9 arms, all 0.5b. The 1.5B rung had zero scheduled compute.
  * Addendum A SS A.4 branch 2 calls a PASS AT 1.5B the strongest commercial
    outcome, because the wedge then runs without GPUs. With the weekly T4
    quota spent (33.29/30 h, refresh 08-15), free CPU is the only compute
    that can reach a scale number before Saturday.

Six arms at ~3x the 0.5b per-arm cost is ~23 h, so this spans two sessions;
skip-if-exists plus dataset seeding resumes it. Weights stay float32, the
same dtype as the 0.5b rung, so the cross-rung scaling comparison carries no
dtype confound (the bf16 4B CPU attempt does - see that entry's caveat).
"""

from __future__ import annotations

import glob as _glob
import json as _json
import os as _os
import random as _random
import time as _time

import torch as _torch

from arcttt.model import TTTConfig
from arcttt.text_task import from_cord_gt
from arcttt.text_ttt import TextPredictor, predict_text_voted, score_text_output

RUNGS = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "4b": "Qwen/Qwen3-4B-Instruct-2507",
}
EVAL_N = 20
_ARM_IDENTITY = {"rung", "k", "seed", "arm"}
POOL_SAMPLES = 5  # 1 greedy + 4 sampled (T=0.7, model config default)
DATE = "2026-08-11"
WALL_BUDGET_SECONDS = 11.0 * 3600
MARGIN_SECONDS = 25 * 60  # stop starting arms this close to the deadline

# k=10 arms first: they alone gate G-E2 (mean paired delta over seeds 1-3).
ARM_ORDER = [  # GATE-ONLY build: k=10 arms, the six that decide G-E2
    (k, seed, arm)
    for k in (10,)
    for seed in (1, 2, 3)
    for arm in ("adapted", "kshot")
]


def _find_data() -> str:
    matches = _glob.glob("/kaggle/input/**/cord_validation.jsonl", recursive=True)
    if not matches:
        raise RuntimeError("cord_validation.jsonl not found under /kaggle/input")
    return matches[0]


def _write_atomic(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    open(tmp, "w").write(_json.dumps(payload, indent=2))
    _os.replace(tmp, path)


def _seed_resume_artifacts() -> int:
    """Copy previously landed KERNEL arm artifacts from attached inputs to cwd.

    A fresh kernel version starts with a clean /kaggle/working; attaching a
    prior run's output lets skip-if-exists resume across pushes, not just
    within one session.

    Only kernel-produced artifacts may seed a kernel run. An artifact records
    a "device" field iff this entry wrote it, so its absence marks a
    locally-produced arm. Seeding one of those makes the kernel skip that arm
    and leaves its pair split across two environments — which is exactly the
    contamination cord_scale_summary.py refuses to score. Seeding it here
    would turn a hard refusal into a silently missing number, so the filter
    belongs at the source. (Cost of getting this wrong, observed 08-11: the
    0.5b k=10 gate went undecidable on seed 2.)
    """

    import shutil

    seeded = skipped_local = 0
    for path in _glob.glob("/kaggle/input/**/cord_scale_*.json", recursive=True):
        name = _os.path.basename(path)
        if name.startswith("cord_scale_summary"):
            continue
        if _os.path.exists(name):
            continue
        try:
            record = _json.loads(open(path).read())
        except Exception:
            continue
        if not _ARM_IDENTITY.issubset(record):
            # Not an arm at all: hand-written incident records (the 4B CPU OOM
            # postmortem) sit in the same directory and match the same glob,
            # and one of them carries a "device" field. Identity keys, not
            # provenance keys, decide what is an arm - same rule as
            # scripts/cord_scale_summary.py.
            skipped_local += 1
            print(f"not seeding (not an arm record): {name}", flush=True)
            continue
        if "device" not in record or "error" in record:
            # No device tag -> produced locally. Error record -> the arm never
            # produced a number, so seeding it would retire the arm forever
            # instead of retrying it under a changed config (e.g. the 4B
            # fp32 -> bf16 switch). Both re-run here.
            skipped_local += 1
            print(f"not seeding (will re-run here): {name}", flush=True)
            continue
        shutil.copy(path, name)
        seeded += 1
    if skipped_local:
        print(f"{skipped_local} local-env artifacts ignored for environment homogeneity", flush=True)
    return seeded


def _rung_plan(device: "_torch.device") -> list[str]:
    """GPU session -> the 4B rung. CPU session -> the two CPU rungs in order."""

    override = _os.environ.get("CORD_SCALE_RUNG")
    if override:
        return [override]
    return ["1.5b"]  # this build is pinned to the 1.5B rung on ANY device


def run_rung(rung: str, rows: list, deadline: float, device: "_torch.device") -> None:
    model_id = RUNGS[rung]
    print(f"rung {rung} -> {model_id}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _torch.bfloat16 if device.type == "cuda" else _torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)

    done = skipped = 0
    for k, seed, arm in ARM_ORDER:
        if _time.time() > deadline - MARGIN_SECONDS:
            print(f"deadline margin reached; stopping before k{k} seed{seed} {arm}", flush=True)
            break
        out_path = f"cord_scale_{rung}_k{k}_seed{seed}_{arm}_{DATE}.json"
        if _os.path.exists(out_path):
            skipped += 1
            print(f"skip (exists): {out_path}", flush=True)
            continue
        arm_started = _time.monotonic()
        rng = _random.Random(seed)
        shuffled = list(rows)
        rng.shuffle(shuffled)  # per-seed reshuffle, same as the dev sweep
        task = from_cord_gt(
            shuffled[:k],
            shuffled[k : k + EVAL_N],
            task_id=f"cord-scale-{rung}-k{k}-seed{seed}",
        )
        config = TTTConfig(
            lora_rank=16,
            lora_alpha=32,
            epochs=1 if arm == "adapted" else 0,
            max_new_tokens=512,
            max_sequence_tokens=8192 if k == 30 else 4096,
            # unconditional: math-identical, and CPU sessions OOM without it
            gradient_checkpointing=True,
            shuffle_examples=True,
        )
        try:
            predictor = TextPredictor(model, tokenizer, config, device)
            adapt_started = _time.monotonic()
            predictor.adapt_text(task, shuffle_seeds=(seed,))
            adapt_seconds = _time.monotonic() - adapt_started

            results = []
            exact = 0
            f1_sum = 0.0
            scored = 0
            invalid = 0
            no_completion = 0
            for index in range(len(task.test)):
                gold = task.test[index].output_text
                assert gold is not None
                selected = predict_text_voted(predictor, task, index, samples=POOL_SAMPLES)
                if selected is None:
                    no_completion += 1
                    results.append({"index": index, "error": "no completion"})
                    continue
                score = score_text_output(selected, gold)
                scored += 1
                exact += int(score.exact_match)
                invalid += int(not score.valid_json)
                f1_sum += score.micro_f1
                results.append(
                    {
                        "index": index,
                        "valid_json": score.valid_json,
                        "exact_match": score.exact_match,
                        "micro_f1": round(score.micro_f1, 4),
                    }
                )
            report = {
                "spec": "ENTERPRISE_EVAL_SPEC.md Addendum A scaled run (frozen 2026-08-08)",
                "dataset": "naver-clova-ix/cord-v2 (CC BY 4.0), text-only post-OCR",
                "rung": rung,
                "model": model_id,
                "arm": arm,
                "k": k,
                "eval_n": EVAL_N,
                "seed": seed,
                "decode": "vote/rescore ON: 1 greedy + 4 sampled (T=0.7), "
                "canonical-JSON pooling, count+likelihood top-1",
                "config": {
                    "rank": 16,
                    "alpha": 32,
                    "epochs": config.epochs,
                    "max_new_tokens": 512,
                    "max_seq": config.max_sequence_tokens,
                },
                "device": str(device),
                "adapt_seconds": round(adapt_seconds, 1),
                "exact_match": exact,
                "scored": scored,
                "invalid_json": invalid,
                "no_completion": no_completion,
                "mean_micro_f1": round(f1_sum / scored, 4) if scored else 0.0,
                "results": results,
            }
            _write_atomic(out_path, report)
            done += 1
            print(
                _json.dumps(
                    {
                        "artifact": out_path,
                        "mean_micro_f1": report["mean_micro_f1"],
                        "invalid_json": invalid,
                        "wall_seconds": round(_time.monotonic() - arm_started, 1),
                    }
                ),
                flush=True,
            )
        except _torch.OutOfMemoryError:
            _torch.cuda.empty_cache()
            print(f"OOM: k{k} seed{seed} {arm} — recorded, not imputed", flush=True)
            _write_atomic(
                out_path,
                {
                    "rung": rung,
                    "arm": arm,
                    "k": k,
                    "seed": seed,
                    "device": str(device),  # provenance stays uniform across outcomes
                    "error": "oom",
                    "note": "arm OOMed at frozen config; no number imputed",
                },
            )
        except Exception as error:  # an arm failure must not kill the rung
            import traceback

            print(f"ERROR k{k} seed{seed} {arm}: {type(error).__name__}: {error}", flush=True)
            if done == 0 and skipped <= 3:  # full context once, not 18 times
                traceback.print_exc()
    del model
    print(f"rung {rung}: {done} arms run, {skipped} skipped", flush=True)


def main() -> None:
    started = _time.time()
    deadline = started + WALL_BUDGET_SECONDS
    rows = [_json.loads(line) for line in open(_find_data()).read().splitlines()]
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    seeded = _seed_resume_artifacts()
    print(f"{len(rows)} receipts | device {device} | {seeded} artifacts resumed", flush=True)
    for rung in _rung_plan(device):
        if _time.time() > deadline - MARGIN_SECONDS:
            print(f"deadline margin reached; rung {rung} not started", flush=True)
            break
        run_rung(rung, rows, deadline, device)
    print(f"session done | {(_time.time()-started)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
