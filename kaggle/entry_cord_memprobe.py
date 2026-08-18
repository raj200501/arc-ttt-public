"""Kaggle kernel entry: does a 1.5B fp32 CORD arm FIT on a CPU session?

Measured facts this exists to reconcile (2026-08-11):
  * 0.5B fp32 on Kaggle CPU: nine arms completed, no OOM.
  * 4B fp32 on Kaggle CPU: SIGKILLed during the TTT backward, 0 arms.
  * 1.5B sits between them and is the rung Addendum A SS A.4 calls the
    strongest commercial outcome, so its six gate arms are goal 2's only
    path that needs no GPU quota.
  * A RUNNING kernel exposes NOTHING - `kaggle kernels output` and
    `kaggle kernels logs` both return empty until the session ends. So the
    1.5B gate cannot report an OOM until ~12 h have already been spent.

This probe runs the heaviest single step of one arm - LoRA adaptation over
k=10 receipts at the frozen 4096-token context, then one voted prediction -
and exits. Peak RSS is sampled around each phase. Roughly 20-30 min instead
of 12 h, and it answers the only question that matters before committing a
session: does the backward survive.

It deliberately does NOT write an arm artifact. Its output name is
memprobe_*.json so it can never be picked up as an arm by
scripts/cord_scale_summary.py or seeded as one by a gate build - the arm
identity keys are absent for the same reason.
"""

from __future__ import annotations

import glob as _glob
import json as _json
import os as _os
import random as _random
import resource as _resource
import time as _time

import torch as _torch

from arcttt.model import TTTConfig
from arcttt.text_task import from_cord_gt
from arcttt.text_ttt import TextPredictor, predict_text_voted

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
RUNG = "1.5b"
K = 10
EVAL_N = 1  # one prediction is enough to price the generate path
POOL_SAMPLES = 5  # same pool width as a real arm: 1 greedy + 4 sampled
DATE = "2026-08-11"


def _peak_gb() -> float:
    """Peak RSS of this process. ru_maxrss is KiB on Linux."""

    return round(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1048576, 2)


def _find_data() -> str:
    matches = _glob.glob("/kaggle/input/**/cord_validation.jsonl", recursive=True)
    if not matches:
        raise RuntimeError("cord_validation.jsonl not found under /kaggle/input")
    return matches[0]


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = _time.monotonic()
    report: dict = {
        "probe": "cord 1.5b fp32 CPU fit",
        "model": MODEL_ID,
        "rung": RUNG,  # label only - arm identity keys (k/seed/arm) stay absent
        "k": K,
        "date_utc": DATE,
        "question": "does a 1.5B fp32 LoRA backward at 4096 tokens survive a "
        "Kaggle CPU session, or does it SIGKILL like the 4B fp32 attempt",
    }
    out = f"memprobe_{RUNG}_cpu_{DATE}.json"

    def flush(status: str) -> None:
        report["status"] = status
        report["peak_rss_gb"] = _peak_gb()
        report["elapsed_minutes"] = round((_time.monotonic() - started) / 60, 1)
        tmp = f"{out}.tmp"
        open(tmp, "w").write(_json.dumps(report, indent=2))
        _os.replace(tmp, out)
        print(_json.dumps(report), flush=True)

    rows = [_json.loads(line) for line in open(_find_data()).read().splitlines()]
    device = _torch.device("cpu")
    report["device"] = str(device)
    report["dtype"] = "float32"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=_torch.float32)
    model = model.to(device)
    report["peak_rss_gb_after_load"] = _peak_gb()
    flush("weights_loaded")

    rng = _random.Random(1)
    shuffled = list(rows)
    rng.shuffle(shuffled)  # seed 1, same procedure as the gate arms
    task = from_cord_gt(
        shuffled[:K], shuffled[K : K + EVAL_N], task_id=f"memprobe-{RUNG}"
    )
    config = TTTConfig(
        lora_rank=16,
        lora_alpha=32,
        epochs=1,
        max_new_tokens=512,
        max_sequence_tokens=4096,
        gradient_checkpointing=True,
        shuffle_examples=True,
    )
    predictor = TextPredictor(model, tokenizer, config, device)

    adapt_started = _time.monotonic()
    predictor.adapt_text(task, shuffle_seeds=(1,))
    report["adapt_seconds"] = round(_time.monotonic() - adapt_started, 1)
    report["peak_rss_gb_after_backward"] = _peak_gb()
    flush("backward_survived")

    predict_started = _time.monotonic()
    selected = predict_text_voted(predictor, task, 0, samples=POOL_SAMPLES)
    report["predict_seconds"] = round(_time.monotonic() - predict_started, 1)
    report["got_completion"] = selected is not None
    report["peak_rss_gb_after_predict"] = _peak_gb()

    # The number the gate session is actually budgeted against. The six gate
    # arms are 3 adapted + 3 kshot, and kshot runs epochs=0 - it pays the
    # generate cost only, no backward - so the two arm types are priced
    # separately rather than multiplying the adapted arm by six.
    eval_seconds = report["predict_seconds"] * 20  # EVAL_N=1 here, 20 in a real arm
    report["projected_adapted_arm_minutes"] = round(
        (report["adapt_seconds"] + eval_seconds) / 60, 1
    )
    report["projected_kshot_arm_minutes"] = round(eval_seconds / 60, 1)
    report["projected_six_arm_hours"] = round(
        3
        * (report["projected_adapted_arm_minutes"] + report["projected_kshot_arm_minutes"])
        / 60,
        1,
    )
    report["sessions_needed_at_11h_budget"] = round(
        report["projected_six_arm_hours"] / 10.6 + 0.49
    )
    flush("complete")


if __name__ == "__main__":
    main()
