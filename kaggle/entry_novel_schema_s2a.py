"""Kaggle kernel entry: Addendum B novel-schema gate, 0.5B CPU.

The one experiment that can still rescue the quality thesis, run exactly as
frozen in ENTERPRISE_EVAL_SPEC.md Addendum B (2026-08-12T19:40Z, before any
record existed):

- Corpus: synthetic novel-schema tenants (novel_schema.make_task) — the ONLY
  variable changed from Addendum A. Model, LoRA config, 1 epoch, decode,
  scorer, pairing and seed discipline are Addendum A's frozen values.
- Decision point: k=30. k=10 is a comparability point next to Addendum A's
  k=10 numbers and may NOT be promoted if it alone comes out positive.
- eval_n=60 per seed (180 paired records over seeds {1,2,3}), the power fix:
  MDE ~4 F1 at CORD-observed spread, below the +5 bar for the first time.
- Validity gates BEFORE the delta is read: k-shot mean < 0.15 -> FLOOR
  (task too hard for the rung; delta uninformative); > 0.95 -> CEILING
  (no headroom). Both are stamped into every artifact so the summary can
  refuse to interpret an invalid rung without re-deriving anything.

No dataset file is needed for the corpus — generation is seeded and
deterministic in-kernel (schema seed -> tenant, record seeds disjoint
between train and eval by construction). The attached dataset is used only
to seed RESUME artifacts from prior sessions, same skip-if-exists protocol
as the CORD builds; novel-schema documents are ~12 short lines, so arms are
cheaper per record than CORD arms despite eval_n=60.

Each seed is a DIFFERENT tenant (schema seed = arm seed): the gate then
averages over three novel schemas rather than three draws of one schema,
so a pass cannot be a quirk of one lucky vocabulary.
"""

from __future__ import annotations

import os as _os_early

# Before torch import: lets the CUDA caching allocator grow segments instead
# of hunting for one contiguous block - the v5/v6 OOMs died asking for
# 3.3 GB contiguous while 2.3 GB free + 0.8 GB reserved-unallocated existed.
_os_early.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import glob as _glob
import json as _json
import os as _os
import random as _random
import time as _time

import torch as _torch

from arcttt.model import TTTConfig
from arcttt.novel_schema import make_task
from arcttt.text_ttt import TextPredictor, predict_text_voted, score_text_output
from arcttt.lora import inject_lora, remove_lora

RUNG = "0.5b"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EVAL_N = 60  # Addendum B power fix; CORD's 20 could not resolve the bar
POOL_SAMPLES = 5  # frozen: 1 greedy + 4 sampled (T=0.7)
DATE = "2026-08-12"
FLOOR = 0.15
CEILING = 0.95
_ARM_IDENTITY = {"rung", "k", "seed", "arm"}
WALL_BUDGET_SECONDS = 11.0 * 3600
MARGIN_SECONDS = 25 * 60

# Decision arms (k=30) first, comparability arms (k=10) after — a cancelled
# session should die holding the gate, not the garnish.
# Arm-scoped shard (B.7-r5): ONE k=30 arm per kernel so every arm
# completes well inside Kaggle's 12h CPU cap and saves at natural
# session end (v11 was cancelled ~9h in with zero k=30 arms saved;
# a pair may not fit one session). Races the pair shards under the
# gate is decidable at the slowest shard, not the end of a sequential
# chain. Duplicate-arm policy (decided before banking): first terminal
# kernel's artifact banks; the duplicate is preserved as a free
# same-environment reproducibility datum.
ARM_ORDER = [(30, 2, "adapted")]


def _write_atomic(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    open(tmp, "w").write(_json.dumps(payload, indent=2))
    _os.replace(tmp, path)


def _seed_resume_artifacts() -> int:
    """Copy prior KERNEL novel-schema artifacts from attached inputs.

    Same environment-homogeneity filter as the CORD builds: only artifacts
    carrying a "device" field (kernel-produced) and no "error" field seed a
    resume. There are no local novel-schema arms today, but the filter is
    what KEEPS that true tomorrow.
    """

    import shutil

    seeded = 0
    for path in _glob.glob("/kaggle/input/**/novel_schema_*.json", recursive=True):
        name = _os.path.basename(path)
        if _os.path.exists(name):
            continue
        try:
            record = _json.loads(open(path).read())
        except Exception:
            continue
        if not _ARM_IDENTITY.issubset(record):
            continue
        if "device" not in record or "error" in record:
            print(f"not seeding (will re-run here): {name}", flush=True)
            continue
        shutil.copy(path, name)
        seeded += 1
    return seeded


def _seed_resume_checkpoints() -> int:
    """Copy B.7-r6 checkpoint files (novel_ckpt_*) from attached inputs.

    Named outside every novel_schema_* glob on purpose: the banker, the
    artifact seeding filter and the r4 purge must never see them.
    """

    import shutil

    seeded = 0
    for path in _glob.glob("/kaggle/input/**/novel_ckpt_*", recursive=True):
        name = _os.path.basename(path)
        if not _os.path.exists(name):
            shutil.copy(path, name)
            seeded += 1
    return seeded


def main() -> None:
    started = _time.time()
    deadline = started + WALL_BUDGET_SECONDS
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Fail FAST on a wrong-architecture draw. Kaggle's batch pool holds
        # P100s (cc 6.0), and this image's torch ships no sm_60 kernel
        # images - every op raises cudaErrorNoKernelImageForDevice, the
        # per-arm handler swallows it, and the session ends COMPLETE with
        # zero arms (observed 2026-08-15, novel-schema v2; same class as
        # ARC incident v6). Two minutes of loud failure beats an hour of
        # silent nothing; the push side pins --accelerator NvidiaTeslaT4,
        # and this probe is the backstop if the pin is ever ignored.
        major, minor = _torch.cuda.get_device_capability()
        name = _torch.cuda.get_device_name()
        if major < 7:
            print(f"WRONG GPU: {name} (cc {major}.{minor}) has no kernel "
                  "images in this torch build; exiting for re-push on T4",
                  flush=True)
            raise SystemExit(1)
        print(f"gpu ok: {name} (cc {major}.{minor})", flush=True)
    seeded = _seed_resume_artifacts()
    ckpts = _seed_resume_checkpoints()
    if ckpts:
        print(f"seeded {ckpts} checkpoint files", flush=True)
    if device.type == "cpu":
        # B.7-r4: the k=30 pairs run on CPU/fp32 - the proven path. Purge
        # any seeded k=30 artifact from the GPU attempts so both sides of
        # every pair are produced here.
        purged = 0
        for name in list(_glob.glob("novel_schema_*_k30_*.json")):
            try:
                record = _json.loads(open(name).read())
            except Exception:
                continue
            if not (
                record.get("device") == "cpu"
                and record.get("dtype") == "torch.float32"
            ):
                _os.remove(name)
                purged += 1
        if purged:
            print(f"purged {purged} non-cpu/fp32 k30 artifacts", flush=True)
    if device.type == "cuda":
        # Purge seeded k=30 arms that were NOT produced under fp16: their
        # pairs must be dtype-homogeneous with the fp16 adapted arms this
        # run produces. k=10 pairs stay bf16/bf16 (both sides done) and are
        # untouched.
        purged = 0
        for name in list(_glob.glob("novel_schema_*_k30_*.json")):
            try:
                record = _json.loads(open(name).read())
            except Exception:
                continue
            if record.get("dtype") != "torch.float16":
                _os.remove(name)
                purged += 1
        if purged:
            print(f"purged {purged} non-fp16 k30 artifacts for pair homogeneity", flush=True)
    print(f"device {device} | {seeded} artifacts resumed", flush=True)
    print(f"rung {RUNG} -> {MODEL_ID}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _torch.bfloat16 if device.type == "cuda" else _torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if device.type == "cuda":
        # B.7-r3: fp16, not bf16. v5-v7 forensics: SDPA's memory-efficient
        # kernel is sm80+ for bf16, so on this sm75 T4 every path (eager,
        # sdpa-math) materializes the T^2 attention per layer and the k=30
        # trunk backward dies at the same 3.3 GB fp32 softmax buffer -
        # with chunked loss AND verified gradient checkpointing. fp16 IS
        # mem-efficient-eligible on sm75: attention memory goes linear in T.
        # Pair homogeneity is preserved by re-running the k=30 KSHOT arms in
        # fp16 too (the purge below); dtype is stamped into every artifact.
        dtype = _torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=dtype, attn_implementation="sdpa"
        ).to(device)
        print(f"attention: sdpa | dtype: {dtype}", flush=True)
    except (TypeError, ValueError) as error:
        print(f"sdpa unavailable ({error}); falling back to default attention", flush=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)

    done = skipped = 0
    for k, seed, arm in ARM_ORDER:
        if _time.time() > deadline - MARGIN_SECONDS:
            print(f"deadline margin reached; stopping before k{k} seed{seed} {arm}", flush=True)
            break
        out_path = f"novel_schema_{RUNG}_k{k}_seed{seed}_{arm}_{DATE}.json"
        if _os.path.exists(out_path):
            skipped += 1
            print(f"skip (exists): {out_path}", flush=True)
            continue
        arm_started = _time.monotonic()
        if device.type == "cuda":
            _torch.cuda.empty_cache()
            _torch.cuda.reset_peak_memory_stats()
        # seed -> tenant AND draws: paired arms at (k, seed) share the exact
        # corpus; different seeds are different invented schemas.
        task, schema = make_task(
            seed=seed,
            n_train=k,
            n_test=EVAL_N,
            task_id=f"novel-{RUNG}-k{k}-seed{seed}",
        )
        config = TTTConfig(
            lora_rank=16,
            lora_alpha=32,
            epochs=1 if arm == "adapted" else 0,
            max_new_tokens=512,
            max_sequence_tokens=8192 if k == 30 else 4096,
            # B.7-r2: identical math, sliced logits — the full seq x vocab
            # logits tensor OOMed the T4 on every k=30 adapted arm (all
            # three recorded as error artifacts, 2026-08-15). Gradient
            # equivalence to the labels path is pinned by
            # tests/test_chunked_loss.py before this could ship.
            chunked_loss_tokens=512,
            gradient_checkpointing=True,
            shuffle_examples=True,
        )
        try:
            predictor = TextPredictor(model, tokenizer, config, device)
            # B.7-r6: cancellation-proofing. The adapter is saved once after
            # training and every scored doc is journaled; a relaunch seeded
            # with these files resumes instead of restarting. Same frozen
            # computation - the adapter weights are restored bit-identically
            # and completed docs are not re-decoded.
            ckpt_stem = f"novel_ckpt_{RUNG}_k{k}_seed{seed}_{arm}"
            adapter_path = f"{ckpt_stem}_adapter.pt"
            docs_path = f"{ckpt_stem}_docs.jsonl"
            resumed_adapter = False
            if arm == "adapted" and _os.path.exists(adapter_path):
                saved = _torch.load(adapter_path, map_location="cpu")
                remove_lora(model)
                inject_lora(model, config.lora_rank, config.lora_alpha, use_rslora=True)
                lora_state = {n: p for n, p in model.named_parameters() if "lora_" in n}
                if set(saved) != set(lora_state):
                    raise SystemExit(f"adapter checkpoint mismatch: {adapter_path}")
                with _torch.no_grad():
                    for n, p in lora_state.items():
                        p.copy_(saved[n].to(p.device, p.dtype))
                model.eval()  # parity with post-adapt state (adapt_on_examples ends in eval())
                adapt_seconds = _json.loads(open(f"{ckpt_stem}_meta.json").read())["adapt_seconds"]
                resumed_adapter = True
                print(f"resumed adapter: {adapter_path}", flush=True)
            else:
                adapt_started = _time.monotonic()
                predictor.adapt_text(task, shuffle_seeds=(seed,))
                adapt_seconds = _time.monotonic() - adapt_started
                if arm == "adapted":
                    _adapter_tmp = f"{adapter_path}.tmp"
                    _torch.save(
                        {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n},
                        _adapter_tmp,
                    )
                    _os.replace(_adapter_tmp, adapter_path)
                    _write_atomic(f"{ckpt_stem}_meta.json", {"adapt_seconds": adapt_seconds})

            results = []
            exact = 0
            f1_sum = 0.0
            scored = 0
            invalid = 0
            no_completion = 0
            done_indices = set()
            if _os.path.exists(docs_path):
                for line in open(docs_path):
                    line = line.strip()
                    if not line:
                        continue
                    row = _json.loads(line)
                    done_indices.add(row["index"])
                    if "error" in row:
                        no_completion += 1
                        results.append({"index": row["index"], "error": row["error"]})
                        continue
                    scored += 1
                    exact += int(row["exact_match"])
                    invalid += int(not row["valid_json"])
                    f1_sum += row["micro_f1_raw"]
                    results.append(
                        {
                            "index": row["index"],
                            "valid_json": row["valid_json"],
                            "exact_match": row["exact_match"],
                            "micro_f1": round(row["micro_f1_raw"], 4),
                        }
                    )
                print(f"resumed {len(done_indices)} docs from journal", flush=True)
            docs_log = open(docs_path, "a")
            for index in range(len(task.test)):
                if index in done_indices:
                    continue
                gold = task.test[index].output_text
                assert gold is not None
                selected = predict_text_voted(predictor, task, index, samples=POOL_SAMPLES)
                if selected is None:
                    no_completion += 1
                    results.append({"index": index, "error": "no completion"})
                    docs_log.write(_json.dumps({"index": index, "error": "no completion"}) + "\n")
                    docs_log.flush()
                    _os.fsync(docs_log.fileno())
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
                docs_log.write(_json.dumps(
                    {
                        "index": index,
                        "valid_json": score.valid_json,
                        "exact_match": score.exact_match,
                        "micro_f1_raw": score.micro_f1,
                    }
                ) + "\n")
                docs_log.flush()
                _os.fsync(docs_log.fileno())
            docs_log.close()
            results.sort(key=lambda row: row["index"])
            mean_f1 = round(f1_sum / scored, 4) if scored else 0.0
            validity = "ok"
            if arm == "kshot":  # Addendum B B.5: judged on the BASELINE arm
                if mean_f1 < FLOOR:
                    validity = "floor"
                elif mean_f1 > CEILING:
                    validity = "ceiling"
            report = {
                "spec": "ENTERPRISE_EVAL_SPEC.md Addendum B (frozen 2026-08-12T19:40Z)",
                "dataset": "synthetic novel-schema tenants (novel_schema.py), no external data",
                "tenant": schema.tenant_id,
                "schema": schema.describe(),  # artifact-only; never in any prompt
                "rung": RUNG,
                "model": MODEL_ID,
                "arm": arm,
                "k": k,
                "eval_n": EVAL_N,
                "resumed": bool(resumed_adapter or done_indices),
                "seed": seed,
                "gate_role": "DECISION" if k == 30 else "comparability-only (may not be promoted)",
                "validity": validity,
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
                "dtype": str(dtype),
                "adapt_seconds": round(adapt_seconds, 1),
                "exact_match": exact,
                "scored": scored,
                "invalid_json": invalid,
                "no_completion": no_completion,
                "mean_micro_f1": mean_f1,
                "results": results,
            }
            _write_atomic(out_path, report)
            done += 1
            print(
                _json.dumps(
                    {
                        "artifact": out_path,
                        "mean_micro_f1": mean_f1,
                        "validity": validity,
                        "invalid_json": invalid,
                        "wall_seconds": round(_time.monotonic() - arm_started, 1),
                    }
                ),
                flush=True,
            )
        except _torch.OutOfMemoryError:
            import traceback

            # The site matters: v4 proved the LOSS was not the (only) hog,
            # so a bare "OOM" line hides the next bug. Print where and how
            # much before recording.
            traceback.print_exc()
            if device.type == "cuda":
                print(
                    f"cuda mem: allocated {_torch.cuda.memory_allocated()/1e9:.2f} GB, "
                    f"peak {_torch.cuda.max_memory_allocated()/1e9:.2f} GB, "
                    f"reserved {_torch.cuda.memory_reserved()/1e9:.2f} GB",
                    flush=True,
                )
                _torch.cuda.empty_cache()
            print(f"OOM: k{k} seed{seed} {arm} — recorded, not imputed", flush=True)
            _write_atomic(
                out_path,
                {
                    "rung": RUNG,
                    "arm": arm,
                    "k": k,
                    "seed": seed,
                    "device": str(device),
                    "error": "oom",
                    "note": "arm OOMed at frozen config; no number imputed",
                },
            )
        except Exception as error:
            import traceback

            print(f"ERROR k{k} seed{seed} {arm}: {type(error).__name__}: {error}", flush=True)
            if done == 0 and skipped <= 3:
                traceback.print_exc()
    del model
    print(f"novel-schema {RUNG}: {done} arms run, {skipped} skipped | "
          f"{(_time.time()-started)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
