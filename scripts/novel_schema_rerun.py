#!/usr/bin/env python3
"""Addendum Z — the +46.5 gate, run again with every prediction stored.

The Addendum B gate arms (k=30, seeds {1,2,3}, `experiments/novel_schema_
0.5b_k30_seed*_{adapted,kshot}_2026-08-12.json`) banked per-receipt scores
and discarded the model outputs at generation time, so the most-cited
number in this repository is ARITHMETIC-verifiable at best: a reader can
re-add the scores, not re-derive them. This runner executes the same
frozen recipe — same generator seeds, model snapshot, LoRA config, one
epoch, decode, scorer, pairing, caps — on this CPU box and keeps
everything: every completion in the pool, each distinct completion's
log-probability, the selected text, the parsed score, the prompt hash
and the wall time, per document, plus the adapter weights on disk. The
unchanged reader (`scripts/novel_schema_summary.py`) computes the
verdict on the new arms under `--date 2026-09-23`; this script adds the
four Addendum Z readings by arithmetic and refuses to read — and writes
nothing — until all six arms exist and the verdict is decidable.

    PYTHONPATH=src python3 scripts/novel_schema_rerun.py --run         # all six arms, in order, resumable
    PYTHONPATH=src python3 scripts/novel_schema_rerun.py --run --seed 1 --arm adapted
    PYTHONPATH=src python3 scripts/novel_schema_rerun.py --read        # withholds until every arm exists

Protocol: docs/research/ADDENDUM_Z_PROTOCOL.md (frozen and anchored
before the first arm ran). Sampling is unseeded, exactly as in the banked
run (spec B.9.6), so this is a fresh measurement of the same quantity,
not a byte re-derivation of the old one — the protocol says which of the
two numbers is quoted afterwards, whichever way it moves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RUNG = "0.5b"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# The snapshot the banked kernels resolved is not recorded (they ran with
# internet on and no revision). This is the snapshot cached on this box,
# pinned so the arms are at least a function of a named set of weights,
# chat template and generation defaults.
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
K = 30
EVAL_N = 60
POOL_SAMPLES = 5  # frozen: 1 greedy + 4 sampled (T=0.7)
SEEDS = (1, 2, 3)
ARMS = ("adapted", "kshot")
DATE = "2026-09-23"
BANKED_DATE = "2026-08-12"
FLOOR = 0.15
CEILING = 0.95
THREADS = 4
AGREE_TOL = 0.05  # Z2: |new gate mean - banked gate mean| within this = AGREES
SPREAD_PREDICTION = 0.03  # Z3 prediction, per arm, written before the run
B93_SPREAD = 0.0098  # the single same-environment duplicate of spec B.9.3
SPEC = "ENTERPRISE_EVAL_SPEC.md Addendum B (frozen 2026-08-12T19:40Z)"
PROTOCOL = "docs/research/ADDENDUM_Z_PROTOCOL.md (frozen 2026-09-23, anchored 2026-09-23T0219Z before the first arm)"
ARM_ORDER = [(seed, arm) for seed in SEEDS for arm in ARMS]
OVER_CAP = "prompt over max_sequence_tokens"
EMPTY_POOL = "every completion empty"

DEFAULT_WORK = ROOT / "work" / "z"
DEFAULT_OUT = ROOT / "experiments"
BANKED_DIR = ROOT / "experiments"


# -- small helpers -----------------------------------------------------------


def _write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as handle:
        handle.write(json.dumps(payload, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _fsync_replace(tmp: Path, final: Path) -> None:
    """Rename a fully written temp file into place, durably."""

    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, final)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def artifact_name(seed: int, arm: str, date: str = DATE) -> str:
    return f"novel_schema_{RUNG}_k{K}_seed{seed}_{arm}_{date}.json"


def ckpt_stem(seed: int, arm: str) -> str:
    # deliberately NOT the novel_schema_ prefix: the reader's glob, the
    # banker and the coverage classifier must never see a checkpoint
    return f"novel_ckpt_z_{RUNG}_k{K}_seed{seed}_{arm}"


def environment(model=None, tokenizer=None) -> dict:
    import torch
    import transformers

    cpu = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    out = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
        "dtype": "torch.float32",
        "model_revision": MODEL_REVISION,
    }
    if model is not None:
        out["model_commit_hash"] = getattr(getattr(model, "config", None), "_commit_hash", None)
        gen = getattr(model, "generation_config", None)
        out["generation_config"] = gen.to_dict() if gen is not None else None
    if tokenizer is not None:
        template = getattr(tokenizer, "chat_template", None)
        out["chat_template_sha256"] = _sha256(template) if isinstance(template, str) else None
    return out


# -- the decode, traced ------------------------------------------------------


def voted_with_trace(predictor, task, index: int, samples: int) -> tuple[str | None, dict]:
    """`predict_text_voted`, step for step, returning what it discarded.

    Same calls in the same order: `predict_text` (greedy first, then the
    samples; empty completions dropped), one `log_probabilities_text` per
    DISTINCT completion, `vote_text_candidates`, `select_text_attempts`.
    The returned selection is what `predict_text_voted` would have
    returned for the same pool; tests/test_novel_schema_rerun.py pins
    that with a fake predictor.
    """

    from arcttt.text_ttt import select_text_attempts, vote_text_candidates

    texts = predictor.predict_text(task, index, samples)
    if not texts:
        return None, {"completions": [], "distinct_log_probabilities": [], "candidates": []}
    distinct = list(dict.fromkeys(texts))
    lps = [predictor.log_probabilities_text(task, index, [text])[0] for text in distinct]
    lp_by_text = dict(zip(distinct, lps))
    candidates = vote_text_candidates(texts, [lp_by_text[text] for text in texts])
    selected = select_text_attempts(candidates, attempts=1)
    trace = {
        "completions": list(texts),
        "distinct_log_probabilities": [[text, lp] for text, lp in zip(distinct, lps)],
        "candidates": [
            {
                "text": c.text,
                "key": c.key,
                "found_count": c.found_count,
                "mean_log_probability": c.mean_log_probability,
            }
            for c in candidates
        ],
    }
    return (selected[0] if selected else None), trace


# -- one arm -----------------------------------------------------------------


def _load_journal(path: Path, expected: int | None = None) -> list[dict]:
    """Journal rows, first occurrence per index.

    A torn LAST line (a kill mid-write, typically without its newline) is
    dropped, the file is rewritten without it so the next append cannot
    land on the fragment, and its document is re-decoded. A torn earlier
    line is an error because nothing should have written after it. With
    `expected`, an index outside range(expected) is refused by name so a
    stale journal cannot loop the arm forever on "incomplete".
    """

    rows: list[dict] = []
    seen: set[int] = set()
    if not path.exists():
        return rows
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    torn = False
    for position, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if position == len(lines) - 1:
                print(f"journal: dropping torn last line of {path.name}", flush=True)
                torn = True
                break
            raise
        if expected is not None and not 0 <= int(row["index"]) < expected:
            raise SystemExit(f"journal {path.name}: index {row['index']} outside 0..{expected - 1}")
        if row["index"] in seen:
            continue
        seen.add(row["index"])
        rows.append(row)
    if torn:
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    return rows


def _append_journal(handle, row: dict) -> None:
    handle.write(json.dumps(row) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _prompt_length(predictor, turns) -> tuple[int, str]:
    """Token count and sha256 of a prompt the predictor refused (over the cap)."""

    from arcttt.model import _template_ids, turns_to_chat

    ids = _template_ids(
        predictor.tokenizer.apply_chat_template(
            turns_to_chat(turns), add_generation_prompt=True, return_tensors="pt"
        )
    )
    return int(ids.shape[1]), _sha256(json.dumps(ids[0].tolist()))


def _lora_state(model) -> dict:
    return {n: p for n, p in model.named_parameters() if "lora_" in n}


def _adapter_digest(state: dict) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode())
        h.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run_arm(seed: int, arm: str, model, tokenizer, work: Path, out: Path) -> Path:
    import torch

    from arcttt.lora import inject_lora, remove_lora
    from arcttt.model import TTTConfig
    from arcttt.novel_schema import make_task
    from arcttt.text_ttt import (
        TextPredictor,
        score_text_output,
        text_task_to_messages,
        text_ttt_training_examples,
    )

    out_path = out / artifact_name(seed, arm)
    if out_path.exists():
        print(f"skip (exists): {out_path.name}", flush=True)
        return out_path
    work.mkdir(parents=True, exist_ok=True)
    stem = ckpt_stem(seed, arm)
    adapter_path = work / f"{stem}_adapter.pt"
    meta_path = work / f"{stem}_meta.json"
    docs_path = work / f"{stem}_docs.jsonl"
    device = torch.device("cpu")
    arm_started = time.monotonic()

    # Every process start on this arm after the first is a restart, whether
    # it died during adaptation, between adapter and meta, or mid-journal.
    # Sampling is unseeded (B.9.6) and so is the LoRA-A initialisation, so
    # the count is disclosed in the artifact.
    meta: dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    restarts = int(meta.get("restarts", -1)) + 1
    meta["restarts"] = restarts
    meta.setdefault("started_utc", _utc_now())
    _write_atomic(meta_path, meta)
    if restarts:
        print(f"restart {restarts} of {stem}", flush=True)

    # seed -> tenant AND draws: paired arms at (k, seed) share the exact
    # corpus; different seeds are different invented schemas. Identical
    # call to the banked kernel's.
    task, schema = make_task(
        seed=seed, n_train=K, n_test=EVAL_N, task_id=f"novel-{RUNG}-k{K}-seed{seed}"
    )
    config = TTTConfig(
        lora_rank=16,
        lora_alpha=32,
        epochs=1 if arm == "adapted" else 0,
        max_new_tokens=512,
        max_sequence_tokens=8192,
        chunked_loss_tokens=512,
        gradient_checkpointing=True,
        shuffle_examples=True,
    )
    predictor = TextPredictor(model, tokenizer, config, device)

    # how many LOO training examples the cap admits (tokenization only;
    # the same _encode the trainer applies, so the count is the trainer's)
    examples = text_ttt_training_examples(task, seed)
    encoded_count = sum(
        1 for turns in examples if predictor._encode(turns, supervise_final=True) is not None
    )

    resumed_adapter = False
    adapt_seconds = None
    adapter_sha = None
    if arm == "adapted" and adapter_path.exists():
        # restore instead of retrain — and check what was restored against
        # the digest saved with it; a torn or mismatched file is discarded
        # and the arm retrains (counted above as a restart)
        try:
            saved = torch.load(adapter_path, map_location="cpu")
            remove_lora(model)
            inject_lora(model, config.lora_rank, config.lora_alpha, use_rslora=True)
            state = _lora_state(model)
            if set(saved) != set(state):
                raise ValueError("adapter checkpoint names do not match the model")
            with torch.no_grad():
                for name, param in state.items():
                    param.copy_(saved[name].to(param.device, param.dtype))
            restored_sha = _adapter_digest(state)
            if meta.get("adapter_sha256") not in (None, restored_sha):
                raise ValueError(f"adapter digest {restored_sha[:12]} != saved {meta['adapter_sha256'][:12]}")
        except Exception as error:  # noqa: BLE001 — any unreadable file means retrain, loudly
            print(f"adapter checkpoint unusable ({error}); retraining", flush=True)
            adapter_path.unlink(missing_ok=True)
            meta.pop("adapter_sha256", None)
            meta.pop("adapt_seconds", None)
            _write_atomic(meta_path, meta)
        else:
            model.eval()  # parity with post-adapt state (adapt_on_examples ends in eval())
            adapt_seconds = meta.get("adapt_seconds")  # None if killed between adapter and meta
            adapter_sha = restored_sha
            resumed_adapter = True
            print(f"resumed adapter: {adapter_path.name} ({restored_sha[:12]})", flush=True)
    if not resumed_adapter:
        adapt_started = time.monotonic()
        predictor.adapt_text(task, shuffle_seeds=(seed,))
        adapt_seconds = time.monotonic() - adapt_started
        if arm == "adapted":
            state = {n: p.detach().cpu() for n, p in _lora_state(model).items()}
            adapter_sha = _adapter_digest(state)
            # meta first, then the weights: a kill between the two leaves a
            # meta whose digest names a file that is absent, and the next
            # start retrains; the reverse order would leave weights with no
            # record of how long they took
            meta.update({"adapt_seconds": adapt_seconds, "adapter_sha256": adapter_sha})
            _write_atomic(meta_path, meta)
            tmp = adapter_path.with_name(adapter_path.name + ".tmp")
            torch.save(state, tmp)
            _fsync_replace(tmp, adapter_path)
    if arm == "adapted" and "adapter_sha256" not in meta:
        meta["adapter_sha256"] = adapter_sha
        _write_atomic(meta_path, meta)

    journal = _load_journal(docs_path, expected=len(task.test))
    done = {row["index"] for row in journal}
    if journal:
        print(f"resumed {len(journal)} docs from journal", flush=True)
    handle = open(docs_path, "a")
    try:
        for index in range(len(task.test)):
            if index in done:
                continue
            gold = task.test[index].output_text
            assert gold is not None
            doc_started = time.monotonic()
            prompt = predictor._prompt_ids(text_task_to_messages(task, index))
            if prompt is None:
                # over max_sequence_tokens: no prediction, excluded from the
                # mean (never scored 0) — B.9.2's rule, recorded with the
                # reason and the length that tripped it
                length, over_sha = _prompt_length(predictor, text_task_to_messages(task, index))
                row = {
                    "index": index,
                    "error": "no completion",
                    "reason": OVER_CAP,
                    "prompt_tokens": length,
                    "prompt_sha256": over_sha,
                    "cap": config.max_sequence_tokens,
                    "seconds": round(time.monotonic() - doc_started, 2),
                }
                _append_journal(handle, row)
                print(json.dumps({"seed": seed, "arm": arm, "index": index, "error": OVER_CAP,
                                  "prompt_tokens": length}), flush=True)
                continue
            prompt_sha = _sha256(json.dumps(prompt[0].tolist()))
            selected, trace = voted_with_trace(predictor, task, index, POOL_SAMPLES)
            if selected is None:
                row = {
                    "index": index,
                    "error": "no completion",
                    "reason": EMPTY_POOL,
                    "prompt_tokens": int(prompt.shape[1]),
                    "prompt_sha256": prompt_sha,
                    "seconds": round(time.monotonic() - doc_started, 2),
                    **trace,
                }
                _append_journal(handle, row)
                print(json.dumps({"seed": seed, "arm": arm, "index": index, "error": EMPTY_POOL}), flush=True)
                continue
            score = score_text_output(selected, gold)
            row = {
                "index": index,
                "prompt_tokens": int(prompt.shape[1]),
                "prompt_sha256": prompt_sha,
                **trace,
                "prediction": selected,
                "valid_json": score.valid_json,
                "exact_match": score.exact_match,
                "micro_f1_raw": score.micro_f1,
                "seconds": round(time.monotonic() - doc_started, 2),
            }
            _append_journal(handle, row)
            # progress only — no score is printed before the reading
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "arm": arm,
                        "index": index,
                        "pool": len(trace["completions"]),
                        "distinct": len(trace["distinct_log_probabilities"]),
                        "seconds": row["seconds"],
                    }
                ),
                flush=True,
            )
    finally:
        handle.close()

    journal = sorted(_load_journal(docs_path, expected=len(task.test)), key=lambda r: r["index"])
    if len(journal) != len(task.test):
        raise SystemExit(f"journal incomplete: {len(journal)}/{len(task.test)}")
    # canonical arm mean: raw micro-F1 summed in document-index order over
    # the scored documents, divided by their count, rounded to 4 places
    results = []
    exact = invalid = scored = no_completion = 0
    f1_sum = 0.0
    for row in journal:
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
    mean_f1 = round(f1_sum / scored, 4) if scored else 0.0
    validity = "ok"
    if arm == "kshot":  # B.5: judged on the BASELINE arm
        if mean_f1 < FLOOR:
            validity = "floor"
        elif mean_f1 > CEILING:
            validity = "ceiling"
    report = {
        "spec": SPEC,
        "protocol": PROTOCOL,
        "dataset": "synthetic novel-schema tenants (novel_schema.py), no external data",
        "tenant": schema.tenant_id,
        "schema": schema.describe(),  # artifact-only; never in any prompt
        "rung": RUNG,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "arm": arm,
        "k": K,
        "eval_n": EVAL_N,
        "seed": seed,
        "gate_role": "DECISION",
        "replicates": artifact_name(seed, arm, BANKED_DATE),
        "validity": validity,
        "decode": "vote/rescore ON: 1 greedy + 4 sampled (T=0.7), "
        "canonical-JSON pooling, count+likelihood top-1; the checkpoint's "
        "generation_config defaults apply through model.generate in every "
        "pass, as in the banked run (recorded under environment)",
        "config": {
            "rank": 16,
            "alpha": 32,
            "epochs": config.epochs,
            "max_new_tokens": 512,
            "max_seq": config.max_sequence_tokens,
            "chunked_loss_tokens": config.chunked_loss_tokens,
            "gradient_checkpointing": config.gradient_checkpointing,
            "shuffle_examples": config.shuffle_examples,
            "shuffle_seeds": [seed],
            "learning_rate": config.learning_rate,
        },
        "device": "cpu",
        "dtype": "torch.float32",
        "environment": environment(model, tokenizer),
        "adapter": {
            "training_examples": len(examples),
            "training_examples_within_cap": encoded_count,
            "sha256": adapter_sha,
            "file": adapter_path.name if arm == "adapted" else None,
            "initialisation": "LoRA-A kaiming-uniform from the unseeded global RNG, as in the banked kernels",
        },
        "resumed": bool(restarts),
        "restarts": restarts,
        # ours to report: the anchor proves the protocol's bytes existed by
        # its stamp; that the arm started after it rests on these clocks
        "started_utc": meta.get("started_utc"),
        "banked_utc": _utc_now(),
        "adapt_seconds": round(adapt_seconds, 1) if adapt_seconds is not None else None,
        "decode_seconds": round(sum(float(r.get("seconds", 0.0)) for r in journal), 1),
        # adaptation plus every document's decode, across every process
        # that touched the arm; the wall time of the LAST process only is
        # kept beside it under its own name
        "compute_seconds": round((adapt_seconds or 0.0) + sum(float(r.get("seconds", 0.0)) for r in journal), 1),
        "last_process_wall_seconds": round(time.monotonic() - arm_started, 1),
        "exact_match": exact,
        "scored": scored,
        "invalid_json": invalid,
        "no_completion": no_completion,
        "no_completion_reasons": {
            OVER_CAP: sum(1 for r in journal if r.get("reason") == OVER_CAP),
            EMPTY_POOL: sum(1 for r in journal if r.get("reason") == EMPTY_POOL),
        },
        "mean_micro_f1": mean_f1,
        "results": results,
        # PRIMARY-verifiable: every pooled completion, its log-probability,
        # the selection and the raw score, per document
        "predictions": journal,
    }
    _write_atomic(out_path, report)
    print(json.dumps({"artifact": out_path.name, "scored": scored, "no_completion": no_completion,
                      "compute_seconds": report["compute_seconds"]}), flush=True)
    return out_path


def run(seeds, arms, work: Path, out: Path, threads: int) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32, attn_implementation="sdpa"
    )
    print(json.dumps({"environment": environment(model, tokenizer)}), flush=True)
    for seed, arm in ARM_ORDER:
        if seed not in seeds or arm not in arms:
            continue
        run_arm(seed, arm, model, tokenizer, work, out)
    return 0


# -- readings, by arithmetic -------------------------------------------------


def excluded_indices(record: dict, reason: str | None = None) -> list[int]:
    """Excluded document indices; with `reason`, only those the runner
    journaled under it (the banked arms carry no reason: B.9.2 records
    every exclusion there as over-cap, and that is assumed)."""

    if reason is None or "predictions" not in record:
        return sorted(r["index"] for r in record["results"] if "error" in r)
    return sorted(r["index"] for r in record["predictions"] if r.get("reason") == reason)


def agreement_reading(new_summary: dict, banked_summary: dict, tol: float = AGREE_TOL) -> dict:
    """Z2. AGREES / GO BUT MOVED / DISAGREES, from the two summaries' words and means."""

    new_word = new_summary["VERDICT"]
    old_word = banked_summary["VERDICT"]
    if old_word != "GO":
        raise ValueError("the banked verdict is GO by construction; refusing to read against anything else")
    new_mean = new_summary["gate_k30"].get("mean_delta")
    old_mean = banked_summary["gate_k30"].get("mean_delta")
    if new_word == "UNDECIDABLE":
        return {"reading": "WITHHELD", "detail": "gating pairs missing; nothing is read"}
    out = {
        "new_verdict": new_word,
        "banked_verdict": old_word,
        "new_mean_delta": new_mean,
        "banked_mean_delta": old_mean,
        "tolerance": tol,
    }
    if new_word != old_word:
        out["reading"] = "DISAGREES"
        out["detail"] = (
            f"the re-run reads {new_word} where the banked run read {old_word}; "
            "the headline is withdrawn and the row says so"
        )
        return out
    # the summaries carry means rounded to 4 places; compare at that
    # precision so 0.4648 + 0.05 is not "just past" the tolerance by 4e-17
    moved = round(abs(float(new_mean) - float(old_mean)), 4)
    out["moved_by"] = moved
    if moved <= tol:
        out["reading"] = "AGREES"
        out["detail"] = (
            f"same verdict ({new_word}) and the gate mean moved by {moved:.4f} "
            f"<= {tol}: the primary-verifiable number replaces the banked one on every page"
        )
    else:
        out["reading"] = "GO BUT MOVED"
        out["detail"] = (
            f"both GO but the gate mean moved by {moved:.4f} > {tol}: the "
            "primary-verifiable number replaces the banked one on every page and "
            "the movement publishes beside it at full size"
        )
    return out


def spread_reading(new_arms: dict, banked_arms: dict, prediction: float = SPREAD_PREDICTION) -> dict:
    """Z3. Per-arm |new mean - banked mean|; the max is the cross-environment spread."""

    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            new = new_arms[(K, seed, arm)]
            old = banked_arms[(K, seed, arm)]
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "new": new["mean_micro_f1"],
                    "banked": old["mean_micro_f1"],
                    "abs_diff": round(abs(new["mean_micro_f1"] - old["mean_micro_f1"]), 4),
                }
            )
    largest = max(rows, key=lambda r: r["abs_diff"])
    replaces = largest["abs_diff"] > B93_SPREAD
    return {
        "per_arm": rows,
        "max_abs_diff": largest["abs_diff"],
        "max_at": {"seed": largest["seed"], "arm": largest["arm"]},
        "predicted_max": prediction,
        "prediction_holds": largest["abs_diff"] <= prediction,
        "reading": (
            "WITHIN PREDICTED SPREAD" if largest["abs_diff"] <= prediction else "SPREAD LARGER THAN PREDICTED"
        ),
        "b93_spread": B93_SPREAD,
        "replaces_b93": replaces,
        "detail": (
            f"the largest per-arm difference is {largest['abs_diff']} against the single-pair "
            f"{B93_SPREAD} of spec B.9.3; it publishes by dated erratum beside B.9.3 as an upper "
            "bound on run-to-run spread for these arms (run-to-run plus environment, which this "
            "design cannot separate): "
            + (
                "it is larger, so it is the figure quoted from now on"
                if replaces
                else "it is not larger, so B.9.3's figure stands and this one sits beside it"
            )
        ),
    }


def attrition_reading(new_arms: dict, banked_arms: dict) -> dict:
    """Z4. The over-cap exclusions must be the same documents, seed by seed, arm by arm."""

    rows = []
    same = True
    for seed in SEEDS:
        for arm in ARMS:
            record = new_arms[(K, seed, arm)]
            new = excluded_indices(record, OVER_CAP)
            empty = excluded_indices(record, EMPTY_POOL)
            old = excluded_indices(banked_arms[(K, seed, arm)])
            rows.append({"seed": seed, "arm": arm, "over_cap_new": new, "over_cap_banked": old,
                         "empty_pool_new": empty, "same": new == old})
            same = same and new == old
    return {
        "per_arm": rows,
        "reading": "SAME EXCLUSIONS" if same else "DIFFERENT EXCLUSIONS",
        "banked_assumption": (
            "the banked arms journaled no reason; every banked exclusion is taken as over-cap "
            "per spec B.9.2 (the banked kernel could not produce an empty pool of five non-empty "
            "completions in any recorded arm)"
        ),
        "detail": (
            "the cap is a property of the corpus and the tokenizer, so the over-cap documents "
            "must not depend on the box; a difference here is a tokenizer or generator change, "
            "and when it fires Z2 and Z3 are NOT COMPARABLE and the citation rule is suspended"
        ),
    }


def _six(directory: Path, date: str) -> tuple[dict, list[str]]:
    """The six exact gate artifacts of one date, or the names that are missing/unusable."""

    from novel_schema_summary import IDENTITY

    arms: dict = {}
    problems: list[str] = []
    for seed in SEEDS:
        for arm in ARMS:
            path = directory / artifact_name(seed, arm, date)
            if not path.exists():
                problems.append(f"missing: {path.name}")
                continue
            record = json.loads(path.read_text())
            if not IDENTITY.issubset(record) or "device" not in record:
                problems.append(f"not an arm: {path.name}")
                continue
            if "error" in record:
                problems.append(f"errored arm: {path.name} ({record['error']})")
                continue
            if date == DATE and (record.get("protocol") != PROTOCOL or "predictions" not in record):
                problems.append(f"not an Addendum Z arm: {path.name}")
                continue
            arms[(K, seed, arm)] = record
    return arms, problems


def read(out: Path, banked_dir: Path = BANKED_DIR) -> int:
    from novel_schema_summary import analyse_k
    from novel_schema_summary import main as summary_main
    from novel_schema_summary import verdict

    new_arms, problems = _six(out, DATE)
    banked_arms, banked_problems = _six(banked_dir, BANKED_DATE)
    banked_summary_path = banked_dir / f"novel_schema_summary_{BANKED_DATE}.json"
    if not banked_summary_path.exists():
        banked_problems.append(f"missing: {banked_summary_path.name}")
    if problems or banked_problems:
        print("WITHHELD: nothing is read and nothing is written:", flush=True)
        for line in problems + banked_problems:
            print(f"  {line}", flush=True)
        return 2
    # the unchanged reader's arithmetic, before anything is written: if the
    # gate is UNDECIDABLE on six present arms, nothing lands on disk
    word, _ = verdict(analyse_k(new_arms, K))
    if word == "UNDECIDABLE":
        print("WITHHELD: the unchanged reader returns UNDECIDABLE on six present arms; nothing is written", flush=True)
        return 2
    banked_summary = json.loads(banked_summary_path.read_text())
    z4 = attrition_reading(new_arms, banked_arms)
    comparable = z4["reading"] == "SAME EXCLUSIONS"

    summary_main(["--dir", str(out), "--date", DATE])  # the unchanged reader writes Z1
    new_summary = json.loads((out / f"novel_schema_summary_{DATE}.json").read_text())
    assert new_summary["VERDICT"] == word
    z1 = {
        "reading": word,
        "detail": new_summary["verdict_detail"],
        "gate_k30": new_summary["gate_k30"],
        "reader": "scripts/novel_schema_summary.py, unchanged",
    }
    if comparable:
        z2 = agreement_reading(new_summary, banked_summary)
        z3 = spread_reading(new_arms, banked_arms)
    else:
        suspended = ("NOT COMPARABLE: Z4 read DIFFERENT EXCLUSIONS, so the two runs scored "
                     "different document sets; the citation rule is suspended and the "
                     "discrepancy goes to CORRECTIONS.md before any number moves")
        z2 = {"reading": "NOT COMPARABLE", "detail": suspended,
              "new_mean_delta": new_summary["gate_k30"].get("mean_delta"),
              "banked_mean_delta": banked_summary["gate_k30"].get("mean_delta")}
        z3 = {"reading": "NOT COMPARABLE", "detail": suspended}
    preregistered = {
        "Z1": {"predicted": "GO", "observed": z1["reading"], "holds": z1["reading"] == "GO"},
        "Z2": {"predicted": "AGREES", "observed": z2["reading"], "holds": z2["reading"] == "AGREES"},
        "Z3": {
            "predicted": f"max per-arm spread <= {SPREAD_PREDICTION}",
            "observed": z3.get("max_abs_diff"),
            "holds": bool(z3.get("prediction_holds", False)),
        },
        "Z4": {
            "predicted": "SAME EXCLUSIONS",
            "observed": z4["reading"],
            "holds": z4["reading"] == "SAME EXCLUSIONS",
        },
    }
    headline = {
        "banked": banked_summary["gate_k30"].get("mean_delta"),
        "primary_verifiable": new_summary["gate_k30"].get("mean_delta"),
        "rule": (
            "the primary-verifiable number is the one quoted from now on, whichever way "
            "it moved; the banked number stays as history beside it; if Z1 is not GO the "
            "headline is withdrawn; if Z4 read DIFFERENT EXCLUSIONS nothing moves until "
            "the discrepancy is filed"
        ),
        "quote": (
            new_summary["gate_k30"].get("mean_delta") if (word == "GO" and comparable) else None
        ),
    }
    report = {
        "protocol": PROTOCOL,
        "spec": SPEC,
        "date": DATE,
        "replicates": f"novel_schema_summary_{BANKED_DATE}.json and its six arms",
        "environment": environment(),
        "arms": {
            artifact_name(s, a): {
                "mean_micro_f1": new_arms[(K, s, a)]["mean_micro_f1"],
                "scored": new_arms[(K, s, a)]["scored"],
                "no_completion": new_arms[(K, s, a)]["no_completion"],
                "restarts": new_arms[(K, s, a)].get("restarts"),
                "adapt_seconds": new_arms[(K, s, a)].get("adapt_seconds"),
                "compute_seconds": new_arms[(K, s, a)].get("compute_seconds"),
            }
            for s in SEEDS
            for a in ARMS
        },
        "Z1_gate": z1,
        "Z2_agreement": z2,
        "Z3_spread": z3,
        "Z4_attrition": z4,
        "preregistered": preregistered,
        "headline": headline,
        "verifiability": (
            "this file is AGGREGATE (readings over the six arms); the six arms are PRIMARY: "
            "every pooled completion, its log-probability, the selected text and the raw "
            "score per document, gold regenerable from the seed"
        ),
    }
    out_path = out / f"novel_schema_rerun_{DATE}.json"
    _write_atomic(out_path, report)
    print(json.dumps({k: v["reading"] for k, v in report.items() if isinstance(v, dict) and "reading" in v}, indent=2))
    print(json.dumps(preregistered, indent=2))
    print(f"wrote {out_path.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="run the arms (resumable, skip-if-exists)")
    parser.add_argument("--read", action="store_true", help="compute the readings; withholds until all six arms exist")
    parser.add_argument("--seed", type=int, action="append", help="restrict to a seed (repeatable)")
    parser.add_argument("--arm", action="append", choices=ARMS, help="restrict to an arm (repeatable)")
    parser.add_argument("--work", default=str(DEFAULT_WORK), help="checkpoint dir (adapters, journals)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="artifact dir")
    parser.add_argument("--banked-dir", default=str(BANKED_DIR), help="where the 2026-08-12 arms and summary live")
    parser.add_argument("--threads", type=int, default=THREADS)
    args = parser.parse_args(argv)
    if not (args.run or args.read):
        parser.error("one of --run / --read")
    code = 0
    if args.run:
        code = run(
            tuple(args.seed) if args.seed else SEEDS,
            tuple(args.arm) if args.arm else ARMS,
            Path(args.work),
            Path(args.out),
            args.threads,
        )
    if args.read:
        code = read(Path(args.out), Path(args.banked_dir))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
