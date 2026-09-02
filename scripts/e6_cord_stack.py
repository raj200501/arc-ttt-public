#!/usr/bin/env python3
"""Ladder rung E6: the E5 stack on CORD, where headroom exists.

Preregistration: docs/research/ADAPTATION_ENGINEERING_LADDER.md, rung E6
(added 2026-08-31 before any E6 data). Same stacked configuration as E5
-- adapted 3B PLUS k=20 demonstrations, bfloat16 -- against the same bar
rule -- prompted 3B k=20, same dtype -- on the 100 CORD validation
receipts. Both arms fresh, both bank raw text, fence handling symmetric
(the ladder reader scores both arms from raw with the shipped tool).

Three stages, each resumable, because this box reclaims processes
between agent turns and a 3B CPU arm takes hours:

    PYTHONPATH=src python3 scripts/e6_cord_stack.py --stage split
    PYTHONPATH=src python3 scripts/e6_cord_stack.py --stage arm --arm prompted
    PYTHONPATH=src python3 scripts/e6_cord_stack.py --stage arm --arm adapted

SPLIT is deterministic and banked under experiments/ (committed, so it
survives container recycles): seed-1 shuffle of the 100 receipts in file
order -- the same convention as every prior CORD arm in this tree
(cord_scale_run.py: random.Random(seed).shuffle) -- first 20 are the
training pairs (adaptation set AND the k=20 demonstrations), the
remaining 80 are evaluation. The arms REFUSE to run without the split
manifest, and every checkpoint line carries the split hash, so an arm
cannot silently mix splits.

ARMS write one checkpoint line per document (config-keyed, torn-line
tolerant, deleted on success, resume disclosed in the artifact) -- the
scale_rung_arm.py pattern, adopted after a VM suspend killed an arm at
17/30 and cost the afternoon.

Scoring does NOT happen here: the ladder reader applies the frozen bars
to both arms' raw text symmetrically, so the instrument and the number
cannot disagree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

SPLIT_DIR = REPO / "experiments" / "ladder_e6_cord_split"
WORK = REPO / "work" / "e6"
DATE = "2026-08-31"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
DTYPE = "bfloat16"
K = 20
EVAL_N = 80
MAX_NEW_TOKENS = 512
# Measured with the 3B tokenizer BEFORE either arm ran: the k=20 CORD
# prompts span 3199-3401 tokens, so the waybill arms' 8192 budget holds
# here unchanged -- the same number E5 used, for both arms.
MAX_SEQ = 8192
SEED = 1


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def do_split() -> int:
    from arcttt.text_task import from_cord_gt

    src = REPO / "demo" / "cord_validation.jsonl"
    rows = [json.loads(line) for line in
            src.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 100:
        raise SystemExit(f"expected 100 CORD validation rows, got {len(rows)}")
    ids = [f"cord-{i:03d}" for i in range(len(rows))]
    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)
    train_idx, eval_idx = order[:K], order[K:K + EVAL_N]

    # Render through the SAME adapter every prior CORD arm used, so the
    # text and target format are the tree's, not this script's.
    task = from_cord_gt([rows[i] for i in train_idx],
                        [rows[i] for i in eval_idx], task_id="e6-cord")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for j, i in enumerate(train_idx):
            f.write(json.dumps({"id": ids[i], "text": task.train[j].input_text,
                                "gold": json.loads(task.train[j].output_text)},
                               ensure_ascii=False) + "\n")
    with open(SPLIT_DIR / "holdout.jsonl", "w", encoding="utf-8") as f:
        for j, i in enumerate(eval_idx):
            f.write(json.dumps({"id": ids[i], "text": task.test[j].input_text},
                               ensure_ascii=False) + "\n")
    with open(SPLIT_DIR / "gold.jsonl", "w", encoding="utf-8") as f:
        for j, i in enumerate(eval_idx):
            f.write(json.dumps({"id": ids[i],
                                "gold": json.loads(task.test[j].output_text)},
                               ensure_ascii=False) + "\n")
    manifest = {
        "what": "E6 CORD split, deterministic, made before either arm ran.",
        "source": "demo/cord_validation.jsonl",
        "source_sha256": _sha(src),
        "rule": f"ids cord-000..cord-099 in file order; "
                f"random.Random({SEED}).shuffle(indices); first {K} train "
                f"(adaptation set and the k={K} demonstrations), next "
                f"{EVAL_N} evaluation — the cord_scale_run.py convention",
        "date": DATE,
        "n_train": K, "n_eval": EVAL_N,
        "files": {name: _sha(SPLIT_DIR / name)
                  for name in ("train.jsonl", "holdout.jsonl", "gold.jsonl")},
    }
    (SPLIT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["files"], indent=2))
    print(f"split banked: {SPLIT_DIR}")
    return 0


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_ckpt(path: pathlib.Path, config_key: str) -> dict[str, dict]:
    """Config-keyed resume; foreign config refused; torn last line dropped."""
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break  # torn tail from a killed write; everything before it counts
        if row.get("config") != config_key:
            raise SystemExit(
                f"checkpoint {path} was written by a different configuration "
                f"({row.get('config')!r}); refusing to mix arms. Delete it "
                "if that run is abandoned.")
        done[row["id"]] = row
    return done


def do_arm(arm: str) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from arcttt.model import TTTConfig
    from arcttt.text_ttt import (TextPredictor, text_task_to_messages,
                                 text_docmode_training_examples)
    from run_challenge import build_task  # noqa: E402  (scripts/ on path)

    manifest_path = SPLIT_DIR / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("no split manifest; run --stage split first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, want in manifest["files"].items():
        have = _sha(SPLIT_DIR / name)
        if have != want:
            raise SystemExit(f"{name} sha {have[:12]} != manifest "
                             f"{want[:12]}; the split changed under the arm")
    split_sha = manifest["files"]["holdout.jsonl"][:12]

    train = _read_jsonl(SPLIT_DIR / "train.jsonl")
    holdout = _read_jsonl(SPLIT_DIR / "holdout.jsonl")
    task = build_task(train, holdout)

    WORK.mkdir(parents=True, exist_ok=True)
    out_path = REPO / "experiments" / f"ladder_e6_cord_{arm}_{DATE}.json"
    if out_path.exists():
        print(f"already banked: {out_path.name}")
        return 0

    config_key = (f"{MODEL}|{arm}|{DTYPE}|k={K}|mnt={MAX_NEW_TOKENS}"
                  f"|seq={MAX_SEQ}|split={split_sha}")
    ckpt_path = WORK / f"{arm}.ckpt.jsonl"
    done = _load_ckpt(ckpt_path, config_key)
    resumed_from = len(done)
    if resumed_from:
        print(f"[e6:{arm}] resuming: {resumed_from}/{len(holdout)} "
              "documents already checkpointed", flush=True)

    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=getattr(torch, DTYPE))
    config = TTTConfig(lora_rank=16, lora_alpha=32,
                       epochs=1 if arm == "adapted" else 0,
                       max_new_tokens=MAX_NEW_TOKENS,
                       max_sequence_tokens=MAX_SEQ,
                       gradient_checkpointing=True,  # the Q memory contingency
                       chunked_loss_tokens=512)
    predictor = TextPredictor(model, tokenizer, config, torch.device("cpu"))

    adapt_seconds = 0.0
    if arm == "adapted":
        adapter_path = WORK / "adapter.pt"
        sentinel = WORK / "adapter.complete"
        if sentinel.exists() and adapter_path.exists():
            from arcttt.lora import inject_lora
            inject_lora(model, rank=16, alpha=32)
            state = torch.load(adapter_path, map_location="cpu")
            if not state:
                raise SystemExit("adapter.pt contained no tensors")
            load = model.load_state_dict(state, strict=False)
            if load.unexpected_keys:
                raise SystemExit(f"adapter keys not in model: "
                                 f"{load.unexpected_keys[:3]}")
            print(f"[e6:adapted] loaded durable adapter {adapter_path}",
                  flush=True)
        else:
            started = time.monotonic()
            predictor.adapt_on_examples(text_docmode_training_examples(task))
            adapt_seconds = round(time.monotonic() - started, 1)
            state = {name: p.detach().cpu()
                     for name, p in model.named_parameters() if "lora_" in name}
            torch.save(state, adapter_path)
            sentinel.touch()
            print(f"[e6:adapted] adapted in {adapt_seconds}s; adapter saved",
                  flush=True)
    model.eval()

    predictions: dict[str, str] = {}
    seconds: dict[str, float] = {}
    with open(ckpt_path, "a", encoding="utf-8") as ckpt:
        for i, row in enumerate(holdout):
            if row["id"] in done:
                predictions[row["id"]] = done[row["id"]]["raw"]
                seconds[row["id"]] = done[row["id"]]["seconds"]
                continue
            ids = predictor._prompt_ids(
                text_task_to_messages(task, i, include_demos=True))
            if ids is None:
                raise SystemExit(f"prompt {row['id']} exceeded the "
                                 f"{MAX_SEQ}-token budget")
            began = time.monotonic()
            with torch.no_grad():
                out = model.generate(input_ids=ids,
                                     attention_mask=torch.ones_like(ids),
                                     max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
            took = round(time.monotonic() - began, 1)
            text = tokenizer.decode(out[0][ids.shape[1]:],
                                    skip_special_tokens=True).strip()
            predictions[row["id"]] = text
            seconds[row["id"]] = took
            ckpt.write(json.dumps({"config": config_key, "id": row["id"],
                                   "raw": text, "seconds": took},
                                  ensure_ascii=False) + "\n")
            ckpt.flush()
            print(f"[e6:{arm}] {row['id']} {i + 1}/{len(holdout)} {took}s",
                  flush=True)

    record = {
        "what": f"Ladder rung E6, {arm} arm: raw CORD outputs. Scoring "
                "and the frozen reading live in scripts/ladder_reader.py, "
                "applied to both arms symmetrically.",
        "prereg": "docs/research/ADAPTATION_ENGINEERING_LADDER.md rung E6",
        "date": DATE,
        "model": MODEL,
        "arm": ("adapted_plus_kshot" if arm == "adapted" else "kshot"),
        "dtype": DTYPE,
        "k": K,
        "n_eval": len(holdout),
        "decode": "greedy, samples=1, max_new_tokens=512",
        "config": {"rank": 16, "alpha": 32,
                   "epochs": 1 if arm == "adapted" else 0,
                   "max_seq": MAX_SEQ, "seed": SEED},
        "split_manifest_sha256": {n: manifest["files"][n]
                                  for n in manifest["files"]},
        "adapt_seconds": adapt_seconds,
        "resumed_from_checkpoint": resumed_from,
        "per_document_seconds": seconds,
        "predictions": predictions,
    }
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out_path)
    ckpt_path.unlink(missing_ok=True)
    print(f"banked: {out_path.name} ({len(predictions)} raw outputs)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("split", "arm"))
    parser.add_argument("--arm", choices=("prompted", "adapted"))
    args = parser.parse_args()
    if args.stage == "split":
        return do_split()
    if not args.arm:
        raise SystemExit("--stage arm requires --arm prompted|adapted")
    return do_arm(args.arm)


if __name__ == "__main__":
    raise SystemExit(main())
