#!/usr/bin/env python3
"""Ladder II rung E7: the E6 arms re-decoded with the JSON-constrained
greedy decoder — both arms, same split, same adapter, same prompts.

Preregistration: docs/research/ADAPTATION_ENGINEERING_LADDER_II.md
(frozen 2026-09-02 before this ran). The decoder is schema-blind and
identical for both arms; the ADAPT reading compares the two E7 arms,
the SYSTEM reading compares E7-adapted against E6's frozen greedy
prompted bar. Scoring lives in scripts/ladder_reader.py.

    PYTHONPATH=src python3 scripts/e7_constrained.py --arm prompted
    PYTHONPATH=src python3 scripts/e7_constrained.py --arm adapted

Per-document checkpoints, split-hash-keyed; refuses a drifted split or
a foreign checkpoint; banks raw text plus the decoder's per-document
accounting (fallbacks, constrained steps, stop reason).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

SPLIT_DIR = REPO / "experiments" / "ladder_e6_cord_split"
WORK = REPO / "work" / "e7"
ADAPTER = REPO / "work" / "e6" / "adapter.pt"
ADAPTER_SENTINEL = REPO / "work" / "e6" / "adapter.complete"
DATE = "2026-09-02"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
DTYPE = "bfloat16"
K = 20
MAX_NEW_TOKENS = 512
MAX_SEQ = 8192
TOP_K = 16
SEED = 1


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("prompted", "adapted"))
    args = parser.parse_args()
    arm = args.arm

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from arcttt.constrained_json import constrained_greedy_generate
    from arcttt.lora import inject_lora
    from arcttt.model import TTTConfig
    from arcttt.text_ttt import TextPredictor, text_task_to_messages
    from run_challenge import build_task

    manifest = json.loads((SPLIT_DIR / "manifest.json").read_text(encoding="utf-8"))
    for name, want in manifest["files"].items():
        if _sha(SPLIT_DIR / name) != want:
            raise SystemExit(f"{name} drifted from the E6 split manifest")
    split_sha = manifest["files"]["holdout.jsonl"][:12]
    train = _read_jsonl(SPLIT_DIR / "train.jsonl")
    holdout = _read_jsonl(SPLIT_DIR / "holdout.jsonl")
    task = build_task(train, holdout)

    out_path = REPO / "experiments" / f"ladder_e7_cord_{arm}_{DATE}.json"
    if out_path.exists():
        print(f"already banked: {out_path.name}")
        return 0
    WORK.mkdir(parents=True, exist_ok=True)
    config_key = (f"{MODEL}|{arm}|{DTYPE}|k={K}|mnt={MAX_NEW_TOKENS}|seq={MAX_SEQ}"
                  f"|constrained_topk={TOP_K}|split={split_sha}")
    ckpt_path = WORK / f"{arm}.ckpt.jsonl"
    done: dict[str, dict] = {}
    if ckpt_path.exists():
        for line in ckpt_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if row.get("config") != config_key:
                raise SystemExit(f"checkpoint {ckpt_path} belongs to "
                                 f"{row.get('config')!r}; refusing")
            done[row["id"]] = row
    resumed = len(done)
    if resumed:
        print(f"[e7:{arm}] resuming {resumed}/{len(holdout)}", flush=True)

    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=getattr(torch, DTYPE))
    adapter_sha = None
    if arm == "adapted":
        if not (ADAPTER.exists() and ADAPTER_SENTINEL.exists()):
            raise SystemExit("E6's durable adapter is missing; E7 reuses it by "
                             "design and does not retrain")
        inject_lora(model, rank=16, alpha=32)
        state = torch.load(ADAPTER, map_location="cpu")
        load = model.load_state_dict(state, strict=False)
        if load.unexpected_keys:
            raise SystemExit(f"adapter keys not in model: {load.unexpected_keys[:3]}")
        adapter_sha = _sha(ADAPTER)
        print(f"[e7:adapted] loaded E6 adapter {adapter_sha[:12]}", flush=True)
    model.eval()
    config = TTTConfig(lora_rank=16, lora_alpha=32, epochs=0,
                       max_new_tokens=MAX_NEW_TOKENS, max_sequence_tokens=MAX_SEQ,
                       gradient_checkpointing=False, chunked_loss_tokens=512)
    predictor = TextPredictor(model, tokenizer, config, torch.device("cpu"))

    predictions: dict[str, str] = {}
    decode: dict[str, dict] = {}
    with open(ckpt_path, "a", encoding="utf-8") as ckpt:
        for i, row in enumerate(holdout):
            if row["id"] in done:
                predictions[row["id"]] = done[row["id"]]["raw"]
                decode[row["id"]] = done[row["id"]]["decode"]
                continue
            ids = predictor._prompt_ids(text_task_to_messages(task, i, include_demos=True))
            if ids is None:
                raise SystemExit(f"prompt {row['id']} exceeded the {MAX_SEQ} budget")
            began = time.monotonic()
            res = constrained_greedy_generate(model, tokenizer, ids,
                                              max_new_tokens=MAX_NEW_TOKENS,
                                              top_k=TOP_K)
            took = round(time.monotonic() - began, 1)
            acct = {"seconds": took, "fallbacks": res.fallbacks,
                    "constrained_steps": res.constrained_steps,
                    "steps": res.steps, "stopped_on": res.stopped_on}
            predictions[row["id"]] = res.text
            decode[row["id"]] = acct
            ckpt.write(json.dumps({"config": config_key, "id": row["id"],
                                   "raw": res.text, "decode": acct},
                                  ensure_ascii=False) + "\n")
            ckpt.flush()
            print(f"[e7:{arm}] {row['id']} {i + 1}/{len(holdout)} {took}s "
                  f"steps={res.steps} constrained={res.constrained_steps} "
                  f"fallbacks={res.fallbacks} stop={res.stopped_on}", flush=True)

    record = {
        "what": f"Ladder II rung E7, {arm} arm: E6's prompts and adapter, "
                "re-decoded with the JSON-constrained greedy decoder. Scoring "
                "and both frozen readings live in scripts/ladder_reader.py.",
        "prereg": "docs/research/ADAPTATION_ENGINEERING_LADDER_II.md rung E7",
        "date": DATE, "model": MODEL, "dtype": DTYPE, "k": K,
        "arm": "adapted_plus_kshot_constrained" if arm == "adapted" else "kshot_constrained",
        "decoder": {"kind": "constrained greedy", "top_k": TOP_K,
                    "validator": "arcttt.constrained_json.is_json_prefix (schema-blind)",
                    "fallback": "top-1 token when no candidate keeps a valid prefix"},
        "adapter_sha256": adapter_sha,
        "split_manifest_sha256": manifest["files"],
        "n_eval": len(holdout), "seed": SEED, "max_new_tokens": MAX_NEW_TOKENS,
        "resumed_from_checkpoint": resumed,
        "decode_accounting": decode,
        "predictions": predictions,
    }
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out_path)
    ckpt_path.unlink(missing_ok=True)
    print(f"banked: {out_path.name} ({len(predictions)} raw outputs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
