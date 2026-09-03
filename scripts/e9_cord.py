#!/usr/bin/env python3
"""Ladder II rung E9: a larger adaptation set with RETRIEVED demonstrations.

Preregistration: docs/research/ADAPTATION_ENGINEERING_LADDER_II.md (rung
E9, and the dated protocol note of 2026-09-03 that fixes demonstration
ORDER to the split order -- BM25 SELECTS the 20 nearest of the 40
training receipts, it does not order them -- so E8's ordering confound
cannot recur).

    PYTHONPATH=src python3 scripts/e9_cord.py --stage split
    PYTHONPATH=src python3 scripts/e9_cord.py --stage adapt
    PYTHONPATH=src python3 scripts/e9_cord.py --stage arm --arm prompted
    PYTHONPATH=src python3 scripts/e9_cord.py --stage arm --arm adapted
    PYTHONPATH=src python3 scripts/e9_cord.py --stage arm --arm prompted_greedy

Split: seed-2 shuffle of the 100 receipts, 40 train / 60 eval, SHA-banked
under experiments/ladder_e9_cord_split/ (committed, recycle-proof).
Adapter: trained on the 40 with E6's recipe, durable sentinel.
Arms: prompted + E7 decoder (the ADAPT bar), adapted + E7 decoder (the
stack), prompted + greedy (the SYSTEM bar on this eval set). Per-document
checkpoints, raw text banked, decoder accounting banked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import re
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

SPLIT_DIR = REPO / "experiments" / "ladder_e9_cord_split"
WORK = REPO / "work" / "e9"
DATE = "2026-09-03"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
DTYPE = "bfloat16"
N_TRAIN, N_EVAL, K = 40, 60, 20
MAX_NEW_TOKENS, MAX_SEQ, TOP_K, SEED = 512, 8192, 16, 2


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25(doc: str, query: str, corpus_tokens: list[list[str]], k1=1.5, b=0.75) -> float:
    import math
    n = len(corpus_tokens)
    avgdl = sum(len(d) for d in corpus_tokens) / n
    dtoks = _tokens(doc); dl = len(dtoks)
    tf: dict[str, int] = {}
    for t in dtoks:
        tf[t] = tf.get(t, 0) + 1
    score = 0.0
    for q in set(_tokens(query)):
        if q not in tf:
            continue
        df = sum(1 for d in corpus_tokens if q in d)
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        f = tf[q]
        score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
    return score


def do_split() -> int:
    from arcttt.text_task import from_cord_gt
    src = REPO / "demo" / "cord_validation.jsonl"
    rows = _read_jsonl(src)
    assert len(rows) == 100
    ids = [f"cord-{i:03d}" for i in range(100)]
    order = list(range(100)); random.Random(SEED).shuffle(order)
    tr, ev = order[:N_TRAIN], order[N_TRAIN:N_TRAIN + N_EVAL]
    task = from_cord_gt([rows[i] for i in tr], [rows[i] for i in ev], task_id="e9-cord")
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for j, i in enumerate(tr):
            f.write(json.dumps({"id": ids[i], "text": task.train[j].input_text,
                                "gold": json.loads(task.train[j].output_text)}, ensure_ascii=False) + "\n")
    with open(SPLIT_DIR / "holdout.jsonl", "w", encoding="utf-8") as f:
        for j, i in enumerate(ev):
            f.write(json.dumps({"id": ids[i], "text": task.test[j].input_text}, ensure_ascii=False) + "\n")
    with open(SPLIT_DIR / "gold.jsonl", "w", encoding="utf-8") as f:
        for j, i in enumerate(ev):
            f.write(json.dumps({"id": ids[i], "gold": json.loads(task.test[j].output_text)}, ensure_ascii=False) + "\n")
    manifest = {"what": "E9 CORD split, deterministic, made before any E9 arm ran.",
                "source_sha256": _sha(src), "rule": f"random.Random({SEED}).shuffle; first {N_TRAIN} train, next {N_EVAL} eval",
                "date": DATE, "n_train": N_TRAIN, "n_eval": N_EVAL,
                "files": {n: _sha(SPLIT_DIR / n) for n in ("train.jsonl", "holdout.jsonl", "gold.jsonl")}}
    (SPLIT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["files"], indent=2)); print("split banked")
    return 0


def _check_split() -> dict:
    manifest = json.loads((SPLIT_DIR / "manifest.json").read_text(encoding="utf-8"))
    for name, want in manifest["files"].items():
        if _sha(SPLIT_DIR / name) != want:
            raise SystemExit(f"{name} drifted from the E9 split manifest")
    return manifest


def _load_model(with_adapter: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4); torch.manual_seed(1)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=getattr(torch, DTYPE))
    if with_adapter:
        from arcttt.lora import inject_lora
        inject_lora(model, rank=16, alpha=32)
        state = torch.load(WORK / "adapter.pt", map_location="cpu")
        load = model.load_state_dict(state, strict=False)
        if load.unexpected_keys:
            raise SystemExit(f"adapter keys not in model: {load.unexpected_keys[:3]}")
        print(f"[e9] loaded adapter {_sha(WORK / 'adapter.pt')[:12]}", flush=True)
    model.eval()
    return model, tok


def do_adapt() -> int:
    import torch
    from arcttt.model import TTTConfig
    from arcttt.text_ttt import TextPredictor, text_docmode_training_examples
    from run_challenge import build_task
    _check_split()
    WORK.mkdir(parents=True, exist_ok=True)
    if (WORK / "adapter.complete").exists():
        print("adapter already banked"); return 0
    train = _read_jsonl(SPLIT_DIR / "train.jsonl")
    # build_task requires a test pair; adaptation reads only task.train.
    task = build_task(train, _read_jsonl(SPLIT_DIR / "holdout.jsonl")[:1])
    model, tok = _load_model(with_adapter=False)
    config = TTTConfig(lora_rank=16, lora_alpha=32, epochs=1, max_new_tokens=MAX_NEW_TOKENS,
                       max_sequence_tokens=MAX_SEQ, gradient_checkpointing=True, chunked_loss_tokens=512)
    predictor = TextPredictor(model, tok, config, torch.device("cpu"))
    t0 = time.monotonic()
    predictor.adapt_on_examples(text_docmode_training_examples(task))
    secs = round(time.monotonic() - t0, 1)
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n}
    torch.save(state, WORK / "adapter.pt"); (WORK / "adapter.complete").touch()
    (WORK / "adapt_seconds.json").write_text(json.dumps({"adapt_seconds": secs, "n_train": N_TRAIN}))
    print(f"[e9] adapted on {N_TRAIN} in {secs}s; adapter saved", flush=True)
    return 0


def do_arm(arm: str) -> int:
    import torch
    from arcttt.constrained_json import constrained_greedy_generate
    from arcttt.model import TTTConfig
    from arcttt.text_ttt import TextPredictor, text_task_to_messages
    from run_challenge import build_task
    manifest = _check_split()
    train = _read_jsonl(SPLIT_DIR / "train.jsonl")
    holdout = _read_jsonl(SPLIT_DIR / "holdout.jsonl")
    out_path = REPO / "experiments" / f"ladder_e9_cord_{arm}_{DATE}.json"
    if out_path.exists():
        print(f"already banked: {out_path.name}"); return 0
    WORK.mkdir(parents=True, exist_ok=True)
    decoder = "greedy" if arm == "prompted_greedy" else f"constrained_topk={TOP_K}"
    config_key = (f"{MODEL}|{arm}|{DTYPE}|k={K}of{N_TRAIN}|select=bm25|order=split|mnt={MAX_NEW_TOKENS}"
                  f"|seq={MAX_SEQ}|{decoder}|split={manifest['files']['holdout.jsonl'][:12]}")
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
                raise SystemExit(f"checkpoint {ckpt_path} belongs to {row.get('config')!r}; refusing")
            done[row["id"]] = row
    resumed = len(done)
    if resumed:
        print(f"[e9:{arm}] resuming {resumed}/{len(holdout)}", flush=True)

    model, tok = _load_model(with_adapter=(arm == "adapted"))
    config = TTTConfig(max_new_tokens=MAX_NEW_TOKENS, max_sequence_tokens=MAX_SEQ)
    predictor = TextPredictor(model, tok, config, torch.device("cpu"))
    corpus_tokens = [_tokens(r["text"]) for r in train]
    train_pos = {r["id"]: i for i, r in enumerate(train)}

    predictions, seconds, accounting, selected = {}, {}, {}, {}
    with open(ckpt_path, "a", encoding="utf-8") as ckpt:
        for i, row in enumerate(holdout):
            if row["id"] in done:
                d = done[row["id"]]
                predictions[row["id"]] = d["raw"]; seconds[row["id"]] = d["seconds"]
                accounting[row["id"]] = d["accounting"]; selected[row["id"]] = d["selected"]
                continue
            # SELECT the K nearest of the N_TRAIN by BM25, then ORDER by split position.
            scored = sorted(train, key=lambda r: -_bm25(r["text"], row["text"], corpus_tokens))[:K]
            demos = sorted(scored, key=lambda r: train_pos[r["id"]])
            task_i = build_task(demos, [row])
            ids = predictor._prompt_ids(text_task_to_messages(task_i, 0, include_demos=True))
            if ids is None:
                raise SystemExit(f"prompt {row['id']} exceeded {MAX_SEQ}")
            began = time.monotonic()
            if arm == "prompted_greedy":
                with torch.no_grad():
                    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                         max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                         pad_token_id=tok.pad_token_id)
                text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
                acct = {"decoder": "greedy"}
            else:
                res = constrained_greedy_generate(model, tok, ids, max_new_tokens=MAX_NEW_TOKENS, top_k=TOP_K)
                text = res.text
                acct = {"steps": res.steps, "constrained_steps": res.constrained_steps,
                        "fallbacks": res.fallbacks, "stopped_on": res.stopped_on}
            took = round(time.monotonic() - began, 1)
            predictions[row["id"]] = text; seconds[row["id"]] = took
            accounting[row["id"]] = acct; selected[row["id"]] = [r["id"] for r in demos]
            ckpt.write(json.dumps({"config": config_key, "id": row["id"], "raw": text, "seconds": took,
                                   "accounting": acct, "selected": selected[row["id"]]}, ensure_ascii=False) + "\n")
            ckpt.flush()
            print(f"[e9:{arm}] {row['id']} {i + 1}/{len(holdout)} {took}s {acct}", flush=True)

    record = {
        "what": f"Ladder II rung E9, {arm} arm: raw CORD outputs on the E9 split. Readings in scripts/ladder_reader.py.",
        "prereg": "docs/research/ADAPTATION_ENGINEERING_LADDER_II.md rung E9 + protocol note 2026-09-03",
        "date": DATE, "model": MODEL, "arm": arm, "dtype": DTYPE,
        "demonstrations": f"BM25 SELECTS {K} of {N_TRAIN} per receipt; ORDER is the split order (E8 confound excluded)",
        "decoder": decoder, "n_eval": len(holdout), "seed": SEED,
        "split_manifest_sha256": manifest["files"],
        "adapter_sha256": (_sha(WORK / "adapter.pt") if arm == "adapted" else None),
        "adapt_seconds": (json.loads((WORK / "adapt_seconds.json").read_text()) if arm == "adapted" and (WORK / "adapt_seconds.json").exists() else None),
        "resumed_from_checkpoint": resumed,
        "per_document_seconds": seconds, "decode_accounting": accounting,
        "selected_demonstrations": selected, "predictions": predictions,
    }
    tmp = out_path.with_suffix(".tmp"); tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out_path); ckpt_path.unlink(missing_ok=True)
    print(f"banked: {out_path.name} ({len(predictions)} raw outputs)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("split", "adapt", "arm"))
    parser.add_argument("--arm", choices=("prompted", "adapted", "prompted_greedy"))
    args = parser.parse_args()
    if args.stage == "split":
        return do_split()
    if args.stage == "adapt":
        return do_adapt()
    if not args.arm:
        raise SystemExit("--stage arm requires --arm")
    return do_arm(args.arm)


if __name__ == "__main__":
    raise SystemExit(main())
