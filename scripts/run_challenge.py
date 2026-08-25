#!/usr/bin/env python3
"""Founder-side runner for a blind-holdout challenge.

Consumes the two files a challenger sends (train.jsonl with gold,
holdout.jsonl without), adapts a fresh LoRA on the training pairs in the
document-only serving configuration (the Addendum F recipe), predicts
every holdout document with the gate-passing voted decode, and writes
the protocol deliverables:

    predictions.jsonl   one line per holdout id: {"id", "prediction"}
                        (prediction is the parsed JSON object, or null
                        when the model's output failed to parse — the
                        protocol scores those 0)
    adapter.pt          the trained LoRA tensors
    manifest.json       adapter sha256, repo commit, exact command,
                        seed/config, per-doc raw completion texts

The founder never sees gold: this script reads only text fields from
holdout.jsonl and never receives a gold file at all.

REAL CHALLENGES run this from a fresh clone of the PUBLIC repo at the
commit pinned in the challenge terms — the manifest's repo_commit must
be a commit the challenger can fetch, or the regenerability deliverable
fails (a dress rehearsal caught exactly this).

    python3 scripts/run_challenge.py --train train.jsonl \
        --holdout holdout.jsonl --out-dir run/ --seed 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def read_jsonl(path: str | pathlib.Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_task(train_rows: list[dict], holdout_rows: list[dict]):
    """Challenge files -> a TextTask (gold serialized canonically)."""
    from arcttt.text_task import TextPair, TextTask
    from arcttt.text_ttt import json_canonical

    train = tuple(
        TextPair(input_text=r["text"], output_text=json_canonical(r["gold"]))
        for r in train_rows
    )
    test = tuple(TextPair(input_text=r["text"], output_text=None) for r in holdout_rows)
    task = TextTask(task_id="challenge", train=train, test=test)
    task.validate()
    return task


def write_outputs(out_dir: pathlib.Path, holdout_rows: list[dict],
                  raw_texts: list[str | None]) -> list[dict]:
    predictions = []
    for row, text in zip(holdout_rows, raw_texts):
        parsed = None
        if text:
            try:
                candidate = json.loads(text)
                if isinstance(candidate, dict):
                    parsed = candidate
            except json.JSONDecodeError:
                parsed = None
        predictions.append({"id": row["id"], "prediction": parsed})
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="challenger's train.jsonl (id/text/gold)")
    parser.add_argument("--holdout", required=True, help="challenger's holdout.jsonl (id/text)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5,
                        help="voted-decode pool size (1 = greedy only)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-seq", type=int, default=8192)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None,
                        choices=("float32", "bfloat16"),
                        help="override the dtype. Default keeps the banked "
                             "behaviour exactly: bfloat16 on CUDA, float32 "
                             "on CPU. Needed for checkpoints too large to "
                             "train in float32 on this box.")
    parser.add_argument("--grad-checkpointing", action="store_true",
                        help="force gradient checkpointing on. Default is "
                             "CUDA-only, which is what produced every banked "
                             "arm; this flag is additive and changes nothing "
                             "unless passed.")
    parser.add_argument("--kshot", action="store_true",
                        help="baseline arm: NO adaptation; the training pairs "
                             "ride in the prompt as demonstrations instead "
                             "(paired-delta comparability for a challenge)")
    parser.add_argument("--allow-unpinned", action="store_true",
                        help="dev/rehearsal only: proceed without a complete "
                             "base-model pin. On a REAL challenge the pin is "
                             "a required deliverable (TERMS T4) and this "
                             "script refuses to emit an unpinned manifest.")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from arcttt.model import TTTConfig
    from arcttt.text_ttt import (TextPredictor, predict_text_voted,
                                 text_docmode_training_examples)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.train)
    holdout_rows = read_jsonl(args.holdout)
    task = build_task(train_rows, holdout_rows)
    print(f"[challenge] {len(train_rows)} training pairs, "
          f"{len(holdout_rows)} holdout documents", flush=True)

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = (getattr(torch, args.dtype) if args.dtype
             else (torch.bfloat16 if device.type == "cuda" else torch.float32))
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    config = TTTConfig(
        lora_rank=args.rank, lora_alpha=args.alpha, epochs=args.epochs,
        max_new_tokens=args.max_new_tokens, max_sequence_tokens=args.max_seq,
        gradient_checkpointing=args.grad_checkpointing or device.type == "cuda",
        chunked_loss_tokens=512,  # the banked-arm loss path (see kernels)
    )
    predictor = TextPredictor(model, tokenizer, config, device)

    if args.kshot:
        adapt_seconds = 0.0
        print("[challenge] k-shot baseline arm: no adaptation; "
              "training pairs ride in the prompt", flush=True)
    else:
        started = time.monotonic()
        predictor.adapt_on_examples(text_docmode_training_examples(task))
        adapt_seconds = round(time.monotonic() - started, 1)
        print(f"[challenge] adapted (doc-only recipe) in {adapt_seconds}s", flush=True)

    adapter = {name: p.detach().cpu() for name, p in model.named_parameters()
               if "lora_" in name}
    torch.save(adapter, out_dir / "adapter.pt")

    raw_texts: list[str | None] = []
    for i in range(len(holdout_rows)):
        t0 = time.monotonic()
        text = predict_text_voted(predictor, task, i, samples=args.samples,
                                  include_demos=args.kshot)
        raw_texts.append(text)
        print(f"[challenge] doc {i + 1}/{len(holdout_rows)} "
              f"({round(time.monotonic() - t0, 1)}s)", flush=True)
    write_outputs(out_dir, holdout_rows, raw_texts)

    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    base_pin = {"name": args.model, "revision": None, "checkpoint_sha256": {}}
    try:
        base_pin["revision"] = getattr(model.config, "_commit_hash", None)
        from huggingface_hub import snapshot_download
        snap = pathlib.Path(snapshot_download(args.model, local_files_only=True))
        for f in sorted(snap.glob("*.safetensors")):
            base_pin["checkpoint_sha256"][f.name] = hashlib.sha256(
                f.read_bytes()).hexdigest()
    except Exception as e:
        base_pin["pin_error"] = f"{type(e).__name__}: {e}"
    if not (base_pin["revision"] and base_pin["checkpoint_sha256"]):
        msg = ("base-model pin incomplete (TERMS T4 requires HF revision + "
               f"checkpoint sha256): {base_pin.get('pin_error', 'fields missing')}")
        if args.allow_unpinned:
            print(f"[challenge] WARNING: {msg} — proceeding because "
                  "--allow-unpinned (NOT valid for a real challenge)", flush=True)
        else:
            sys.exit(f"[challenge] REFUSING to emit deliverables: {msg} — "
                     "fix the pin or pass --allow-unpinned for a rehearsal.")

    manifest = {
        "adapter_sha256": hashlib.sha256((out_dir / "adapter.pt").read_bytes()).hexdigest(),
        "repo_commit": commit,
        "base_model_pin": base_pin,
        "command": " ".join(sys.argv),
        "model": args.model,
        "seed": args.seed,
        "config": {"rank": args.rank, "alpha": args.alpha, "epochs": args.epochs,
                   "samples": args.samples, "max_new_tokens": args.max_new_tokens,
                   "max_seq": args.max_seq, "device": str(device)},
        "arm": "kshot" if args.kshot else "docadapted",
        "adapt_seconds": adapt_seconds,
        "raw_outputs": [{"id": r["id"], "text": t}
                        for r, t in zip(holdout_rows, raw_texts)],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    parsed_n = sum(1 for line in read_jsonl(out_dir / "predictions.jsonl")
                   if line["prediction"] is not None)
    print(f"[challenge] deliverables in {out_dir}/ — "
          f"{parsed_n}/{len(holdout_rows)} predictions parsed as JSON objects; "
          f"adapter sha256 {manifest['adapter_sha256'][:16]}…", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
