#!/usr/bin/env python3
"""Addendum P: the cost ratio when both arms get the same batching.

Addendum O priced a Qwen2.5-3B at batch 1 against our adapted 0.5B at
batch 16. Batching is worth about 3.7x on our own arm, so that ratio
flatters us by roughly that factor. A 196-token schema prompt batches at
least as well as our document-only prompt does, so the 3B gets exactly the
amortisation we give ourselves and the ratio is recomputed.

Every arm runs **in one process, on one box, in one session**. No cost
figure here is compared against one measured somewhere else, because the
instance quote is a price and the seconds are a measurement, and only the
seconds are ours.

Three refusals are built in rather than left to judgement:

* **Padding.** Batching changes throughput, never content: same weights,
  same greedy decode, same prompts. Any per-document difference between a
  batched arm and its own batch-1 reference is a masking defect in the
  batched path, and this refuses to report a cost ratio over arms whose
  predictions moved.
* **Contention.** Per-batch wall clock is banked and the median governs.
  If median and mean disagree by more than 10% the box was shared and the
  arm is marked contaminated rather than published -- the defect that put
  a 20% inflation into Addendum O's first cost figure.
* **Symmetric repair.** Fence-stripping is applied to every arm or to
  none. It is applied, and it is stated in the artifact, because a repair
  granted to one arm and withheld from another is how a 0.8958 was once
  published as a 0.0000.

    PYTHONPATH=src python3 scripts/matched_batch_cost.py \\
        --adapter /tmp/arunL/adapter.pt --batch 16
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import statistics
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

RAW = REPO / "experiments" / "blind_rehearsal_2026-08-20_raw"
OURS = "Qwen/Qwen2.5-0.5B-Instruct"
RIVAL = "Qwen/Qwen2.5-3B-Instruct"

INSTANCE_USD_PER_HOUR = 0.290
RATE_DATE = "2026-08-19"

# Frozen in VERDICT.md Addendum P before this ran.
QUALITY_TOLERANCE = 0.01
CONTENTION_TOLERANCE = 0.10


def _rows(name: str) -> list[dict]:
    return [json.loads(line) for line
            in (RAW / name).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", default=str(
        REPO / "experiments" / "waybill_matched_batch_2026-08-25.json"))
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from arcttt.lora import inject_lora
    from arcttt.model import TTTConfig
    from arcttt.scoring import field_micro_f1, json_canonical, parse_json_object
    from arcttt.text_task import TextTaskFormatError
    from arcttt.text_ttt import TextPredictor, text_task_to_messages

    from fence_rescore import strip_fence  # noqa: E402
    from run_challenge import build_task  # noqa: E402

    torch.set_num_threads(4)
    train, holdout = _rows("train.jsonl"), _rows("holdout.jsonl")
    gold = {r["id"]: r["gold"] for r in _rows("gold_holdout.jsonl")}
    task = build_task(train, holdout)

    # ---- prompt construction, per arm ---------------------------------
    def our_prompts(tokenizer, predictor):
        """Document-only: the schema lives in the adapter weights."""
        ids = []
        for i in range(len(holdout)):
            row = predictor._prompt_ids(
                text_task_to_messages(task, i, include_demos=False))
            if row is None:
                raise SystemExit(f"prompt {i} exceeded the sequence budget")
            ids.append(row[0])
        return ids

    def rival_prompts(tokenizer):
        """Schema line only -- wording asserted identical to the arm that
        took the hosted tier to 0.8930 in Addendum J."""
        first = json.loads((RAW / "gold_holdout.jsonl")
                           .read_text(encoding="utf-8").splitlines()[0])
        instruction = ("Extract these fields from the document and return "
                       "them as a single JSON object with exactly these "
                       "keys, no others:\n"
                       + "\n".join(f"- {f}" for f in sorted(first["gold"])))
        runner = (REPO / "scripts" / "run_market_baseline_waybills.py"
                  ).read_text(encoding="utf-8")
        if "Extract these fields from the document and return " not in runner:
            raise SystemExit("the schema instruction has drifted; this arm "
                             "would no longer be Addendum J's experiment")
        return [tokenizer(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction + "\n\n" + r["text"]}],
                tokenize=False, add_generation_prompt=True),
            return_tensors="pt").input_ids[0] for r in holdout]

    def generate(model, tokenizer, prompts, batch, pad_id, label):
        texts, batch_seconds = [], []
        for start in range(0, len(prompts), batch):
            chunk = prompts[start:start + batch]
            width = max(int(t.shape[0]) for t in chunk)
            input_ids = torch.full((len(chunk), width), pad_id,
                                   dtype=torch.long)
            attention = torch.zeros((len(chunk), width), dtype=torch.long)
            for row, ids in enumerate(chunk):
                input_ids[row, width - ids.shape[0]:] = ids
                attention[row, width - ids.shape[0]:] = 1
            began = time.monotonic()
            with torch.no_grad():
                out = model.generate(input_ids=input_ids,
                                     attention_mask=attention,
                                     max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=pad_id)
            elapsed = time.monotonic() - began
            batch_seconds.append(elapsed)
            texts.extend(
                tokenizer.decode(out[r][width:],
                                 skip_special_tokens=True).strip()
                for r in range(len(chunk)))
            print(f"  [{label}] {min(start + batch, len(prompts))}/"
                  f"{len(prompts)}  {elapsed:7.1f}s", flush=True)
        return texts, batch_seconds

    def score(texts):
        """Fence-stripped, for EVERY arm. Symmetric by construction."""
        per_doc, predictions, invalid = {}, {}, 0
        for row, text in zip(holdout, texts):
            cleaned, _ = strip_fence(text)
            try:
                obj = parse_json_object(cleaned)
            except TextTaskFormatError:
                obj, invalid = None, invalid + 1
            per_doc[row["id"]] = round(
                0.0 if obj is None else field_micro_f1(obj, gold[row["id"]]), 4)
            predictions[row["id"]] = obj
        return per_doc, predictions, invalid

    def summarise(label, batch, texts, batch_seconds):
        per_doc, predictions, invalid = score(texts)
        mean_s = statistics.mean(batch_seconds) / batch
        median_s = statistics.median(batch_seconds) / batch
        contaminated = abs(mean_s - median_s) > CONTENTION_TOLERANCE * median_s
        return {
            "arm": label,
            "batch_size": batch,
            "mean_micro_f1": round(statistics.mean(per_doc.values()), 4),
            "invalid_json_after_fence_strip": invalid,
            "seconds_per_document_mean": round(mean_s, 3),
            "seconds_per_document_median": round(median_s, 3),
            "cost_per_1k_documents_usd": round(
                median_s * 1000 / 3600 * INSTANCE_USD_PER_HOUR, 4),
            "cost_basis": "median seconds/document; the median governs "
                          "because a shared box inflates the mean",
            "contended": contaminated,
            "seconds_per_batch": [round(s, 2) for s in batch_seconds],
            "per_document_micro_f1": per_doc,
            "predictions": predictions,
        }

    arms: list[dict] = []

    # ---- our adapted 0.5B ---------------------------------------------
    print(f"loading {OURS} + adapter", flush=True)
    tok_ours = AutoTokenizer.from_pretrained(OURS)
    if tok_ours.pad_token_id is None:
        tok_ours.pad_token = tok_ours.eos_token
    torch.manual_seed(1)
    model = AutoModelForCausalLM.from_pretrained(OURS, dtype=torch.float32)
    inject_lora(model, rank=16, alpha=32)
    state = torch.load(args.adapter, map_location="cpu")
    missing = model.load_state_dict(state, strict=False)
    if not state:
        raise SystemExit("adapter.pt contained no tensors")
    if missing.unexpected_keys:
        raise SystemExit(f"adapter keys not in model: "
                         f"{missing.unexpected_keys[:3]}")
    model.eval()
    config = TTTConfig(lora_rank=16, lora_alpha=32, epochs=1,
                       max_new_tokens=args.max_new_tokens,
                       max_sequence_tokens=8192,
                       gradient_checkpointing=False, chunked_loss_tokens=512)
    predictor = TextPredictor(model, tok_ours, config, torch.device("cpu"))
    prompts = our_prompts(tok_ours, predictor)
    for batch in (1, args.batch):
        texts, secs = generate(model, tok_ours, prompts, batch,
                               tok_ours.pad_token_id, f"ours/batch{batch}")
        arms.append(summarise(f"our_adapted_0.5b/batch{batch}", batch,
                              texts, secs))
    del model, predictor

    # ---- the unadapted 3B ----------------------------------------------
    print(f"loading {RIVAL}", flush=True)
    tok_rival = AutoTokenizer.from_pretrained(RIVAL)
    if tok_rival.pad_token_id is None:
        tok_rival.pad_token = tok_rival.eos_token
    rival = AutoModelForCausalLM.from_pretrained(RIVAL, dtype=torch.float32)
    rival.eval()
    prompts = rival_prompts(tok_rival)
    for batch in (1, args.batch):
        texts, secs = generate(rival, tok_rival, prompts, batch,
                               tok_rival.pad_token_id, f"3B/batch{batch}")
        arms.append(summarise(f"qwen2.5_3b_schema_only/batch{batch}", batch,
                              texts, secs))
    del rival

    # ---- integrity checks, before any reading ---------------------------
    by_name = {a["arm"]: a for a in arms}
    padding_defects = []
    for family in ("our_adapted_0.5b", "qwen2.5_3b_schema_only"):
        reference = by_name[f"{family}/batch1"]["predictions"]
        batched = by_name[f"{family}/batch{args.batch}"]["predictions"]
        differing = [doc for doc, obj in batched.items()
                     if json_canonical(obj) != json_canonical(reference[doc])]
        if differing:
            padding_defects.append({"family": family,
                                    "documents_differing": differing})
    contended = [a["arm"] for a in arms if a["contended"]]

    ours_b = by_name[f"our_adapted_0.5b/batch{args.batch}"]
    rival_b = by_name[f"qwen2.5_3b_schema_only/batch{args.batch}"]
    ratio = (rival_b["cost_per_1k_documents_usd"]
             / ours_b["cost_per_1k_documents_usd"])
    quality_delta = rival_b["mean_micro_f1"] - ours_b["mean_micro_f1"]

    if padding_defects:
        verdict = "NO READING — PADDING DEFECT"
        why = ("Batched predictions differ from their own batch-1 reference. "
               "Batching changes throughput, never content, so this is a "
               "masking bug in the batched path and is reported as a bug. No "
               "cost ratio is cited over arms whose predictions moved.")
    elif contended:
        verdict = "NO READING — CONTENDED BOX"
        why = (f"Per-batch median and mean disagree by more than "
               f"{CONTENTION_TOLERANCE:.0%} on {contended}. The box was "
               "shared; the arm is re-run rather than reported.")
    elif abs(quality_delta) > QUALITY_TOLERANCE:
        verdict = "(d) QUALITY DIVERGES UNDER BATCHING"
        why = (f"|Q| = {abs(quality_delta):.4f} exceeds the frozen "
               f"{QUALITY_TOLERANCE} tolerance, so the arms are not matched "
               "on quality and no cost ratio is cited.")
    elif ratio >= 5.0:
        verdict = "(a) THE COST ADVANTAGE IS REAL AND LARGE"
        why = (f"At matched batch {args.batch} on one box, the unadapted 3B "
               f"costs {ratio:.1f}x what the adapted 0.5B costs, at a "
               f"quality difference of {quality_delta:+.4f} — inside the "
               f"frozen {QUALITY_TOLERANCE} tolerance. Per-tenant adaptation "
               "delivers the same extraction quality as an unadapted model "
               "six times its size for a fraction of the compute.")
    elif ratio >= 2.0:
        verdict = "(b) REAL BUT MODEST"
        why = (f"{ratio:.1f}x at matched batching, quality difference "
               f"{quality_delta:+.4f}. Published at exactly the measured "
               "ratio.")
    else:
        verdict = "(c) THE FLOOR IS GONE"
        why = (f"{ratio:.2f}x does not carry a company against a checkpoint "
               "that needs no adaptation, no per-tenant training and no "
               "demonstrations. Extraction is not the product.")

    record = {
        "addendum": "P",
        "what": "The cost ratio between our adapted 0.5B and an unadapted "
                "Qwen2.5-3B given only a field list, with BOTH arms batched "
                "the same way, measured in one session on one box.",
        "preregistered": "VERDICT.md Addendum P, frozen before this ran.",
        "fence_policy": "Fence-stripping applied to EVERY arm. Symmetric by "
                        "construction: the same scorer path runs on all four "
                        "arms and there is no flag to disable it for one.",
        "batch_size": args.batch,
        "arms": arms,
        "cost_ratio_theirs_over_ours_at_matched_batch": round(ratio, 3),
        "quality_delta_theirs_minus_ours": round(quality_delta, 4),
        "quality_tolerance": QUALITY_TOLERANCE,
        "padding_defects": padding_defects,
        "contended_arms": contended,
        "verdict": verdict,
        "why": why,
        "scope": f"One corpus, thirty agent-authored documents, one seed, "
                 f"CPU, greedy, batch 1 and {args.batch}, "
                 f"${INSTANCE_USD_PER_HOUR}/hour external instance quote "
                 f"({RATE_DATE}). The dollar figures are seconds we measured "
                 "times a price we did not.",
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__,
                        "threads": torch.get_num_threads()},
        "verification_level": "PRIMARY — per-document predictions stored for "
                              "every arm.",
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")

    print()
    for arm in arms:
        print(f"{arm['arm']:38s} {arm['mean_micro_f1']:.4f}  "
              f"${arm['cost_per_1k_documents_usd']:>8.4f}/1k  "
              f"{arm['seconds_per_document_median']:6.2f}s/doc"
              f"{'  CONTENDED' if arm['contended'] else ''}")
    print(f"\nratio at batch {args.batch}: {ratio:.2f}x   "
          f"quality delta: {quality_delta:+.4f}")
    print(f"{verdict}\n{why}")
    print(f"\nbanked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
