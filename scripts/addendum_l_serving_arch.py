#!/usr/bin/env python3
"""Addendum L: is our ~$0.89 a property of the model or of how we serve it?

Addendum J showed the cheapest hosted configuration that outscores us --
a bare field list, no demonstrations -- costs ~$0.36 per 1,000 documents
against our ~$0.89. That comparison is real and it stands. But our $0.89
was measured at **batch size 1, float32, one document at a time**, and
neither of those is forced by anything.

This runs three serving configurations of the SAME adapted weights over
the SAME 30 held-out waybills, and measures **cost and quality on every
one**. The bar was frozen in VERDICT.md before any quality number here
existed, and the reason is written into the design: quantization is
exactly the kind of change that buys throughput by losing accuracy, so
an arm that reports a cost without its quality beside it would be the
defect this repository keeps correcting.

Two honesty constraints are structural rather than editorial:

* **The adapter is a real trained adapter, and it is retained.** The
  previous cost artifact had to inject an UNTRAINED LoRA because the
  rehearsal adapter was not kept, then correct itself for the drift that
  caused. This run loads `adapter.pt` from a fresh `run_challenge.py`
  adaptation on the 20-document train split.
* **Batching is scoped, not smuggled.** It assumes concurrent documents.
  That is what a tenant processing a day's paperwork has and what an
  interactive single-document request does not, and per-document LATENCY
  rises under it even as cost falls. Both are reported.

    PYTHONPATH=src python3 scripts/addendum_l_serving_arch.py \
        --adapter /tmp/arunL/adapter.pt
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

RAW = REPO / "experiments" / "blind_rehearsal_2026-08-20_raw"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# The same external quote every other cost row on this page uses, so the
# pages cannot drift. A price, not a measurement.
INSTANCE_USD_PER_HOUR = 0.290
RATE_DATE = "2026-08-19"

# The frozen bar, transcribed from VERDICT.md's Addendum L row.
COST_BAR = 0.36        # the hosted k=0+schema arm, Addendum J
QUALITY_TOLERANCE = 0.01
HOSTED_SCHEMA_ARM = {"mean_micro_f1": 0.8930, "cost_per_1k_usd": 0.36,
                     "source": "Addendum J, 3 runs, "
                               "waybill_market_kshot_summary_2026-08-22.json"}


def _rows(name: str) -> list[dict]:
    return [json.loads(line) for line
            in (RAW / name).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True,
                        help="adapter.pt from run_challenge.py on the train split")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--amendment", action="store_true",
                        help="run the preregistered amendment arms: batching "
                             "WITHOUT quantization (fp32 at batch 8 and 16)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", default=str(
        REPO / "experiments" / "waybill_serving_arch_2026-08-22.json"))
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from arcttt.lora import inject_lora
    from arcttt.model import TTTConfig
    from arcttt.scoring import field_micro_f1, parse_json_object
    from arcttt.text_task import TextTaskFormatError
    from arcttt.text_ttt import TextPredictor, text_task_to_messages

    # Reuse the challenge runner's OWN task builder rather than
    # reconstructing it here. The whole point of this addendum is that
    # every arm is fed exactly what the banked arm was fed, and the way to
    # guarantee that is to call the same code rather than to mirror it.
    sys.path.insert(0, str(REPO / "scripts"))
    from run_challenge import build_task  # noqa: E402

    torch.set_num_threads(4)
    train, holdout = _rows("train.jsonl"), _rows("holdout.jsonl")
    gold = {r["id"]: r["gold"] for r in _rows("gold_holdout.jsonl")}
    task = build_task(train, holdout)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    def load_model():
        torch.manual_seed(1)
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
        inject_lora(model, rank=16, alpha=32)
        state = torch.load(args.adapter, map_location="cpu")
        missing = model.load_state_dict(state, strict=False)
        loaded = len(state)
        if loaded == 0:
            raise SystemExit("adapter.pt contained no tensors")
        if missing.unexpected_keys:
            raise SystemExit(f"adapter keys not in model: "
                             f"{missing.unexpected_keys[:3]}")
        model.eval()
        return model, loaded

    model, n_adapter_tensors = load_model()
    config = TTTConfig(lora_rank=16, lora_alpha=32, epochs=1,
                       max_new_tokens=args.max_new_tokens,
                       max_sequence_tokens=8192,
                       gradient_checkpointing=False, chunked_loss_tokens=512)
    predictor = TextPredictor(model, tokenizer, config, torch.device("cpu"))

    # Build every prompt through the SAME construction the challenge runner
    # uses, so no arm here is fed anything the banked arm was not.
    prompts = []
    for i in range(len(holdout)):
        ids = predictor._prompt_ids(
            text_task_to_messages(task, i, include_demos=False))
        if ids is None:
            raise SystemExit(f"prompt {i} exceeded the sequence budget")
        prompts.append(ids[0])

    def decode_batch(m, batch_ids: list) -> tuple[list[str], float]:
        """Left-pad a batch, generate greedily, return completions."""
        width = max(int(t.shape[0]) for t in batch_ids)
        input_ids = torch.full((len(batch_ids), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(batch_ids), width), dtype=torch.long)
        for row, ids in enumerate(batch_ids):
            input_ids[row, width - ids.shape[0]:] = ids
            attention[row, width - ids.shape[0]:] = 1
        started = time.monotonic()
        with torch.no_grad():
            out = m.generate(input_ids=input_ids, attention_mask=attention,
                             max_new_tokens=args.max_new_tokens,
                             do_sample=False, pad_token_id=pad_id)
        elapsed = time.monotonic() - started
        texts = [tokenizer.decode(out[r][width:], skip_special_tokens=True).strip()
                 for r in range(len(batch_ids))]
        return texts, elapsed

    def run_arm(m, batch: int, label: str) -> dict:
        texts, seconds, new_tokens = [], 0.0, 0
        for start in range(0, len(prompts), batch):
            chunk = prompts[start:start + batch]
            got, elapsed = decode_batch(m, chunk)
            texts.extend(got)
            seconds += elapsed
            new_tokens += sum(len(tokenizer(t).input_ids) for t in got)
            print(f"  [{label}] {min(start + batch, len(prompts))}/"
                  f"{len(prompts)}  {elapsed:6.2f}s", flush=True)
        scores, invalid, predictions = [], 0, {}
        for row, text in zip(holdout, texts):
            try:
                obj = parse_json_object(text)
            except TextTaskFormatError:
                obj, invalid = None, invalid + 1
            predictions[row["id"]] = obj
            scores.append(0.0 if obj is None
                          else field_micro_f1(obj, gold[row["id"]]))
        per_doc = seconds / len(prompts)
        return {
            "arm": label,
            "batch_size": batch,
            "mean_micro_f1": round(statistics.mean(scores), 4),
            "invalid_json": invalid,
            "wall_clock_seconds_total": round(seconds, 2),
            "seconds_per_document_amortised": round(per_doc, 3),
            "seconds_per_batch_latency": round(seconds / max(
                1, (len(prompts) + batch - 1) // batch), 3),
            "output_tokens_mean": round(new_tokens / len(prompts), 1),
            "cost_per_1k_documents_usd": round(
                per_doc * 1000 / 3600 * INSTANCE_USD_PER_HOUR, 4),
            "per_document_micro_f1": {r["id"]: round(s, 4)
                                      for r, s in zip(holdout, scores)},
            "predictions": predictions,
        }

    arms = [run_arm(model, 1, "fp32/batch1 (reference)")]

    if args.amendment:
        # Batching WITHOUT quantization. The frozen design made int8 the
        # vehicle for batching and never tested batching alone, which was
        # a design error; this is the amendment preregistered in
        # VERDICT.md before it ran.
        for batch in (8, 16):
            arms.append(run_arm(model, batch, f"fp32/batch{batch}"))
    else:
        print("quantizing to int8 (dynamic, nn.Linear)...", flush=True)
        quantized = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8)
        arms.append(run_arm(quantized, 1, "int8/batch1"))
        arms.append(run_arm(quantized, args.batch, f"int8/batch{args.batch}"))

    # Batching changes throughput, never content: same weights, same greedy
    # decode, same prompts. Any per-document difference between a batched
    # arm and the batch-1 reference is a PADDING OR MASKING DEFECT in our
    # batched path, not a quality trade, and must be reported as a bug.
    # Quantized arms are exempt -- they change the weights on purpose.
    from arcttt.scoring import json_canonical
    reference_predictions = arms[0]["predictions"]
    padding_defects = []
    for arm in arms[1:]:
        if not arm["arm"].startswith("fp32"):
            continue
        differing = [doc_id for doc_id, obj in arm["predictions"].items()
                     if json_canonical(obj)
                     != json_canonical(reference_predictions[doc_id])]
        if differing:
            padding_defects.append({"arm": arm["arm"],
                                    "documents_differing": differing})

    reference = arms[0]
    best = min((a for a in arms if a["cost_per_1k_documents_usd"] < COST_BAR),
               key=lambda a: a["cost_per_1k_documents_usd"], default=None)

    if best is None:
        verdict = "(c) NOT COST-COMPETITIVE ON THIS HARDWARE"
        why = (f"No arm reached the frozen ${COST_BAR}/1k bar. The cheapest "
               f"was {min(a['cost_per_1k_documents_usd'] for a in arms):.4f}. "
               "Addendum J's withdrawal stands unchanged, in the words "
               "frozen before this ran.")
    elif abs(best["mean_micro_f1"] - reference["mean_micro_f1"]) <= QUALITY_TOLERANCE:
        verdict = "(a) THE COST-EFFECTIVE ARCHITECTURE IS REAL"
        why = (
            f"**{best['arm']} serves these documents for "
            f"${best['cost_per_1k_documents_usd']:.4f} per 1,000 at "
            f"{best['mean_micro_f1']:.4f} micro-F1 — inside the frozen "
            f"{QUALITY_TOLERANCE} tolerance of the fp32 reference's "
            f"{reference['mean_micro_f1']:.4f}, and "
            f"{HOSTED_SCHEMA_ARM['cost_per_1k_usd'] / best['cost_per_1k_documents_usd']:.1f}x "
            f"cheaper than the ${HOSTED_SCHEMA_ARM['cost_per_1k_usd']} hosted "
            "arm that outscores us.** In the words frozen before this ran: "
            "the cost-effective architecture is real, and our $0.89 was a "
            "property of how we served the model rather than of the model. "
            "**Said in the same breath, because the bar requires it:** this "
            "is a batch-workload claim, and the hosted arm still scores "
            f"{HOSTED_SCHEMA_ARM['mean_micro_f1']} against our "
            f"{best['mean_micro_f1']:.4f}. Cheaper at our own quality is not "
            "cheaper at theirs.")
    else:
        verdict = "(b) A TRADE, NOT A COST WIN"
        why = (
            f"{best['arm']} reaches ${best['cost_per_1k_documents_usd']:.4f} "
            f"per 1,000 but scores {best['mean_micro_f1']:.4f} against the "
            f"fp32 reference's {reference['mean_micro_f1']:.4f} — a drop of "
            f"{reference['mean_micro_f1'] - best['mean_micro_f1']:.4f}, "
            f"outside the frozen {QUALITY_TOLERANCE} tolerance. Per the "
            "frozen reading this publishes as a trade at exactly those two "
            "numbers and NOT as a cost win.")

    record = {
        "addendum": "L",
        "what": "Is our ~$0.89 per 1,000 documents a property of the model "
                "or of how we serve it?",
        "preregistered": "VERDICT.md Addendum L, frozen before any quality "
                         "number in this artifact existed.",
        "adapter": {
            "path": str(args.adapter),
            "tensors_loaded": n_adapter_tensors,
            "provenance": "run_challenge.py --samples 1 --seed 1 on the "
                          "20-document train split -- a REAL trained "
                          "adapter, retained this time. The previous cost "
                          "artifact had to time an untrained LoRA because "
                          "the rehearsal adapter was not kept.",
        },
        "hardware": {"platform": platform.platform(),
                     "torch_threads": 4, "device": "cpu"},
        "instance_rate": {"usd_per_hour": INSTANCE_USD_PER_HOUR,
                          "quoted": RATE_DATE,
                          "source": "external cloud list price, 8-vCPU "
                                    "on-demand; not a measurement"},
        "bar": {"cost_per_1k_usd": COST_BAR,
                "quality_tolerance_vs_fp32_reference": QUALITY_TOLERANCE},
        "comparator_hosted_arm": HOSTED_SCHEMA_ARM,
        "arms": [{k: v for k, v in a.items() if k != "predictions"}
                 for a in arms],
        "verdict": verdict,
        "why": why,
        "batched_path_padding_defects": padding_defects,
        "padding_check": "Batching changes throughput, never content: same "
                         "weights, same greedy decode, same prompts. Every "
                         "fp32 batched arm's predictions are compared "
                         "document by document against the batch-1 "
                         "reference under canonical JSON. A non-empty "
                         "batched_path_padding_defects list is a BUG in our "
                         "batched serving path -- a left-padding or "
                         "attention-mask error -- and not a quality trade, "
                         "and any cost reading here is void until it is "
                         "fixed. Quantized arms are exempt because they "
                         "change the weights deliberately.",
        "batching_scope": "Batching assumes CONCURRENT documents. It is "
                          "legitimate for a tenant processing a day's "
                          "paperwork and illegitimate for interactive "
                          "single-document latency. Per-document latency "
                          "RISES under it even as cost falls: the "
                          "seconds_per_batch_latency field is the number an "
                          "interactive caller would feel, and "
                          "seconds_per_document_amortised is the one the "
                          "cost is computed from. Both are reported so the "
                          "trade cannot be hidden by quoting one.",
        "what_this_does_not_claim": "That we beat the hosted tier on "
                                    "quality. We do not, on this corpus: "
                                    "Addendum I and J are unchanged by "
                                    "anything here, and a cheaper worse "
                                    "answer is only worth buying to someone "
                                    "who cannot buy the better one -- which "
                                    "is the on-prem scoping and the whole "
                                    "remaining claim.",
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")

    print(f"\n{'arm':28s} {'F1':>7s} {'$/1k':>8s} {'s/doc':>7s} {'invalid':>8s}")
    for a in arms:
        print(f"{a['arm']:28s} {a['mean_micro_f1']:7.4f} "
              f"{a['cost_per_1k_documents_usd']:8.4f} "
              f"{a['seconds_per_document_amortised']:7.2f} "
              f"{a['invalid_json']:8d}")
    print(f"{'hosted k=0+schema (J)':28s} "
          f"{HOSTED_SCHEMA_ARM['mean_micro_f1']:7.4f} "
          f"{HOSTED_SCHEMA_ARM['cost_per_1k_usd']:8.4f}")
    if padding_defects:
        print("\n*** BATCHED-PATH DEFECT: batching changed the output on "
              f"{sum(len(d['documents_differing']) for d in padding_defects)} "
              "document(s). This is a padding/masking bug, not a result. "
              "The cost reading below is VOID until it is fixed. ***")
        for defect in padding_defects:
            print(f"    {defect['arm']}: {defect['documents_differing']}")
    print(f"\nVERDICT: {verdict}\n{why}")
    print(f"\nbanked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
