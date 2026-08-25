#!/usr/bin/env python3
"""Our serving cost ON THE WAYBILL CORPUS — the half that was still blank.

`waybill_cost.py` measured the hosted arm's cost on these 30 documents
and deliberately left ours empty, because every throughput number in this
repository was measured on the synthetic corpus at a different prompt
shape and does not transfer. This fills that blank by measurement rather
than by transfer.

What is measured: wall-clock per document to serve the 30 held-out
waybills in the **document-only** configuration — no demonstrations in
the prompt, which is the whole point of putting the schema in the weights
and is the configuration Addendum F gated.

**What this does NOT measure, stated because it would be easy to imply
otherwise:** quality. The banked rehearsal adapter was not retained, so
this injects a LoRA of the same rank and shape and times the forward
pass. Decode *time* is a function of the architecture and the token
counts, not of what the weights contain, so the timing transfers; the
0.8833 quality figure comes from the banked run and is not re-measured
here. An artifact that timed one thing and implied another is the defect
this repository keeps correcting, so the two are kept apart.

    PYTHONPATH=src python3 scripts/measure_waybill_serving.py
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
# The same 8-vCPU on-demand quote VERDICT.md already uses, so the two
# pages cannot drift. An external price, not a measurement.
INSTANCE_USD_PER_HOUR = 0.290
RATE_DATE = "2026-08-19"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(
        REPO / "experiments" / "waybill_serving_throughput_2026-08-22.json"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from arcttt.lora import inject_lora
    from arcttt.model import TTTConfig

    holdout = [json.loads(line) for line in
               (RAW / "holdout.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]

    torch.manual_seed(1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32).to("cpu")
    config = TTTConfig(lora_rank=16, lora_alpha=32, epochs=1,
                       max_new_tokens=args.max_new_tokens,
                       max_sequence_tokens=8192,
                       gradient_checkpointing=False, chunked_loss_tokens=512)
    inject_lora(model, rank=config.lora_rank, alpha=config.lora_alpha)
    model.eval()

    per_doc, prompt_tokens, output_tokens = [], 0, 0
    for row in holdout:
        messages = [{"role": "user", "content": row["text"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        started = time.monotonic()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        elapsed = time.monotonic() - started
        n_in = int(inputs["input_ids"].shape[1])
        n_out = int(out.shape[1]) - n_in
        prompt_tokens += n_in
        output_tokens += n_out
        per_doc.append({"id": row["id"], "seconds": round(elapsed, 3),
                        "prompt_tokens": n_in, "output_tokens": n_out})
        print(f"  {row['id']} {elapsed:6.2f}s  in={n_in} out={n_out}",
              flush=True)

    seconds = [d["seconds"] for d in per_doc]
    mean_s = statistics.mean(seconds)
    cost_per_k = mean_s * 1000 / 3600 * INSTANCE_USD_PER_HOUR

    record = {
        "what": "Document-only serving throughput of the 0.5B configuration "
                "on the 30 held-out freight waybills. The half of the cost "
                "comparison that waybill_cost.py left blank.",
        "status": "TIMING ONLY. Quality is not measured here and must not "
                  "be read off this artifact: the banked rehearsal adapter "
                  "was not retained, so a LoRA of the same rank and shape "
                  "was injected and timed. Decode time depends on the "
                  "architecture and the token counts, not on what the "
                  "weights contain, so the timing transfers; the 0.8833 "
                  "quality figure comes from the banked run.",
        "model": MODEL, "device": "cpu", "dtype": "torch.float32",
        "configuration": "document-only (include_demos=False) — no "
                         "demonstrations in the prompt, which is what "
                         "putting the schema in the weights buys",
        "decode": "greedy, samples=1",
        "n_documents": len(per_doc),
        "seconds_per_document": {"mean": round(mean_s, 3),
                                 "median": round(statistics.median(seconds), 3),
                                 "min": round(min(seconds), 3),
                                 "max": round(max(seconds), 3)},
        "tokens_per_1k_documents": {
            "prompt": round(prompt_tokens / len(per_doc) * 1000),
            "output": round(output_tokens / len(per_doc) * 1000)},
        "instance_rate": {"usd_per_hour": INSTANCE_USD_PER_HOUR,
                          "quoted": RATE_DATE,
                          "source": "external cloud list price, 8-vCPU "
                                    "on-demand — the same quote VERDICT.md "
                                    "already uses; not a measurement, and "
                                    "deliberately cost-OVERSTATING against "
                                    "this 4-thread box"},
        "cost_per_1k_documents_usd": round(cost_per_k, 3),
        "excludes": "the one-time per-tenant adaptation cost, which is "
                    "amortised over a tenant's whole document volume and "
                    "is a different number from serving; it is reported "
                    "separately in the F artifacts (~272 s/tenant there).",
        "host": {"platform": platform.platform(),
                 "python": platform.python_version(),
                 "torch_threads": torch.get_num_threads()},
        "per_document": per_doc,
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")
    print(f"\nmean {mean_s:.2f}s/doc  ->  ~${cost_per_k:.2f} per 1,000 "
          f"documents at ${INSTANCE_USD_PER_HOUR}/hr ({RATE_DATE} quote)")
    print(f"tokens per 1k docs: {record['tokens_per_1k_documents']['prompt']:,}"
          f" in / {record['tokens_per_1k_documents']['output']:,} out")
    print(f"\nbanked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
