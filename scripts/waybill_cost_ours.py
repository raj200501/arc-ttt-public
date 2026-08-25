#!/usr/bin/env python3
"""Our serving cost on the waybill corpus, corrected for a known bias.

`measure_waybill_serving.py` times document-only decoding on the 30
held-out waybills. It has to inject an UNTRAINED LoRA, because the banked
rehearsal adapter was not retained -- and an untrained adapter does not
know when to stop. It drifts: 375 output tokens on one document where the
banked adapted arm's actual answer is a compact JSON object of roughly a
quarter that length.

Wall-clock per document from that run is therefore **biased upward**: it
measures the cost of generating tokens a trained adapter would never
emit. Publishing it unadjusted would overstate our own cost, which is the
safe direction and still the wrong number.

This corrects it the only honest way available: take **tokens per second**
from the timing run -- that is weight-independent, it is what the
hardware does -- and apply it to the **actual output lengths of the
banked adapted predictions**, tokenized with the same tokenizer. The
result is what the measured hardware would take to produce the answers
the measured model actually produced.

Both numbers are reported. The unadjusted one is kept visible so nobody
has to take the correction on trust.

    PYTHONPATH=src python3 scripts/waybill_cost_ours.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

RAW = REPO / "experiments" / "blind_rehearsal_2026-08-20_raw"
TIMING = REPO / "experiments" / "waybill_serving_throughput_2026-08-22.json"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
INSTANCE_USD_PER_HOUR = 0.290
RATE_DATE = "2026-08-19"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(
        REPO / "experiments" / "waybill_cost_ours_2026-08-22.json"))
    args = parser.parse_args()

    if not TIMING.exists():
        raise SystemExit(f"timing run not banked yet: {TIMING}")
    timing = json.loads(TIMING.read_text(encoding="utf-8"))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # What the banked ADAPTED arm actually emitted, tokenized.
    adapted = [json.loads(line) for line in
               (RAW / "predictions_adapted_greedy.jsonl")
               .read_text(encoding="utf-8").splitlines() if line.strip()]
    real_out = []
    for row in adapted:
        text = json.dumps(row["prediction"], sort_keys=True) \
            if row["prediction"] is not None else ""
        real_out.append(len(tokenizer(text)["input_ids"]))

    per_doc = timing["per_document"]
    total_seconds = sum(d["seconds"] for d in per_doc)
    total_generated = sum(d["output_tokens"] for d in per_doc)
    tokens_per_second = total_generated / total_seconds

    mean_prompt = sum(d["prompt_tokens"] for d in per_doc) / len(per_doc)
    mean_real_out = sum(real_out) / len(real_out)
    # Prompt processing is a forward pass over a short prompt; the decode
    # loop dominates on CPU. Corrected time is the decode of the tokens the
    # trained arm actually produced, at the measured rate.
    corrected_seconds = mean_real_out / tokens_per_second
    corrected_cost_k = corrected_seconds * 1000 / 3600 * INSTANCE_USD_PER_HOUR
    unadjusted_cost_k = (timing["seconds_per_document"]["mean"] * 1000
                         / 3600 * INSTANCE_USD_PER_HOUR)

    record = {
        "what": "Our document-only serving cost per 1,000 waybills, "
                "corrected for the untrained-adapter drift in the timing "
                "run.",
        "status": "The RATE (tokens/second) is measured on this hardware "
                  "and is weight-independent. The OUTPUT LENGTHS are the "
                  "banked adapted arm's actual predictions, tokenized. The "
                  "instance price is an external quote, not a measurement.",
        "why_a_correction_was_needed":
            "The timing run had to inject an untrained LoRA (the banked "
            "rehearsal adapter was not retained). An untrained adapter does "
            "not stop: it generated "
            f"{total_generated / len(per_doc):.0f} tokens per document on "
            f"average, against {mean_real_out:.0f} for the answers the "
            "trained arm actually produced. Unadjusted wall-clock therefore "
            "overstates our cost -- the safe direction, and still wrong.",
        "measured_tokens_per_second": round(tokens_per_second, 3),
        "mean_prompt_tokens": round(mean_prompt, 1),
        "mean_output_tokens_untrained_adapter": round(
            total_generated / len(per_doc), 1),
        "mean_output_tokens_banked_adapted_arm": round(mean_real_out, 1),
        "seconds_per_document_unadjusted":
            timing["seconds_per_document"]["mean"],
        "seconds_per_document_corrected": round(corrected_seconds, 3),
        "instance_rate": {"usd_per_hour": INSTANCE_USD_PER_HOUR,
                          "quoted": RATE_DATE,
                          "source": "external 8-vCPU on-demand list price; "
                                    "deliberately cost-overstating against "
                                    "this 4-thread box"},
        "cost_per_1k_documents_usd_unadjusted": round(unadjusted_cost_k, 3),
        "cost_per_1k_documents_usd": round(corrected_cost_k, 3),
        "excludes": "the one-time per-tenant adaptation cost, amortised "
                    "over that tenant's volume and reported separately.",
        "quality_note": "This artifact contains no quality claim. Our "
                        "adapted arm scores 0.8833 on these documents and "
                        "the hosted tier scores 0.9708-1.0000; that "
                        "comparison lives in Addendum I and is not "
                        "improved by being cheaper.",
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")

    print(f"measured rate            : {tokens_per_second:.2f} tok/s")
    print(f"untrained-adapter output : "
          f"{total_generated / len(per_doc):.0f} tok/doc  -> "
          f"${unadjusted_cost_k:.2f}/1k  (biased UP)")
    print(f"banked adapted output    : {mean_real_out:.0f} tok/doc  -> "
          f"${corrected_cost_k:.2f}/1k  (corrected)")
    print(f"\nbanked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
