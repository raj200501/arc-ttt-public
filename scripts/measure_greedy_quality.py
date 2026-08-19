#!/usr/bin/env python3
"""M1b: quality of GREEDY document-only serving (the cheap decode).

The throughput measurement (novel_serving_throughput_cpu_2026-08-19)
showed the banked-quality decode (1 greedy + 4 sampled votes) costs
4.7x more serving time than a single greedy decode. This measures what
that 4.7x buys: greedy-only predictions on the same adapter/corpus,
scored against gold, raw predictions stored. Context measurement for
the cost analysis — NOT a gate arm; never pooled with kernel arms.

Usage:
    PYTHONPATH=src python3 scripts/measure_greedy_quality.py \
        <adapter.pt> <seed> <n_docs> <out.json>
"""

import json
import os
import pathlib
import sys
import time

import torch


def main() -> int:
    adapter_path, seed, n_docs, out_path = (
        sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from arcttt.model import TTTConfig
    from arcttt.lora import inject_lora, remove_lora
    from arcttt.novel_schema import make_task
    from arcttt.text_ttt import TextPredictor, score_text_output

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to("cpu")
    config = TTTConfig(
        lora_rank=16, lora_alpha=32, epochs=1, max_new_tokens=512,
        max_sequence_tokens=8192, chunked_loss_tokens=512,
        gradient_checkpointing=True, shuffle_examples=True)
    predictor = TextPredictor(model, tokenizer, config, "cpu")

    saved = torch.load(adapter_path, map_location="cpu")
    remove_lora(model)
    inject_lora(model, config.lora_rank, config.lora_alpha, use_rslora=True)
    lora_state = {n: p for n, p in model.named_parameters() if "lora_" in n}
    if set(saved) != set(lora_state):
        raise SystemExit(f"adapter checkpoint mismatch: {adapter_path}")
    with torch.no_grad():
        for n, p in lora_state.items():
            p.copy_(saved[n].to(p.device, p.dtype))
    model.train()  # parity with post-adapt state

    task, schema = make_task(seed=seed, n_train=30, n_test=60,
                             task_id=f"greedyq-seed{seed}")
    journal = out_path + ".journal.jsonl"
    rows = []
    if os.path.exists(journal):
        rows = [json.loads(l) for l in open(journal) if l.strip()]
    done = {r["index"] for r in rows}
    for index in range(n_docs):
        if index in done:
            continue
        t0 = time.monotonic()
        texts = predictor.predict_text(task, index, samples=1,
                                       include_demos=False)
        text = texts[0] if texts else ""
        gold = task.test[index].output_text
        score = score_text_output(text, gold)
        row = {"index": index, "prediction": text,
               "micro_f1": round(score.micro_f1, 4),
               "exact": score.micro_f1 == 1.0,
               "seconds": round(time.monotonic() - t0, 2)}
        rows.append(row)
        with open(journal, "a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(f"doc {index}: f1={row['micro_f1']}", flush=True)

    mean = sum(r["micro_f1"] for r in rows) / len(rows)
    artifact = {
        "artifact": "M1b greedy-decode document-only quality (context "
                    "measurement for the cost analysis; NOT a gate arm)",
        "adapter": os.path.basename(adapter_path),
        "seed": seed, "tenant": schema.tenant_id,
        "model": model_id, "device": "cpu", "dtype": "torch.float32",
        "decode": "single greedy decode, include_demos=False",
        "banked_comparison": "same adapter/corpus as the Addendum F "
                             "seed-{} docadapted arm (voted decode)".format(seed),
        "n": len(rows),
        "mean_micro_f1": round(mean, 4),
        "exact": sum(1 for r in rows if r["exact"]),
        "invalid_or_empty": sum(1 for r in rows if not r["prediction"]),
        "per_doc": rows,
    }
    tmp = out_path + ".tmp"
    open(tmp, "w").write(json.dumps(artifact, indent=1))
    os.replace(tmp, out_path)
    print(json.dumps({k: artifact[k] for k in ("mean_micro_f1", "exact", "n")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
