#!/usr/bin/env python3
"""M1: measure document-only serving throughput of the F configuration.

Motivation (COST_APPENDIX 2026-08-19): the honest cost comparison
against frontier-API-with-prompt-caching hinges on one unmeasured
number — how many documents per hour the adapted 0.5B actually serves.
This measures it, in the EXACT configuration whose quality is banked
(Addendum F PASS: docadapted adapter, document-only prompts,
vote/rescore decode 1 greedy + 4 sampled), plus the greedy-only floor
(deployment-cheap mode, quality UNMEASURED — labeled as such).

Method: restore a banked F adapter bit-identically (same checkpoint
mismatch check as the kernel), regenerate the same seed's corpus, and
decode eval documents doc-only, journaling wall-clock per document.
Hardware is recorded in the artifact; $/M-docs conversions are made by
the reader against cited cloud rates — this artifact stores only
measured times.

Usage:
    PYTHONPATH=src python3 scripts/measure_serving_throughput.py \
        <adapter.pt> <seed> <n_docs> <out.json>
"""

import json
import os
import pathlib
import platform
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
    from arcttt.text_ttt import TextPredictor, predict_text_voted

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device)
    config = TTTConfig(
        lora_rank=16, lora_alpha=32, epochs=1, max_new_tokens=512,
        max_sequence_tokens=8192, chunked_loss_tokens=512,
        gradient_checkpointing=True, shuffle_examples=True)
    predictor = TextPredictor(model, tokenizer, config, device)

    # bit-identical adapter restore, same checks as the kernel entry
    saved = torch.load(adapter_path, map_location="cpu")
    remove_lora(model)
    inject_lora(model, config.lora_rank, config.lora_alpha, use_rslora=True)
    lora_state = {n: p for n, p in model.named_parameters() if "lora_" in n}
    if set(saved) != set(lora_state):
        raise SystemExit(f"adapter checkpoint mismatch: {adapter_path}")
    with torch.no_grad():
        for n, p in lora_state.items():
            p.copy_(saved[n].to(p.device, p.dtype))
    model.train()  # parity with post-adapt state, same as the kernel

    task, schema = make_task(seed=seed, n_train=30, n_test=60,
                             task_id=f"throughput-seed{seed}")

    journal = out_path + ".journal.jsonl"
    rows = []
    if os.path.exists(journal):
        rows = [json.loads(l) for l in open(journal) if l.strip()]
    done = {(r["mode"], r["index"]) for r in rows}

    def run(mode: str):
        for index in range(n_docs):
            if (mode, index) in done:
                continue
            t0 = time.monotonic()
            if mode == "voted":
                text = predict_text_voted(predictor, task, index, samples=5,
                                          include_demos=False)
            else:
                texts = predictor.predict_text(task, index, samples=1,
                                               include_demos=False)
                text = texts[0] if texts else ""
            dt = time.monotonic() - t0
            row = {"mode": mode, "index": index, "seconds": round(dt, 3),
                   "output_chars": len(text or "")}
            rows.append(row)
            with open(journal, "a") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            print(f"{mode} doc {index}: {dt:.1f}s", flush=True)

    run("voted")   # the banked-quality configuration (F PASS decode)
    run("greedy")  # deployment floor, quality UNMEASURED

    def summarize(mode):
        secs = [r["seconds"] for r in rows if r["mode"] == mode]
        total = sum(secs)
        return {"n_docs": len(secs), "total_seconds": round(total, 1),
                "mean_seconds_per_doc": round(total / len(secs), 2),
                "docs_per_hour": round(3600 * len(secs) / total, 1)}

    artifact = {
        "artifact": "M1 document-only serving throughput, F configuration",
        "spec_context": "COST_APPENDIX 2026-08-19 M1; quality of the "
                        "voted mode is the banked Addendum F PASS config; "
                        "greedy mode quality is UNMEASURED",
        "adapter": os.path.basename(adapter_path),
        "seed": seed,
        "tenant": schema.tenant_id,
        "model": model_id,
        "device": "cpu",
        "dtype": "torch.float32 (matches banked runs; quantized serving "
                 "would only improve throughput)",
        "hardware": {
            "cpu_model": platform.processor() or open("/proc/cpuinfo").read().split("model name\t: ")[1].split("\n")[0],
            "cpu_count": os.cpu_count(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "decode": {"voted": "1 greedy + 4 sampled (T=0.7), canonical-JSON "
                            "pooling — the banked F decode",
                   "greedy": "single greedy decode"},
        "include_demos": False,
        "modes": {m: summarize(m) for m in ("voted", "greedy")},
        "per_doc": rows,
    }
    tmp = out_path + ".tmp"
    open(tmp, "w").write(json.dumps(artifact, indent=1))
    os.replace(tmp, out_path)
    print(json.dumps(artifact["modes"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
