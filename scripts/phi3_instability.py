#!/usr/bin/env python3
"""Addendum Y: does the decoder reproduce, and when does it stop?

    PYTHONPATH=src python3 scripts/phi3_instability.py --sequence phi3-mini --dtype bfloat16
    PYTHONPATH=src python3 scripts/phi3_instability.py --sequence qwen2.5-0.5b --dtype bfloat16
    PYTHONPATH=src python3 scripts/phi3_instability.py --sequence qwen2.5-0.5b --dtype float32
    PYTHONPATH=src python3 scripts/phi3_instability.py --threads phi3-mini --dtype bfloat16
    PYTHONPATH=src python3 scripts/phi3_instability.py --read

Preregistration: docs/research/ADDENDUM_Y_PROTOCOL.md (frozen 2026-09-17,
before any arm ran). Every decode is checkpointed by (document, pass), so
an interrupted run resumes in the same process order it was frozen in --
the order IS part of the design, because state dependence is what is
being measured.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import cord_fence_tax as cft  # noqa: E402
import cord_constrained_families as v  # noqa: E402
import cord_decoder_isolation as x  # noqa: E402  (prompt rendering, TOP_K, caps)

N_DOCS = 10
THREAD_DOCS = ("cord-004", "cord-000")
PAIRS = {  # (family, dtype) -> (constrained banked cell, generate banked cell) or None
    ("phi3-mini", "bfloat16"): (
        v.CELLS_DIR / "phi3-mini_schema_constrained.json",
        v.CELLS["phi3-mini"][2]),
    ("qwen2.5-0.5b", "bfloat16"): (None, None),
    ("qwen2.5-0.5b", "float32"): (
        v.CELLS_DIR / "qwen2.5-0.5b_schema_constrained.json",
        v.CELLS["qwen2.5-0.5b"][2]),
}
OUT_DIR = REPO / "experiments" / "phi3_instability_cells"
OUT = REPO / "experiments" / "phi3_instability_2026-09-17.json"
WORK = REPO / "work" / "y"


def _docs():
    return v._holdout()[:N_DOCS]


def _load(family: str, dtype_name: str, threads: int = 4):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = v.CELLS[family][0]
    torch.set_num_threads(threads)
    torch.manual_seed(1)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype_name)).eval()
    return model, tok, model_id


def _decode_c(model, tok, ids):
    from arcttt.constrained_json import constrained_greedy_generate
    r = constrained_greedy_generate(model, tok, ids, max_new_tokens=x.MAX_NEW_TOKENS, top_k=x.TOP_K)
    return r.text


def _decode_g(model, tok, ids):
    import torch
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=x.MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


# The frozen order of the process: (pass, decoder). Y1 is a1 vs a2, Y2 is
# b vs a1, Y3 is g2 vs g1.
PASSES = (("a1", "C"), ("a2", "C"), ("b", "C"), ("g1", "G"), ("g2", "G"))


def _plan(docs):
    """Pass A interleaves a1/a2 per document; the later passes sweep."""
    plan = []
    for d in docs:
        plan += [(d["id"], "a1"), (d["id"], "a2")]
    for name in ("b", "g1", "g2"):
        plan += [(d["id"], name) for d in docs]
    return plan


def run_sequence(family: str, dtype_name: str) -> int:
    cft._assert_vocab()
    cell = OUT_DIR / f"{family}_{dtype_name}_sequence.json"
    if cell.exists():
        print(f"cell banked already: {cell.name}")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    docs = _docs()
    by_id = {d["id"]: d for d in docs}
    config_key = f"{v.CELLS[family][0]}|{dtype_name}|Y-sequence|n={N_DOCS}|mnt={x.MAX_NEW_TOKENS}"
    ckpt_path = WORK / f"{family}_{dtype_name}_sequence.ckpt.jsonl"
    done = x._checkpoint(ckpt_path, config_key)  # keyed by "id|pass"
    plan = _plan(docs)
    model = tok = None
    texts = {}       # (id, pass) -> text
    seconds = {}
    with open(ckpt_path, "a", encoding="utf-8") as ckpt:
        for i, (doc_id, pname) in enumerate(plan):
            key = f"{doc_id}|{pname}"
            if key in done:
                texts[(doc_id, pname)] = done[key]["raw"]
                seconds[key] = done[key]["seconds"]
                continue
            if model is None:
                model, tok, _ = _load(family, dtype_name)
                if done:
                    print(f"[Y:{family}:{dtype_name}] resuming {len(done)}/{len(plan)} "
                          "(a resumed process has a different history than an unbroken "
                          "one; the resume point is banked)", flush=True)
            ids, digest = x._prompt_ids(tok, family, by_id[doc_id]["text"])
            began = time.monotonic()
            decoder = dict(PASSES)[pname]
            text = _decode_c(model, tok, ids) if decoder == "C" else _decode_g(model, tok, ids)
            took = round(time.monotonic() - began, 1)
            texts[(doc_id, pname)] = text
            seconds[key] = took
            ckpt.write(json.dumps({"config": config_key, "id": key, "raw": text,
                                   "seconds": took, "prompt_sha256_16": digest,
                                   "position_in_process": i}, ensure_ascii=False) + "\n")
            ckpt.flush()
            print(f"[Y:{family}:{dtype_name}] {doc_id} {pname} {i + 1}/{len(plan)} {took}s", flush=True)

    resume_points = sorted({int(r.get("position_in_process", -1)) for r in done.values()})
    record = {
        "what": f"Addendum Y sequence cell: {v.CELLS[family][0]} in {dtype_name}. Ten "
                "documents; per document the constrained decoder twice back to back "
                "(a1, a2), then a third pass (b) after twenty decodes, then "
                "model.generate twice in two sweeps (g1, g2). One process, this order.",
        "protocol": "docs/research/ADDENDUM_Y_PROTOCOL.md (frozen 2026-09-17)",
        "family": family, "model": v.CELLS[family][0], "dtype": dtype_name,
        "n": N_DOCS, "order": [f"{d}|{p}" for d, p in plan],
        "resumed_from_checkpoint_entries": len(done),
        "resume_note": ("Every entry decoded in a process that had already been "
                        "interrupted is marked by its position; a resume changes "
                        "process history, which is the quantity under test, so the "
                        "reader reports it beside the counts rather than hiding it."
                        if done else None),
        "per_pass_seconds": seconds,
        "texts": {f"{d}|{p}": t for (d, p), t in texts.items()},
    }
    x._bank(cell, record)
    return 0


def run_threads(family: str, dtype_name: str) -> int:
    cft._assert_vocab()
    cell = OUT_DIR / f"{family}_{dtype_name}_threads.json"
    if cell.exists():
        print(f"cell banked already: {cell.name}")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import torch
    docs = {d["id"]: d for d in _docs()}
    model, tok, model_id = _load(family, dtype_name, threads=4)
    rows = {}
    for doc_id in THREAD_DOCS:
        ids, _ = x._prompt_ids(tok, family, docs[doc_id]["text"])
        out = {}
        for n in (4, 1):
            torch.set_num_threads(n)
            began = time.monotonic()
            out[n] = _decode_c(model, tok, ids)
            print(f"[Y:{family}:threads] {doc_id} threads={n} {round(time.monotonic() - began, 1)}s",
                  flush=True)
        rows[doc_id] = {"threads_4": out[4], "threads_1": out[1], "differ": out[4] != out[1]}
    torch.set_num_threads(4)
    x._bank(cell, {
        "what": f"Addendum Y thread-count cell: {model_id} in {dtype_name}, the constrained "
                "decoder at 4 threads and at 1 thread on two documents, one process.",
        "protocol": "docs/research/ADDENDUM_Y_PROTOCOL.md (frozen 2026-09-17)",
        "family": family, "model": model_id, "dtype": dtype_name,
        "documents": rows,
    })
    return 0


# --------------------------------------------------------------------------
# frozen readings -- arithmetic only
# --------------------------------------------------------------------------

def pair_reading(y1: int, y2: int, y3: int) -> str:
    if y1 >= 1:
        return f"IMMEDIATE-REPEAT UNSTABLE (Y1={y1})"
    if y2 + y3 >= 1:
        return f"STATE-DEPENDENT (Y2={y2}, Y3={y3})"
    return "REPRODUCES"


def dtype_reading(qwen_bf16: str, qwen_fp32: str) -> str:
    if qwen_fp32 != "REPRODUCES":
        return ("NOT A DTYPE STORY: the float32 control does not reproduce; Addendum X's "
                "'four families reproduce' narrows by erratum to what this run shows")
    if qwen_bf16 != "REPRODUCES":
        return "BFLOAT16 IS SUFFICIENT: the same family reproduces in float32 and not in bfloat16"
    return ("BFLOAT16 IS NOT SUFFICIENT ON THIS FAMILY: both Qwen pairs reproduce, so "
            "Phi-3's instability is not explained by its dtype alone")


def thread_reading(y5: int) -> str:
    return "THREAD-SENSITIVE" if y5 >= 1 else "THREAD-INSENSITIVE ON THIS SAMPLE"


def read() -> int:
    fc = cft._fc()
    strip = lambda t: fc.strip_fence(t)[0]  # noqa: E731
    needed = [OUT_DIR / f"{f}_{d}_sequence.json" for f, d in PAIRS] + \
             [OUT_DIR / "phi3-mini_bfloat16_threads.json"]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        print("WITHHELD: missing cells " + ", ".join(missing))
        return 1
    rows, per_pair = {}, {}
    for (family, dtype_name), (c_cell, g_cell) in PAIRS.items():
        cell = json.loads((OUT_DIR / f"{family}_{dtype_name}_sequence.json").read_text())
        T = cell["texts"]
        ids = [d["id"] for d in _docs()]
        y1 = sum(strip(T[f"{d}|a1"]) != strip(T[f"{d}|a2"]) for d in ids)
        y2 = sum(strip(T[f"{d}|b"]) != strip(T[f"{d}|a1"]) for d in ids)
        y3 = sum(strip(T[f"{d}|g2"]) != strip(T[f"{d}|g1"]) for d in ids)
        then_now = {}
        if c_cell is not None:
            bc = json.loads(c_cell.read_text())["predictions"]
            then_now["constrained_a1_vs_banked"] = sum(strip(T[f"{d}|a1"]) != strip(bc[d]) for d in ids)
        if g_cell is not None:
            bg = json.loads(g_cell.read_text())["predictions"]
            then_now["generate_g1_vs_banked"] = sum(strip(T[f"{d}|g1"]) != strip(bg[d]) for d in ids)
        reading = pair_reading(y1, y2, y3)
        per_pair[f"{family}/{dtype_name}"] = reading
        rows[f"{family}/{dtype_name}"] = {
            "Y1_immediate_repeat_differs": y1, "Y2_state_differs": y2,
            "Y3_generate_state_differs": y3, "then_vs_now": then_now,
            "documents_differing": {
                "Y1": [d for d in ids if strip(T[f"{d}|a1"]) != strip(T[f"{d}|a2"])],
                "Y2": [d for d in ids if strip(T[f"{d}|b"]) != strip(T[f"{d}|a1"])],
                "Y3": [d for d in ids if strip(T[f"{d}|g2"]) != strip(T[f"{d}|g1"])]},
            "resumed_from_checkpoint_entries": cell.get("resumed_from_checkpoint_entries", 0),
            "reading": reading,
        }
        print(f"{family:13s} {dtype_name:8s} Y1={y1} Y2={y2} Y3={y3} then_vs_now={then_now}  {reading}")
    th = json.loads((OUT_DIR / "phi3-mini_bfloat16_threads.json").read_text())
    y5 = sum(r["differ"] for r in th["documents"].values())
    # Post-hoc, computed not read: does the text Addendum X's gate flaked to
    # equal the 1-thread decode here? If so, the flake IS a reduction-order
    # change and nothing else. Suggested by a simulated reviewer who checked
    # the bytes; the frozen readings above do not depend on it.
    gate_flake = None
    run1 = x.OUT_DIR / "phi3-mini_determinism_run1.json"
    if run1.exists() and "cord-004" in th["documents"]:
        docs = {d["id"]: d for d in json.loads(run1.read_text())["documents"]}
        flaked = docs.get("cord-004", {}).get("rerun_C_text")
        if flaked is not None:
            gate_flake = {
                "x_gate_run1_flaked_text_equals_y_1_thread_text":
                    strip(flaked) == strip(th["documents"]["cord-004"]["threads_1"]),
                "y_4_thread_text_equals_v_banked_text":
                    strip(th["documents"]["cord-004"]["threads_4"]) == strip(
                        json.loads(PAIRS[("phi3-mini", "bfloat16")][0].read_text())["predictions"]["cord-004"]),
                "what_it_means": ("If both are true, the gate's flake was byte-for-byte the "
                                  "1-thread decode: a change of reduction order, not a random "
                                  "decoder and not a different prompt."),
            }
    dtype = dtype_reading(per_pair["qwen2.5-0.5b/bfloat16"], per_pair["qwen2.5-0.5b/float32"])
    threads = thread_reading(y5)
    reproducing = sorted(k for k, r in per_pair.items() if r == "REPRODUCES")
    record = {
        "what": "Addendum Y: the decoder's reproducibility measured directly on the family "
                "that failed Addendum X's gate, and on a second family in both dtypes so "
                "the dtype hypothesis can be wrong in public.",
        "protocol": "docs/research/ADDENDUM_Y_PROTOCOL.md (frozen 2026-09-17)",
        "date": "2026-09-17", "environment": x._environment(),
        "rows": rows, "per_pair": per_pair,
        "Y5_thread_count_differs_of_2": y5,
        "post_hoc_gate_flake_identity": gate_flake,
        "thread_reading": threads, "dtype_reading": dtype,
        "pairs_that_may_be_called_reproducible": reproducing,
        "the_sentence_licensed": (
            "'This repository's cells reproduce' may be said only of: "
            + (", ".join(reproducing) if reproducing else "no pair measured here")
            + " -- named, on ten documents, never unqualified."),
        "predictions_written_before_the_run": [
            "Y1 = 0 on all three pairs",
            "Phi-3/bfloat16 reads STATE-DEPENDENT",
            "Phi-3/bfloat16 is THREAD-SENSITIVE",
            "the Qwen pairs are not predicted"],
        "how_to_recompute": "PYTHONPATH=src python3 scripts/phi3_instability.py --read",
    }
    OUT.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nthreads: {threads} (Y5={y5}/2)\ndtype: {dtype}\nbanked: {OUT.relative_to(REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequence", choices=sorted({f for f, _ in PAIRS}))
    ap.add_argument("--threads", choices=["phi3-mini"])
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--read", action="store_true")
    a = ap.parse_args()
    if a.sequence:
        if (a.sequence, a.dtype) not in PAIRS:
            raise SystemExit(f"{a.sequence}/{a.dtype} is not a preregistered pair")
        return run_sequence(a.sequence, a.dtype)
    if a.threads:
        return run_threads(a.threads, a.dtype)
    if a.read:
        return read()
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
