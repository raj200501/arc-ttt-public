#!/usr/bin/env python3
"""Addendum V: the JSON-constrained decoder on five families, schema-only.

    PYTHONPATH=src python3 scripts/cord_constrained_families.py --cell smollm2-1.7b
    PYTHONPATH=src python3 scripts/cord_constrained_families.py --read

Preregistration: docs/research/ADDENDUM_V_PROTOCOL.md (frozen 2026-09-08,
commit 379346f, before any cell ran). Same prompts as the Addendum S/T
schema-only cells (SCHEMA_INSTRUCTION verbatim, the family's own chat
template), the first 50 receipts in the comparator cell's document order,
E7's decoder unchanged (src/arcttt/constrained_json.py, top-k 16, leading
fence tolerated, greedy fallback counted). Per-document checkpoints keyed
by config; --read withholds until all five cells exist and applies the
frozen readings by arithmetic.
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

import cord_fence_tax as cft  # noqa: E402  (SCHEMA_INSTRUCTION, FAMILIES, VOCAB, helpers)

N_DOCS = 50
TOP_K = 16
MAX_NEW_TOKENS = cft.MAX_NEW_TOKENS
MAX_SEQ = cft.MAX_SEQ
CELLS_DIR = REPO / "experiments" / "cord_constrained_families_cells"
OUT = REPO / "experiments" / "cord_constrained_families_2026-09-08.json"
WORK = REPO / "work" / "v"

# family key -> (model id, dtype, greedy comparator cell)
CELLS = {
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", "float32",
                     REPO / "experiments" / "cord_fence_tax_cells" / "0.5b_schema.json"),
    "smollm2-1.7b": (*cft.FAMILIES["smollm2-1.7b"],
                     cft.FAMILIES_CELLS_DIR / "smollm2-1.7b_schema.json"),
    "granite-2b": (*cft.FAMILIES["granite-2b"], cft.FAMILIES_CELLS_DIR / "granite-2b_schema.json"),
    "phi3-mini": (*cft.FAMILIES["phi3-mini"], cft.FAMILIES_CELLS_DIR / "phi3-mini_schema.json"),
    "falcon3-1b": (*cft.FAMILIES["falcon3-1b"], cft.FAMILIES_CELLS_DIR / "falcon3-1b_schema.json"),
}


def _holdout() -> list[dict]:
    """The schema-only prompts, rendered exactly as the S/T cells rendered
    them (same corpus adapter), restricted to the first N_DOCS ids."""
    from arcttt.text_task import from_cord_gt
    rows = [json.loads(l) for l in
            (REPO / "demo" / "cord_validation.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    task_all = from_cord_gt(rows[:1], rows, task_id="v-schema")
    return [{"id": f"cord-{i:03d}", "text": task_all.test[i].input_text}
            for i in range(N_DOCS)]


def run_cell(family: str) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from arcttt.constrained_json import constrained_greedy_generate

    cft._assert_vocab()
    model_id, dtype_name, comparator = CELLS[family]
    cell_path = CELLS_DIR / f"{family}_schema_constrained.json"
    if cell_path.exists():
        print(f"cell banked already: {cell_path.name}")
        return 0
    if not comparator.exists():
        raise SystemExit(f"comparator cell missing: {comparator}")
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    holdout = _holdout()
    comp = json.loads(comparator.read_text(encoding="utf-8"))
    comp_ids = list(comp["predictions"])[:N_DOCS]
    if comp_ids != [h["id"] for h in holdout]:
        raise SystemExit("comparator document order does not match the holdout")

    config_key = (f"{model_id}|schema|{dtype_name}|constrained_topk={TOP_K}"
                  f"|mnt={MAX_NEW_TOKENS}|seq={MAX_SEQ}|n={N_DOCS}|V")
    ckpt_path = WORK / f"{family}_schema_constrained.ckpt.jsonl"
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
    if done:
        print(f"[V:{family}] resuming {len(done)}/{len(holdout)}", flush=True)

    torch.set_num_threads(4)
    torch.manual_seed(1)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype_name))
    model.eval()

    predictions, seconds, decode = {}, {}, {}
    resumed = len(done)
    with open(ckpt_path, "a", encoding="utf-8") as ckpt:
        for i, row in enumerate(holdout):
            if row["id"] in done:
                d = done[row["id"]]
                predictions[row["id"]], seconds[row["id"]], decode[row["id"]] = d["raw"], d["seconds"], d["decode"]
                continue
            text_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": cft.SCHEMA_INSTRUCTION + "\n\n" + row["text"]}],
                tokenize=False, add_generation_prompt=True)
            ids = tokenizer(text_prompt, return_tensors="pt").input_ids
            if ids.shape[1] > MAX_SEQ:
                raise SystemExit(f"prompt {row['id']} exceeded {MAX_SEQ}")
            began = time.monotonic()
            res = constrained_greedy_generate(model, tokenizer, ids,
                                              max_new_tokens=MAX_NEW_TOKENS, top_k=TOP_K)
            took = round(time.monotonic() - began, 1)
            acct = {"fallbacks": res.fallbacks, "constrained_steps": res.constrained_steps,
                    "steps": res.steps, "stopped_on": res.stopped_on}
            predictions[row["id"]], seconds[row["id"]], decode[row["id"]] = res.text, took, acct
            ckpt.write(json.dumps({"config": config_key, "id": row["id"], "raw": res.text,
                                   "seconds": took, "decode": acct}, ensure_ascii=False) + "\n")
            ckpt.flush()
            print(f"[V:{family}] {row['id']} {i + 1}/{len(holdout)} {took}s "
                  f"fallbacks={res.fallbacks} steps={res.steps} stop={res.stopped_on}", flush=True)

    record = {
        "what": f"Addendum V cell: {model_id}, schema-only prompts re-decoded with the "
                "JSON-constrained greedy decoder; raw text per document. Readings live in "
                "the assembled artifact (--read).",
        "protocol": "docs/research/ADDENDUM_V_PROTOCOL.md (frozen 2026-09-08, commit 379346f)",
        "model": model_id, "dtype": dtype_name, "regime": "schema", "n": len(holdout), "k": 0,
        "schema_instruction": cft.SCHEMA_INSTRUCTION,
        "comparator_cell": str(comparator.relative_to(REPO)),
        "decoder": {"kind": "constrained greedy", "top_k": TOP_K, "max_new_tokens": MAX_NEW_TOKENS,
                    "allow_leading_fence": True, "validator": "arcttt.constrained_json.is_json_prefix",
                    "stop": "eos | complete | max_new_tokens"},
        "decode": f"constrained greedy top_k={TOP_K}, max_new_tokens={MAX_NEW_TOKENS}, {dtype_name}, CPU",
        "resumed_from_checkpoint": resumed,
        "per_document_seconds": seconds,
        "per_document_decode": decode,
        "predictions": predictions,
    }
    tmp = cell_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(cell_path)
    print(f"banked: {cell_path.name} ({len(predictions)} raw outputs)")
    return 0


# --------------------------------------------------------------------------
# frozen readings -- arithmetic only
# --------------------------------------------------------------------------

def family_reading(i_greedy: int, i_constrained: int, regressions: int) -> str:
    """The frozen per-family reading. The protocol's regression rule --
    `regressions >= 3` is a regression finding regardless of the invalid
    counts -- is appended by arithmetic here; the first banked artifact
    carried it only in prose (protocol erratum 5)."""
    if i_greedy <= 1:
        base = "UNTESTABLE AT SIZE (greedy invalid <= 1)"
    elif i_constrained <= 1 and regressions == 0:
        base = "REMOVES"
    elif i_constrained <= i_greedy // 2:
        base = "REDUCES"
    else:
        base = "NO EFFECT"
    if regressions >= 3:
        base += f"; REGRESSION FINDING ({regressions} >= 3)"
    return base


def combine(per_family: dict) -> str:
    testable = {f: v for f, v in per_family.items() if not v.startswith("UNTESTABLE")}
    n_removes = sum(v.startswith("REMOVES") for v in testable.values())
    exceptions = sorted(f for f, v in testable.items() if v.startswith("NO EFFECT"))
    regress = sorted(f for f, v in per_family.items() if "REGRESSION FINDING" in v)
    n_fam = len(per_family)
    tail = (f" REGRESSION FINDING in {', '.join(regress)} (published regardless of the invalid counts)."
            if regress else "")
    if exceptions:
        return (f"V EXCEPTION IN {', '.join(exceptions)}: named at full size; the sentence becomes "
                f"'on N of the {n_fam} families tested' -- never 'across families'.{tail} "
                f"Per-family: {json.dumps(per_family)}")
    if n_removes >= 4:
        return (f"V HOLDS: the decoder removes invalid JSON in {n_removes} of {n_fam} families and "
                f"no family is an exception.{tail} Per-family: {json.dumps(per_family)}")
    return (f"V MIXED: REMOVES in {n_removes} of {n_fam} families, no exception -- all counts "
            f"publish, no headline.{tail} Per-family: {json.dumps(per_family)}")


def _sign_test(wins: int, losses: int) -> float:
    from math import comb
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n)


def read() -> int:
    from arcttt.scoring import field_micro_f1, parse_json_object
    from arcttt.text_task import TextTaskFormatError
    fc = cft._fc()
    missing = [f for f in CELLS if not (CELLS_DIR / f"{f}_schema_constrained.json").exists()]
    if missing:
        print("WITHHELD: missing cells " + ", ".join(missing) +
              " -- the reading is stated once every preregistered cell exists.")
        return 1
    gold = {json.loads(l)["id"]: json.loads(l)["gold"] for l in
            (cft.SPLIT_DIR / "gold.jsonl").read_text().splitlines() if l.strip()}
    for r in cft._read_jsonl(cft.SPLIT_DIR / "train.jsonl"):
        gold[r["id"]] = r["gold"]

    def classify(text):
        body, fenced = fc.strip_fence(text)
        try:
            return parse_json_object(body), fenced
        except TextTaskFormatError:
            return None, fenced

    rows, per_family = [], {}
    for family, (model_id, dtype_name, comparator) in CELLS.items():
        cell = json.loads((CELLS_DIR / f"{family}_schema_constrained.json").read_text())
        comp = json.loads(comparator.read_text())
        ids = list(cell["predictions"])
        inv_g = inv_c = fen_g = fen_c = regressions = 0
        wins = losses = ties = 0
        deltas = []
        for doc_id in ids:
            og, fg = classify(comp["predictions"][doc_id])
            oc, fcd = classify(cell["predictions"][doc_id])
            inv_g += og is None; inv_c += oc is None; fen_g += fg; fen_c += fcd
            regressions += (og is not None and oc is None)
            sg = field_micro_f1(og, gold[doc_id]) if og is not None else 0.0
            sc = field_micro_f1(oc, gold[doc_id]) if oc is not None else 0.0
            d = sc - sg; deltas.append(d)
            wins += d > 1e-9; losses += d < -1e-9; ties += abs(d) <= 1e-9
        acct = cell["per_document_decode"]
        fallbacks = sum(a["fallbacks"] for a in acct.values())
        csteps = sum(a["constrained_steps"] for a in acct.values())
        steps = sum(a["steps"] for a in acct.values())
        stops = {}
        for a in acct.values():
            stops[a["stopped_on"]] = stops.get(a["stopped_on"], 0) + 1
        reading = family_reading(inv_g, inv_c, regressions)
        per_family[family] = reading
        mean_d = round(sum(deltas) / len(deltas), 4)
        p = _sign_test(wins, losses) if mean_d >= 0 else _sign_test(losses, wins)
        rows.append({
            "family": family, "model": model_id, "n": len(ids),
            "invalid_greedy": inv_g, "invalid_constrained": inv_c, "regressions": regressions,
            "fenced_greedy": fen_g, "fenced_constrained": fen_c,
            "score_mean_delta_constrained_minus_greedy": mean_d,
            "score_wins_losses_ties": [wins, losses, ties], "score_sign_test_p": round(p, 4),
            "score_reading": ("GAIN" if mean_d >= 0.01 and p <= 0.05 else
                              "LOSS" if mean_d <= -0.01 and p <= 0.05 else "UNSEPARATED"),
            "fallbacks": fallbacks, "constrained_steps": csteps, "steps": steps,
            "fallback_share_of_steps": round(fallbacks / steps, 4) if steps else None,
            "stopped_on": stops, "reading": reading,
        })
        print(f"{family:13s} invalid {inv_g:2d} -> {inv_c:2d}  regressions {regressions}  "
              f"fenced {fen_g:2d} -> {fen_c:2d}  dF1 {mean_d:+.4f} ({wins}W/{losses}L/{ties}T p={p:.3f})  "
              f"fallbacks {fallbacks}/{steps}  {reading}")
    finding = combine(per_family)

    # Added 2026-09-09 AFTER the reading, disclosed in the protocol's errata:
    # the residual decomposed by cause, the decoder's intervention counts,
    # and the decoding-path confound between the arms (the comparator cells
    # were decoded by model.generate, which applies each checkpoint's
    # generation_config defaults -- Qwen2.5-0.5B's repetition_penalty 1.1
    # even in greedy mode -- while the constrained decoder is plain top-1
    # over raw logits). The frozen readings above do not read any of this.
    post_hoc = {}
    for family, (model_id, dtype_name, comparator) in CELLS.items():
        cell = json.loads((CELLS_DIR / f"{family}_schema_constrained.json").read_text())
        comp = json.loads(comparator.read_text())
        acct = cell["per_document_decode"]
        inv_by_stop, regress_by_stop = {}, {}
        for doc_id, text in cell["predictions"].items():
            oc, _ = classify(text); og, _ = classify(comp["predictions"][doc_id])
            if oc is None:
                st = acct[doc_id]["stopped_on"]; inv_by_stop[st] = inv_by_stop.get(st, 0) + 1
                if og is not None:
                    regress_by_stop[st] = regress_by_stop.get(st, 0) + 1
        identical = sum(cell["predictions"][d] == comp["predictions"][d] for d in cell["predictions"])
        identical_after_strip = sum(fc.strip_fence(cell["predictions"][d])[0] == fc.strip_fence(comp["predictions"][d])[0]
                                    for d in cell["predictions"])
        # what happened to each greedy-invalid document, and whether the
        # validator fired on it (a constrained step) -- so a removal can be
        # attributed to the constraint or to the decoding path
        outcomes = {}
        for doc_id, text in cell["predictions"].items():
            og, _ = classify(comp["predictions"][doc_id])
            if og is None:
                oc, _ = classify(text)
                key = ("became_valid" if oc is not None else "still_invalid") + (
                    "_with_validator_step" if acct[doc_id]["constrained_steps"] > 0 else "_without_validator_step")
                outcomes[key] = outcomes.get(key, 0) + 1
        docs_with_steps = sorted(d for d, a in acct.items() if a["constrained_steps"] > 0)
        gen_defaults = {}
        try:
            from huggingface_hub import hf_hub_download
            gen_defaults = {k: v for k, v in json.loads(pathlib.Path(
                hf_hub_download(model_id, "generation_config.json")).read_text()).items()
                if k in ("repetition_penalty", "do_sample", "temperature", "top_p", "top_k")}
        except Exception as exc:  # noqa: BLE001
            gen_defaults = {"unavailable": str(exc)[:80]}
        post_hoc[family] = {
            "constrained_invalid_by_stop_reason": inv_by_stop,
            "regressions_by_stop_reason": regress_by_stop,
            "constrained_steps_total": sum(a["constrained_steps"] for a in acct.values()),
            "fallbacks_total": sum(a["fallbacks"] for a in acct.values()),
            "identical_texts_to_greedy_comparator": identical,
            "identical_after_symmetric_strip": identical_after_strip,
            "greedy_invalid_outcomes": outcomes,
            "documents_where_the_validator_fired": docs_with_steps,
            "comparator_generation_config_defaults_applied_by_model_generate": gen_defaults,
        }
    record = {
        "what": "Addendum V: the JSON-constrained greedy decoder on five families, schema-only, "
                "the first 50 receipts, against each family's banked greedy cell on the same "
                "documents. Both arms classified by the shipped fencecheck strip and the "
                "fail-closed parse. Readings applied by arithmetic from the frozen thresholds.",
        "preregistration": "docs/research/ADDENDUM_V_PROTOCOL.md (frozen 2026-09-08, commit 379346f, before any cell ran)",
        "prediction_written_before_the_run": "REMOVES on Qwen2.5-0.5B; at least REDUCES on the other four; fallbacks under 5% of constrained steps",
        "rows": rows, "reading_per_family": per_family, "the_finding": finding,
        "post_hoc_added_2026-09-09": {
            "why": "added after the reading: every constrained-invalid output is decomposed by the "
                   "decoder's stop reason; the decoder's intervention counts show whether the prefix "
                   "validator ever fired; identical-text counts and the comparator's generation "
                   "defaults expose a decoding-path confound the protocol did not anticipate. The "
                   "frozen readings are untouched.",
            "per_family": post_hoc,
        },
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\n{finding}\nbanked: {OUT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cell", choices=sorted(CELLS))
    ap.add_argument("--read", action="store_true")
    a = ap.parse_args()
    if a.read:
        return read()
    if not a.cell:
        ap.error("--cell or --read")
    return run_cell(a.cell)


if __name__ == "__main__":
    sys.exit(main())
