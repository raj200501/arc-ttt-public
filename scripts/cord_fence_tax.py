#!/usr/bin/env python3
"""Addendum S: does the fence tax replicate on a corpus somebody else built?

Preregistration: the Addendum S row in VERDICT.md (2026-08-25). Protocol
gaps fixed before any data: docs/research/ADDENDUM_S_PROTOCOL.md
(2026-09-01). The carried quantity is the FENCE RATE per arm, classified
by the SHIPPED tools/fencecheck.py; scores are banked for context only
and no cross-corpus score comparison is made.

    PYTHONPATH=src python3 scripts/cord_fence_tax.py --cell 0.5b:schema
    PYTHONPATH=src python3 scripts/cord_fence_tax.py --cell 0.5b:kshot
    PYTHONPATH=src python3 scripts/cord_fence_tax.py --cell 1.5b:schema
    PYTHONPATH=src python3 scripts/cord_fence_tax.py --cell 1.5b:kshot
    PYTHONPATH=src python3 scripts/cord_fence_tax.py --read

Cells are per-document checkpointed and bank raw text incrementally;
--read assembles the frozen artifact ONLY when all four cells exist and
applies the preregistered readings by arithmetic — never re-derived
from the shape of the data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

SPLIT_DIR = REPO / "experiments" / "ladder_e6_cord_split"
CELLS_DIR = REPO / "experiments" / "cord_fence_tax_cells"
WORK = REPO / "work" / "s"
OUT = REPO / "experiments" / "cord_fence_tax_2026-08-25.json"

# Addendum T (docs/research/ADDENDUM_T_PROTOCOL.md, frozen 2026-09-03):
# the same cells on four other families. --addendum T swaps the model
# table, the cell directory, the artifact, and the combination rule.
FAMILIES = {"smollm2-1.7b": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "float32"),
            "granite-2b": ("ibm-granite/granite-3.1-2b-instruct", "float32"),
            "phi3-mini": ("microsoft/Phi-3-mini-4k-instruct", "bfloat16"),
            "falcon3-1b": ("tiiuae/Falcon3-1B-Instruct", "float32")}
FAMILIES_CELLS_DIR = REPO / "experiments" / "cord_fence_tax_families_cells"
FAMILIES_OUT = REPO / "experiments" / "cord_fence_tax_families_2026-09-03.json"
ADDENDUM = "S"  # set from the CLI; "T" selects the family table
# The artifact date stamp for Addendum S, whose cells ran 2026-09-01. It is a
# constant, so the Addendum T cells (run 2026-09-03 to 09-05) also carry it;
# the reader now banks a `run_date_note` saying so, and the driver log and
# commit times carry the true dates. Disclosed in the T VERDICT row.
RUN_DATE = "2026-09-01"

RUNGS = {"0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
         "1.5b": "Qwen/Qwen2.5-1.5B-Instruct"}
K = 20
MAX_NEW_TOKENS = 512
MAX_SEQ = 8192

# Asserted against a fresh derivation from the corpus at run time.
VOCAB = {
    "menu": ["cnt", "discountprice", "nm", "num", "price", "sub",
             "unitprice", "vatyn"],
    "sub_total": ["discount_price", "etc", "service_price",
                  "subtotal_price", "tax_price"],
    "total": ["cashprice", "changeprice", "creditcardprice", "emoneyprice",
              "menuqty_cnt", "menutype_cnt", "total_etc", "total_price"],
}

SCHEMA_INSTRUCTION = (
    "Extract the receipt fields from the document and return them as a\n"
    "single JSON object. Use only these keys:\n"
    "- menu: a list of objects; keys among: " + ", ".join(VOCAB["menu"]) + "\n"
    "- sub_total: an object; keys among: " + ", ".join(VOCAB["sub_total"]) + "\n"
    "- total: an object; keys among: " + ", ".join(VOCAB["total"]) + "\n"
    "Omit any key the document does not support.")


def _fc():
    spec = importlib.util.spec_from_file_location(
        "fencecheck", REPO / "tools" / "fencecheck.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assert_vocab() -> None:
    import collections
    rows = [json.loads(l) for l in
            (REPO / "demo" / "cord_validation.jsonl").read_text(
                encoding="utf-8").splitlines() if l.strip()]
    derived: dict[str, set] = collections.defaultdict(set)
    for r in rows:
        for sc in ("menu", "void_menu", "sub_total", "void_total", "total"):
            v = r["gt_parse"].get(sc)
            if v is None:
                continue
            for it in (v if isinstance(v, list) else [v]):
                if isinstance(it, dict):
                    derived[sc].update(it)
    got = {sc: sorted(ks) for sc, ks in derived.items()}
    if got != {sc: ks for sc, ks in VOCAB.items()}:
        raise SystemExit(f"schema vocabulary drifted from the corpus: "
                         f"{got} != {VOCAB}; the protocol note and this "
                         "runner must agree before any arm runs")


def _model_of(rung: str) -> tuple[str, str]:
    if ADDENDUM == "T":
        return FAMILIES[rung]
    return RUNGS[rung], "float32"


def _cells_dir() -> pathlib.Path:
    return FAMILIES_CELLS_DIR if ADDENDUM == "T" else CELLS_DIR


def run_cell(rung: str, regime: str) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from arcttt.model import TTTConfig
    from arcttt.text_ttt import TextPredictor, text_task_to_messages
    from run_challenge import build_task

    _assert_vocab()
    cell_path = _cells_dir() / f"{rung}_{regime}.json"
    if cell_path.exists():
        print(f"cell banked already: {cell_path.name}")
        return 0
    _cells_dir().mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((SPLIT_DIR / "manifest.json").read_text())
    train = _read_jsonl(SPLIT_DIR / "train.jsonl")
    if regime == "kshot":
        holdout = _read_jsonl(SPLIT_DIR / "holdout.jsonl")
    else:
        # Schema arm: all 100 receipts, the row's letter. Rendered through
        # the same corpus adapter as everything else.
        from arcttt.text_task import from_cord_gt
        rows = [json.loads(l) for l in
                (REPO / "demo" / "cord_validation.jsonl").read_text(
                    encoding="utf-8").splitlines() if l.strip()]
        task_all = from_cord_gt(rows[:1], rows, task_id="s-schema")
        holdout = [{"id": f"cord-{i:03d}", "text": task_all.test[i].input_text}
                   for i in range(len(rows))]
    task = build_task(train, holdout)

    model_id, dtype_name = _model_of(rung)
    config_key = (f"{model_id}|{regime}|{dtype_name}|k={K if regime=='kshot' else 0}"
                  f"|mnt={MAX_NEW_TOKENS}|seq={MAX_SEQ}|S")
    ckpt_path = WORK / f"{rung}_{regime}.ckpt.jsonl"
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
                raise SystemExit(f"checkpoint {ckpt_path} belongs to "
                                 f"{row.get('config')!r}; refusing")
            done[row["id"]] = row
    if done:
        print(f"[S:{rung}:{regime}] resuming {len(done)}/{len(holdout)}",
              flush=True)

    torch.set_num_threads(4)
    torch.manual_seed(1)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 dtype=getattr(torch, dtype_name))
    model.eval()
    config = TTTConfig(max_new_tokens=MAX_NEW_TOKENS,
                       max_sequence_tokens=MAX_SEQ)
    predictor = TextPredictor(model, tokenizer, config, torch.device("cpu"))

    predictions: dict[str, str] = {}
    seconds: dict[str, float] = {}
    resumed = len(done)
    with open(ckpt_path, "a", encoding="utf-8") as ckpt:
        for i, row in enumerate(holdout):
            if row["id"] in done:
                predictions[row["id"]] = done[row["id"]]["raw"]
                seconds[row["id"]] = done[row["id"]]["seconds"]
                continue
            if regime == "kshot":
                ids = predictor._prompt_ids(
                    text_task_to_messages(task, i, include_demos=True))
            else:
                text_prompt = tokenizer.apply_chat_template(
                    [{"role": "user",
                      "content": SCHEMA_INSTRUCTION + "\n\n" + row["text"]}],
                    tokenize=False, add_generation_prompt=True)
                enc = tokenizer(text_prompt, return_tensors="pt").input_ids
                ids = enc if enc.shape[1] <= MAX_SEQ else None
            if ids is None:
                raise SystemExit(f"prompt {row['id']} exceeded {MAX_SEQ}")
            began = time.monotonic()
            with torch.no_grad():
                out = model.generate(input_ids=ids,
                                     attention_mask=torch.ones_like(ids),
                                     max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
            took = round(time.monotonic() - began, 1)
            text = tokenizer.decode(out[0][ids.shape[1]:],
                                    skip_special_tokens=True).strip()
            predictions[row["id"]] = text
            seconds[row["id"]] = took
            ckpt.write(json.dumps({"config": config_key, "id": row["id"],
                                   "raw": text, "seconds": took},
                                  ensure_ascii=False) + "\n")
            ckpt.flush()
            print(f"[S:{rung}:{regime}] {row['id']} {i + 1}/{len(holdout)} "
                  f"{took}s", flush=True)

    record = {
        "what": f"Addendum {ADDENDUM} cell: {model_id}, {regime} regime, raw "
                "CORD outputs. Rates and readings live in the assembled "
                "artifact (--read), applied to all cells symmetrically.",
        "protocol": ("docs/research/ADDENDUM_T_PROTOCOL.md" if ADDENDUM == "T" else "docs/research/ADDENDUM_S_PROTOCOL.md"),
        "run_date": RUN_DATE,
        "model": model_id,
        "dtype": dtype_name,
        "regime": regime,
        "n": len(holdout),
        "k": K if regime == "kshot" else 0,
        "schema_instruction": (SCHEMA_INSTRUCTION if regime == "schema"
                               else None),
        "demos": ("experiments/ladder_e6_cord_split/train.jsonl"
                  if regime == "kshot" else None),
        "split_manifest_sha256": manifest["files"],
        "decode": f"greedy, max_new_tokens={MAX_NEW_TOKENS}, {dtype_name}, CPU",
        "resumed_from_checkpoint": resumed,
        "per_document_seconds": seconds,
        "predictions": predictions,
    }
    tmp = cell_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    tmp.replace(cell_path)
    ckpt_path.unlink(missing_ok=True)
    print(f"banked: {cell_path.name} ({len(predictions)} raw outputs)")
    return 0


def family_reading(f_schema: float, f_kshot: float) -> str:
    """Per-model reading, frozen in ADDENDUM_S_PROTOCOL.md / ADDENDUM_T_PROTOCOL.md:
    (a) f_schema >= 0.50 and f_kshot <= 0.10; (b) f_schema >= 0.50 and
    f_kshot > 0.10; (c) f_schema < 0.50. Boundaries are inclusive exactly
    as the frozen text writes them."""
    if f_schema < 0.50:
        return "(c) DOES NOT REPLICATE"
    if f_kshot <= 0.10:
        return "(a) REPLICATES"
    return "(b) PARTIAL"


def combine_families(per_model: dict) -> str:
    """Addendum T combination rule, frozen 2026-09-03: HOLDS only if (a)
    in >= 3 of the families AND no (c); any (c) is a named exception at
    full size; anything else is MIXED with no headline."""
    n_a = sum(v.startswith("(a)") for v in per_model.values())
    c_fams = [r for r, v in per_model.items() if v.startswith("(c)")]
    if c_fams:
        return ("(c) IN " + ", ".join(c_fams) + ": named exception(s) at full "
                "size; the outbound sentence becomes 'on N of the families "
                "tested' with the exception named — never 'across model "
                "families'. Per-family: " + json.dumps(per_model))
    if n_a >= 3:
        return (f"HOLDS ACROSS FAMILIES: (a) in {n_a} of {len(per_model)} families "
                "and no (c) — schema-only prompts get wrapped and k-shot prompts "
                "do not, on public receipts, across organisations' checkpoints. "
                "Per-family: " + json.dumps(per_model))
    return ("MIXED: fewer than 3 families at (a), none at (c) — all rates "
            "publish, no headline. " + json.dumps(per_model))


def combine_sizes(per_model: dict) -> str:
    """Addendum S combination rule (two Qwen sizes), unchanged."""
    if any(v.startswith("(c)") for v in per_model.values()):
        return ("(c) AT " + " AND ".join(
            r for r, v in per_model.items() if v.startswith("(c)")) +
            ": the fence pattern is corpus-specific at that size and the "
            "full re-scoping commitment fires — every statement of the "
            "finding is re-scoped to the waybill corpus in the same "
            "commit that publishes this artifact.")
    if all(v.startswith("(a)") for v in per_model.values()):
        return ("(a) IT REPLICATES AT BOTH SIZES: schema-only prompts "
                "get wrapped and k-shot prompts do not, on public "
                "receipts nobody here authored. The headline moves "
                "onto public data.")
    return ("MIXED/(b): " + json.dumps(per_model) + " — all four "
            "rates publish, no headline movement.")


def read() -> int:
    from arcttt.scoring import field_micro_f1, parse_json_object
    from arcttt.text_task import TextTaskFormatError

    fc = _fc()
    cells = {}
    missing = []
    table = FAMILIES if ADDENDUM == "T" else RUNGS
    for rung in table:
        for regime in ("schema", "kshot"):
            path = _cells_dir() / f"{rung}_{regime}.json"
            if path.exists():
                cells[(rung, regime)] = json.loads(path.read_text())
            else:
                missing.append(path.name)
    if missing:
        print("WITHHELD: missing cells " + ", ".join(missing) +
              " — the reading is stated once every preregistered cell "
              "exists, not from the subset that finished first.")
        return 1

    gold = {json.loads(l)["id"]: json.loads(l)["gold"] for l in
            (SPLIT_DIR / "gold.jsonl").read_text().splitlines() if l.strip()}
    # Schema cells cover all 100; gold for the 20 train receipts comes
    # from the split's train file (same rendering, same canonicalizer).
    for r in _read_jsonl(SPLIT_DIR / "train.jsonl"):
        gold[r["id"]] = r["gold"]

    rows = []
    rates = {}
    for (rung, regime), rec in sorted(cells.items()):
        fenced = invalid = 0
        f1s = []
        for doc_id, text in rec["predictions"].items():
            fenced += fc.strip_fence(text)[1]
            try:
                obj = parse_json_object(fc.strip_fence(text)[0])
            except TextTaskFormatError:
                obj = None
            if obj is None:
                invalid += 1
                f1s.append(0.0)
            else:
                f1s.append(field_micro_f1(obj, gold[doc_id]))
        n = len(rec["predictions"])
        rate = round(fenced / n, 4)
        rates[(rung, regime)] = rate
        rows.append({
            "model": rec["model"], "regime": regime, "n": n,
            "fenced": fenced, "fence_rate": rate, "invalid": invalid,
            "mean_micro_f1_context_only": round(sum(f1s) / n, 4),
        })
        print(f"{rung:5s} {regime:6s} fenced {fenced:3d}/{n:<3d} "
              f"rate {rate:.4f}  invalid {invalid}")

    # The FROZEN readings, per model, combined per the protocol note's
    # non-flattering rule — arithmetic only (pure functions below, unit-
    # tested at the frozen boundaries in tests/test_cord_fence_tax_readings.py).
    per_model = {rung: family_reading(rates[(rung, "schema")],
                                      rates[(rung, "kshot")])
                 for rung in table}
    finding = (combine_families(per_model) if ADDENDUM == "T"
               else combine_sizes(per_model))

    record = {
        "what": (f"Addendum {ADDENDUM}: fence rates on CORD, the shipped tool as "
                 "classifier, readings applied by arithmetic from the "
                 "frozen row and the dated protocol note."),
        "preregistration": "VERDICT.md Addendum S row (2026-08-25); "
                           "protocol gaps fixed in "
                           "docs/research/ADDENDUM_S_PROTOCOL.md "
                           "(2026-09-01) before any cell ran",
        "run_date": RUN_DATE,
        "run_date_note": ("RUN_DATE is the Addendum S stamp; the Addendum T cells ran 2026-09-03 to "
                          "2026-09-05 (driver log work/t_driver.log, commit times). The cell records "
                          "banked before 2026-09-05 also carry the S protocol path and, for the "
                          "bfloat16 Phi-3 cells, a decode string that says float32; the dtype field "
                          "is the true one. Runner constants, fixed 2026-09-05; the cells are not "
                          "re-run." if ADDENDUM == "T" else "the S cells ran on this date"),
        "classified_by": "tools/fencecheck.py strip_fence(), leading-"
                         "fence scope (undercount property documented in "
                         "Addendum R's erratum)",
        "rows": rows,
        "reading_per_model": per_model,
        "the_finding": finding,
        "note_on_n": "schema cells n=100 (the row's letter); k-shot "
                     "cells n=80 — demonstrations are the E6 train-20 "
                     "and a receipt cannot demonstrate itself; deviation "
                     "declared in the protocol note before data",
    }
    out = FAMILIES_OUT if ADDENDUM == "T" else OUT
    record["addendum"] = ADDENDUM
    if ADDENDUM == "T":
        record["preregistration"] = "docs/research/ADDENDUM_T_PROTOCOL.md (frozen 2026-09-03 before any arm)"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\n{finding}\nbanked: {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", help="rung:regime, e.g. 0.5b:schema (S) or smollm2-1.7b:schema (T)")
    parser.add_argument("--read", action="store_true")
    parser.add_argument("--addendum", default="S", choices=("S", "T"),
                        help="T = the family replication (ADDENDUM_T_PROTOCOL.md)")
    args = parser.parse_args()
    global ADDENDUM
    ADDENDUM = args.addendum
    if args.read:
        return read()
    if not args.cell:
        raise SystemExit("pass --cell rung:regime or --read")
    rung, regime = args.cell.split(":")
    table = FAMILIES if ADDENDUM == "T" else RUNGS
    if rung not in table or regime not in ("schema", "kshot"):
        raise SystemExit(f"unknown cell {args.cell}")
    return run_cell(rung, regime)


if __name__ == "__main__":
    raise SystemExit(main())
