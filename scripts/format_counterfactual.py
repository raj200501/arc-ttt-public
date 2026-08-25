#!/usr/bin/env python3
"""Constrained decoding is the rival explanation. This measures it.

The rehearsal's paired baseline (`blind_rehearsal_baseline_2026-08-21.json`)
records that **63% of the delta comes from 3 documents where the prompted
arm emitted invalid JSON**, and concludes the measured benefit on
document-shaped data is "output-format reliability, not extraction
accuracy".

That conclusion has an obvious and cheap rival: *a JSON-grammar-constrained
decoder gets format reliability for free, with no adaptation, no training
and no per-tenant weights.* If the delta is format, a constrained decoder
eats the product. Nothing in this repository had named that explanation,
let alone bounded it. This script bounds it, from banked data, stdlib-only.

**This is a post-hoc analysis of already-collected data, not a
preregistered gate.** No bar was frozen for it before the numbers existed.
It is reported as an upper bound on a confound, and the two-statistics rule
is applied to each reading only so the readings are commensurable with the
published one -- a PASS here is not a gate PASS.

Six readings of the same 30 paired documents:

  A  as-measured          the published result, reproduced here as control
  B  schema-key pruning   prompted arm's extra keys removed (a schema-
                          constrained decoder cannot emit an off-schema key)
  C  B + nulls imputed    the 3 unparseable prompted outputs replaced by
                          that arm's OWN mean on the documents it did parse
                          -- the realistic constrained-decoder proxy
  D  B + nulls at 1.0     the same three replaced by a PERFECT extraction --
                          an impossible fixer, included as a hard ceiling
  E  format-neutral,      only the documents both arms parsed, but STILL
     key-pruned           key-pruned -- so still a grant
  F  format-neutral,      the arms exactly as decoded, on the documents both
     nothing granted      of them parsed. Nothing assumed at all.

C imputes and D is a fiction; **F is the one that assumes nothing**, and it
is the number to quote. B and D bracket it.

**Correction, 2026-08-22:** reading E was originally labeled "assumes
NOTHING" while being computed from B's key-pruned baseline. An outside
auditor found the mislabel. E is kept with an honest description and F was
added; F agrees with the +4.14 `VERDICT.md` already carried for this
subset, which is how the mislabel should have been caught here.

    python3 scripts/format_counterfactual.py
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

RAW = REPO / "experiments" / "blind_rehearsal_2026-08-20_raw"
BAR = 0.05  # the rehearsal's frozen bar, reused so readings are comparable


def load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_one(prediction: object, gold_obj: dict) -> float:
    from arcttt.text_ttt import score_text_output

    if prediction is None:
        return 0.0
    score = score_text_output(json.dumps(prediction), json.dumps(gold_obj))
    return score.micro_f1 if score.valid_json else 0.0


def sign_test(deltas: list[float]) -> dict:
    wins = sum(1 for d in deltas if d > 1e-12)
    losses = sum(1 for d in deltas if d < -1e-12)
    ties = len(deltas) - wins - losses
    n = wins + losses
    p = 1.0
    if n:
        p = sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)
    return {"wins": wins, "losses": losses, "ties": ties,
            "p_value": round(p, 6)}


def reading(name: str, note: str, base: dict, adapt: dict) -> dict:
    from novel_schema_summary import t95  # the one estimator in this repo

    ids = sorted(set(base) & set(adapt))
    deltas = [adapt[i] - base[i] for i in ids]
    mean_base = sum(base[i] for i in ids) / len(ids)
    mean_adapt = sum(adapt[i] for i in ids) / len(ids)
    mean_delta = mean_adapt - mean_base
    st = sign_test(deltas)
    sd = math.sqrt(sum((d - mean_delta) ** 2 for d in deltas)
                   / (len(deltas) - 1))
    half = t95(len(deltas) - 1) * sd / math.sqrt(len(deltas))
    clears = mean_delta >= BAR
    agrees = st["p_value"] < 0.05 and st["wins"] > st["losses"]
    return {
        "reading": name,
        "what_it_assumes": note,
        "n": len(ids),
        "baseline_mean": round(mean_base, 4),
        "adapted_mean": round(mean_adapt, 4),
        "mean_delta": round(mean_delta, 4),
        "ci95": [round(mean_delta - half, 4), round(mean_delta + half, 4)],
        "sign_test": st,
        "two_statistics_verdict": "PASS" if (clears and agrees) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(RAW))
    parser.add_argument("--out", default=str(
        REPO / "experiments" / "format_counterfactual_2026-08-22.json"))
    args = parser.parse_args()

    raw = pathlib.Path(args.raw)
    gold = {r["id"]: r["gold"] for r in load_jsonl(raw / "gold_holdout.jsonl")}
    prompted = {r["id"]: r["prediction"] for r in
                load_jsonl(raw / "predictions_prompted_greedy.jsonl")}
    adapted = {r["id"]: r["prediction"] for r in
               load_jsonl(raw / "predictions_adapted_greedy.jsonl")}

    ids = sorted(gold)
    adapt_scores = {i: score_one(adapted.get(i), gold[i]) for i in ids}

    # --- A: as measured -------------------------------------------------
    base_A = {i: score_one(prompted.get(i), gold[i]) for i in ids}

    unparseable = sorted(i for i in ids if prompted.get(i) is None)

    # --- B: schema-key pruning -----------------------------------------
    # A grammar/schema-constrained decoder emits exactly the schema's keys.
    # Pruning the prompted arm to the gold key set is therefore the free
    # half of what constrained decoding buys, and it costs us nothing to
    # grant it: it can only raise the prompted arm's precision.
    pruned = {}
    off_schema_keys = 0
    for i in ids:
        pred = prompted.get(i)
        if isinstance(pred, dict):
            keep = {k: v for k, v in pred.items() if k in gold[i]}
            off_schema_keys += len(pred) - len(keep)
            pruned[i] = keep
        else:
            pruned[i] = pred
    base_B = {i: score_one(pruned.get(i), gold[i]) for i in ids}

    # --- C: B + unparseable imputed at the arm's own parseable mean -----
    parseable = [i for i in ids if prompted.get(i) is not None]
    own_mean = sum(base_B[i] for i in parseable) / len(parseable)
    base_C = dict(base_B)
    for i in unparseable:
        base_C[i] = own_mean

    # --- D: B + unparseable at a perfect 1.0 (impossible ceiling) -------
    base_D = dict(base_B)
    for i in unparseable:
        base_D[i] = 1.0

    # --- E: format-neutral subset, still key-pruned ---------------------
    base_E = {i: base_B[i] for i in parseable}
    adapt_E = {i: adapt_scores[i] for i in parseable}

    # --- F: format-neutral subset, NOTHING granted ----------------------
    # E restricts to the parseable documents but still hands the baseline
    # the key-pruning of reading B. That is a grant, and calling E
    # "assumption-free" was wrong -- an outside auditor caught it. F is the
    # reading that grants nothing at all: the arms exactly as decoded, on
    # the documents both of them parsed.
    base_F = {i: base_A[i] for i in parseable}
    adapt_F = {i: adapt_scores[i] for i in parseable}

    readings = [
        reading("A_as_measured",
                "nothing -- reproduces the published paired result",
                base_A, adapt_scores),
        reading("B_schema_key_pruned",
                "the prompted arm never emits an off-schema key "
                "(free half of constrained decoding)",
                base_B, adapt_scores),
        reading("C_pruned_plus_imputed",
                "additionally: a constrained decoder turns each unparseable "
                "output into valid JSON of that arm's TYPICAL quality "
                "(imputation, not measurement)",
                base_C, adapt_scores),
        reading("D_pruned_plus_perfect",
                "additionally: the fixer is a perfect extractor on those "
                "documents (impossible; a hard ceiling)",
                base_D, adapt_scores),
        reading("E_format_neutral_subset_key_pruned",
                "restricted to the documents both arms parsed, AND the "
                "key-pruning of reading B. No imputation, but not "
                "assumption-free -- it still grants the free half of "
                "constrained decoding.",
                base_E, adapt_E),
        reading("F_format_neutral_subset_nothing_granted",
                "NOTHING. The arms exactly as decoded, restricted to the "
                "documents both of them parsed. Quote this one.",
                base_F, adapt_F),
    ]

    by_name = {r["reading"]: r for r in readings}
    survives = by_name["F_format_neutral_subset_nothing_granted"]["mean_delta"]
    ceiling = by_name["D_pruned_plus_perfect"]["mean_delta"]

    record = {
        "what": "Upper bound on the constrained-decoding rival explanation "
                "for the rehearsal's paired delta.",
        "status": "POST-HOC ANALYSIS of banked data. NOT a preregistered "
                  "gate: no bar was frozen for it before the numbers "
                  "existed. The two-statistics rule is applied only so "
                  "these readings are commensurable with the published "
                  "one; a PASS here is not a gate PASS.",
        "why_it_exists": "The published conclusion -- that on realistic "
                         "documents the benefit is output-format "
                         "reliability rather than extraction accuracy -- "
                         "invites the reply that a JSON-grammar-constrained "
                         "decoder buys format reliability for free. That "
                         "reply had never been named in this repository. "
                         "It is named here and bounded.",
        "source": str(pathlib.Path(args.raw).relative_to(REPO)),
        "unparseable_prompted_documents": unparseable,
        "off_schema_keys_removed_from_prompted_arm": off_schema_keys,
        "readings": readings,
        "correction": "Until 2026-08-22 the format-neutral reading was "
                      "labeled 'assumes NOTHING' while being computed from "
                      "the key-pruned baseline of reading B. An outside "
                      "auditor found it. The mislabeled reading is kept as "
                      "E with an honest description, and F -- which really "
                      "does grant nothing -- is now the quoted one. F "
                      "agrees with the +4.14 that VERDICT.md already "
                      "carried for this subset.",
        "bottom_line": {
            "format_neutral_delta": survives,
            "hardest_ceiling_delta": ceiling,
            "reading": (
                "On the documents both arms parsed -- where format "
                "reliability cannot be the explanation because neither arm "
                "failed format, and granting nothing else -- the paired "
                "delta is "
                f"{survives:+.4f}. Even granting an impossible fixer that "
                "makes the prompted arm perfect on every document it could "
                f"not parse, the delta is {ceiling:+.4f}."
            ),
        },
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")

    width = max(len(r["reading"]) for r in readings)
    print(f"{'reading'.ljust(width)}  {'n':>3}  {'base':>7}  {'adapt':>7}  "
          f"{'delta':>8}  {'sign':>12}  verdict")
    for r in readings:
        st = r["sign_test"]
        sign = f"{st['wins']}W/{st['losses']}L/{st['ties']}T"
        print(f"{r['reading'].ljust(width)}  {r['n']:>3}  "
              f"{r['baseline_mean']:>7.4f}  {r['adapted_mean']:>7.4f}  "
              f"{r['mean_delta']:>+8.4f}  {sign:>12}  "
              f"{r['two_statistics_verdict']}")
    print()
    print(f"unparseable prompted outputs : {len(unparseable)} "
          f"({', '.join(unparseable) or 'none'})")
    print(f"off-schema keys granted away : {off_schema_keys}")
    print()
    print(record["bottom_line"]["reading"])
    print(f"\nbanked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
