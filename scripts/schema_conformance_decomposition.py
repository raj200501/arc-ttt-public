#!/usr/bin/env python3
"""Does constrained decoding capture the HEADLINE gates too? Measured.

`format_counterfactual.py` showed that on the realistic waybill corpus the
adaptation advantage is mostly output-format reliability, and that a
JSON-grammar-constrained decoder is the better-supported explanation there.
The immediate follow-up question -- the one that decides whether there is a
company -- is whether the same objection reaches gates 1, 4 and 5, where
the headline numbers live.

It has two parts, and both are answered here from banked artifacts:

**Part 1: format.** Restrict each gate to the documents where BOTH arms
emitted valid JSON, so output-format failure cannot be the explanation, and
recompute the paired delta. (Gate 1's arms store per-document `valid_json`
but no predictions; that is enough for part 1 and not for part 2.)

**Part 2: schema conformance.** Valid JSON is not the same as the RIGHT
KEYS. A 30-shot baseline can emit perfectly parseable JSON in a schema it
invented and score near zero, and a schema-constrained decoder would fix
exactly that at no cost. So for every gate arm that stores predictions,
the baseline is given the free win: rebuild its output under the gold key
paths, keeping its own values, and re-score.

    R1  path-repaired   keep the baseline's value wherever its leaf PATH
                        matches gold; drop invented keys, omit the rest.
                        This is what a schema-constrained decoder
                        guarantees by construction.
    R2  name-repaired   additionally, a value stored under the wrong
                        parent counts if its LEAF NAME matches -- so
                        nesting mistakes are forgiven too. Strictly more
                        generous than R1 and than any real decoder.

If the repaired baseline closes most of the gap, the headline is schema
conformance and constrained decoding is a serious rival for it. If it does
not, the gap is value-level extraction, which no decoder can supply.

**Post-hoc analysis of banked data. Not a preregistered gate**, and
labeled so in the artifact. Gold is regenerated from the deterministic
generator, not read from any stored copy.

    PYTHONPATH=src python3 scripts/schema_conformance_decomposition.py
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import Counter

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

GATE1 = [(f"novel_schema_0.5b_k30_seed{s}_kshot_2026-08-12.json",
          f"novel_schema_0.5b_k30_seed{s}_adapted_2026-08-12.json", str(s))
         for s in (1, 2, 3)]
GATE4 = [(f"novel_schema_0.5b_k30_seed{s}_kshot_2026-08-12.json",
          f"novel_schema_f_0.5b_k30_seed{s}_docadapted_2026-08-19.json", str(s))
         for s in (1, 2, 3)]
GATE5 = [(f"novel_schema_e_0.5b_k30_seed{s}_kshot_2026-08-19.json",
          f"novel_schema_e_0.5b_k30_seed{s}_adapted_2026-08-19.json", str(s))
         for s in (203, 204, 206, 207, 208, 209)]


def load(name: str) -> dict:
    return json.loads((REPO / "experiments" / name).read_text(encoding="utf-8"))


def rows(record: dict) -> dict:
    return {r["index"]: r for r in record["results"] if "micro_f1" in r}


def regenerate_task(record: dict, seed: int):
    """Gold from the generator, geometry resolved by stored schema text."""
    from arcttt.novel_schema import make_task

    stored = record.get("schema")
    for geometry in ("fixed", "diverse", "diverse-compact"):
        task, schema = make_task(seed=seed, n_train=record["k"],
                                 n_test=record["eval_n"],
                                 task_id="decompose", geometry=geometry)
        if stored is None or schema.describe() == stored:
            return task
    raise SystemExit(f"schema does not regenerate for seed {seed}")


def leaf_paths(value: object, path: str = "") -> dict[str, object]:
    """Leaf path -> raw value. Mirrors scoring.field_pairs' path rule."""
    out: dict[str, object] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(leaf_paths(item, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, list):
        for item in value:
            out.update(leaf_paths(item, path))
    else:
        out[path] = value
    return out


def repair(pred: dict, gold: dict, by_name: bool) -> dict:
    """Rebuild `pred` under gold's key paths, keeping pred's own values.

    Returns a FLAT path->value dict; scoring only sees leaf paths, so a
    flat rebuild scores identically to the nested one it stands for.
    """
    gold_leaves = leaf_paths(gold)
    pred_leaves = leaf_paths(pred)
    if by_name:
        by_leaf: dict[str, list[object]] = {}
        for path, val in pred_leaves.items():
            by_leaf.setdefault(path.rsplit(".", 1)[-1], []).append(val)
    out: dict[str, object] = {}
    for gpath in gold_leaves:
        if gpath in pred_leaves:
            out[gpath] = pred_leaves[gpath]
        elif by_name:
            candidates = by_leaf.get(gpath.rsplit(".", 1)[-1])
            if candidates:
                out[gpath] = candidates[0]
    return out


def micro_f1_flat(pred_flat: dict, gold: dict) -> float:
    from arcttt.scoring import field_pairs, normalize_value

    pred_pairs: Counter = Counter()
    for path, val in pred_flat.items():
        pred_pairs[(path, normalize_value(val))] += 1
    gold_pairs = field_pairs(gold)
    if not pred_pairs and not gold_pairs:
        return 1.0
    overlap = sum((pred_pairs & gold_pairs).values())
    denom = sum(pred_pairs.values()) + sum(gold_pairs.values())
    return 2.0 * overlap / denom if denom else 0.0


def sign_counts(deltas: list[float]) -> tuple[int, int, int]:
    w = sum(1 for d in deltas if d > 1e-12)
    lo = sum(1 for d in deltas if d < -1e-12)
    return w, lo, len(deltas) - w - lo


def analyse_pair(base_name: str, adapt_name: str, label: str) -> dict:
    from arcttt.scoring import parse_json_object

    base_rec, adapt_rec = load(base_name), load(adapt_name)
    rb, ra = rows(base_rec), rows(adapt_rec)
    ids = sorted(set(rb) & set(ra))
    valid = lambda r: r.get("valid_json", True)  # noqa: E731

    both_valid = [i for i in ids if valid(rb[i]) and valid(ra[i])]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731

    delta_all = mean([ra[i]["micro_f1"] - rb[i]["micro_f1"] for i in ids])
    delta_fmt = mean([ra[i]["micro_f1"] - rb[i]["micro_f1"] for i in both_valid])

    out = {
        "arm": label,
        "n_pairs": len(ids),
        "invalid_json_baseline": sum(1 for i in ids if not valid(rb[i])),
        "invalid_json_adapted": sum(1 for i in ids if not valid(ra[i])),
        "delta_as_measured": round(delta_all, 4),
        "n_format_neutral": len(both_valid),
        "delta_format_neutral": round(delta_fmt, 4),
        "schema_repair": None,
    }

    if not any(r.get("prediction") for r in rb.values()):
        out["schema_repair_note"] = (
            "baseline arm stores no predictions (pre-Addendum-E artifact); "
            "part 2 not computable for this arm")
        return out

    seed = base_rec.get("seed") or int(label)
    task = regenerate_task(base_rec, int(seed))
    gold_by_index = {i: parse_json_object(task.test[i].output_text)
                     for i in ids}

    base_raw, base_r1, base_r2, adapt_raw = [], [], [], []
    gold_leaves = pred_leaves = shared = 0
    exact_key_set = wrong_value = 0
    for i in ids:
        gold = gold_by_index[i]
        adapt_raw.append(ra[i]["micro_f1"])
        base_raw.append(rb[i]["micro_f1"])
        text = rb[i].get("prediction")
        try:
            pred = parse_json_object(text) if text else None
        except Exception:
            pred = None
        if pred is None:
            base_r1.append(0.0)
            base_r2.append(0.0)
            gold_leaves += len(leaf_paths(gold))
            continue
        base_r1.append(micro_f1_flat(repair(pred, gold, by_name=False), gold))
        base_r2.append(micro_f1_flat(repair(pred, gold, by_name=True), gold))
        gl, pl = leaf_paths(gold), leaf_paths(pred)
        gp, pp = set(gl), set(pl)
        gold_leaves += len(gp)
        pred_leaves += len(pp)
        shared += len(gp & pp)
        exact_key_set += int(gp == pp)
        from arcttt.scoring import normalize_value
        wrong_value += sum(1 for k in gp & pp
                           if normalize_value(gl[k]) != normalize_value(pl[k]))

    d1 = [a - b for a, b in zip(adapt_raw, base_r1)]
    d2 = [a - b for a, b in zip(adapt_raw, base_r2)]
    gap = mean(adapt_raw) - mean(base_raw)
    out["schema_repair"] = {
        "baseline_raw": round(mean(base_raw), 4),
        "baseline_path_repaired_R1": round(mean(base_r1), 4),
        "baseline_name_repaired_R2": round(mean(base_r2), 4),
        "adapted": round(mean(adapt_raw), 4),
        "delta_vs_R1": round(mean(d1), 4),
        "delta_vs_R2": round(mean(d2), 4),
        "sign_vs_R2": dict(zip(("wins", "losses", "ties"), sign_counts(d2))),
        "share_of_gap_closed_by_R2": round(
            (mean(base_r2) - mean(base_raw)) / gap, 4) if abs(gap) > 1e-9
        else None,
    }
    out["baseline_schema_conformance"] = {
        "key_path_recall": round(shared / gold_leaves, 4) if gold_leaves else None,
        "key_path_precision": round(shared / pred_leaves, 4) if pred_leaves else None,
        "documents_with_exactly_the_gold_key_set": f"{exact_key_set}/{len(ids)}",
        "wrong_value_share_of_shared_leaves": round(wrong_value / shared, 4)
        if shared else None,
        "reading": "what the baseline gets wrong: KEYS if recall/precision "
                   "are below 1, VALUES if they are at 1 and the wrong-value "
                   "share is high. Only the first is reachable by a "
                   "schema-constrained decoder.",
    }
    return out


def summarise(name: str, arms: list[dict]) -> dict:
    mean = lambda k: sum(a[k] for a in arms) / len(arms)  # noqa: E731
    block = {
        "gate": name,
        "seed_mean_delta_as_measured": round(mean("delta_as_measured"), 4),
        "seed_mean_delta_format_neutral": round(mean("delta_format_neutral"), 4),
        "total_invalid_json_baseline": sum(a["invalid_json_baseline"]
                                           for a in arms),
        "total_invalid_json_adapted": sum(a["invalid_json_adapted"]
                                          for a in arms),
        "arms": arms,
    }
    repaired = [a for a in arms if a["schema_repair"]]
    if repaired:
        rm = lambda k: sum(a["schema_repair"][k] for a in repaired) / len(  # noqa: E731
            repaired)
        block["seed_mean_baseline_raw"] = round(rm("baseline_raw"), 4)
        block["seed_mean_baseline_path_repaired_R1"] = round(
            rm("baseline_path_repaired_R1"), 4)
        block["seed_mean_baseline_name_repaired_R2"] = round(
            rm("baseline_name_repaired_R2"), 4)
        block["seed_mean_adapted"] = round(rm("adapted"), 4)
        block["seed_mean_delta_vs_R1"] = round(rm("delta_vs_R1"), 4)
        block["seed_mean_delta_vs_R2"] = round(rm("delta_vs_R2"), 4)
        block["share_of_gap_closed_by_R2"] = round(
            rm("share_of_gap_closed_by_R2"), 4)
    return block


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(
        REPO / "experiments" / "schema_conformance_decomposition_2026-08-22.json"))
    args = parser.parse_args()

    gates = [
        ("gate1_addendum_B_k30", GATE1),
        ("gate4_addendum_F_document_mode", GATE4),
        ("gate5_addendum_E_r2_diverse", GATE5),
    ]
    blocks = [summarise(name, [analyse_pair(b, a, lab) for b, a, lab in arms])
              for name, arms in gates]

    record = {
        "what": "Does the constrained-decoding objection reach the HEADLINE "
                "gates? Two parts: format-neutral restriction, and an "
                "oracle schema repair of the baseline arm.",
        "status": "POST-HOC ANALYSIS of banked artifacts. NOT a "
                  "preregistered gate. Gold is regenerated from the "
                  "deterministic generator; no stored gold is trusted.",
        "repairs": {
            "R1_path_repaired": "baseline rebuilt under gold's leaf paths, "
                                "keeping its own values; invented keys "
                                "dropped. What a schema-constrained decoder "
                                "guarantees by construction.",
            "R2_name_repaired": "R1, plus values under the wrong parent "
                                "counted when the leaf NAME matches. "
                                "Strictly more generous than any real "
                                "decoder.",
        },
        "gates": blocks,
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")

    for b in blocks:
        print(f"\n{b['gate']}")
        print(f"  invalid JSON: baseline {b['total_invalid_json_baseline']}, "
              f"adapted {b['total_invalid_json_adapted']}")
        print(f"  delta as measured    : {b['seed_mean_delta_as_measured']:+.4f}")
        print(f"  delta format-neutral : "
              f"{b['seed_mean_delta_format_neutral']:+.4f}")
        if "seed_mean_delta_vs_R2" in b:
            print(f"  baseline raw {b['seed_mean_baseline_raw']:.4f} -> "
                  f"path-repaired {b['seed_mean_baseline_path_repaired_R1']:.4f}"
                  f" -> name-repaired {b['seed_mean_baseline_name_repaired_R2']:.4f}"
                  f"   (adapted {b['seed_mean_adapted']:.4f})")
            print(f"  delta vs R1 {b['seed_mean_delta_vs_R1']:+.4f}   "
                  f"delta vs R2 {b['seed_mean_delta_vs_R2']:+.4f}   "
                  f"share of gap closed by R2: "
                  f"{b['share_of_gap_closed_by_R2']:.1%}")
        else:
            note = b["arms"][0].get("schema_repair_note", "")
            print(f"  schema repair: {note}")
    print(f"\nbanked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
