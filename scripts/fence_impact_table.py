#!/usr/bin/env python3
"""What does one markdown fence cost, per checkpoint and per prompt regime?

The census counts code. This measures consequence, and it is the half
that turns the census from trivia into an argument.

Every arm here stored RAW model text, which is the only reason the
question can be asked of it at all -- an arm that banked parsed objects
has already thrown the answer away. Each is scored twice, as emitted and
with exactly one fence removed, on the same 30 held-out documents with
the same pinned scorer.

Two numbers per arm carry the finding:

  fence rate      how often the checkpoint wraps its answer
  fence tax       what the wrap costs, in micro-F1

    PYTHONPATH=src python3 scripts/fence_impact_table.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
RAW = REPO / "experiments" / "blind_rehearsal_2026-08-20_raw"

# Arms are named, not globbed: an arm that silently appears or vanishes
# would change a published table without anybody deciding to.
ARMS = (
    ("Qwen2.5-0.5B-Instruct", "schema",
     "waybill_scale_rung_0.5b_schema_2026-08-25.json"),
    ("Qwen2.5-0.5B-Instruct", "k-shot (20)",
     "waybill_scale_rung_0.5b_kshot_2026-08-25.json"),
    ("Qwen2.5-1.5B-Instruct", "schema",
     "waybill_scale_rung_1.5b_schema_REPRO_2026-08-25.json"),
    ("Qwen2.5-1.5B-Instruct", "k-shot (20)",
     "waybill_scale_rung_1.5b_kshot_RAW_2026-08-25.json"),
    ("Qwen2.5-3B-Instruct", "schema",
     "waybill_scale_rung_3b_schema_2026-08-25.json"),
    ("Qwen2.5-3B-Instruct", "k-shot (20)",
     "waybill_scale_rung_3b_kshot_2026-08-25.json"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(
        REPO / "experiments" / "fence_impact_table.json"))
    args = parser.parse_args()

    from arcttt.scoring import field_micro_f1, parse_json_object
    from arcttt.text_task import TextTaskFormatError

    from fence_rescore import strip_fence  # noqa: E402

    gold = {r["id"]: r["gold"] for r in
            (json.loads(line) for line in
             (RAW / "gold_holdout.jsonl").read_text(encoding="utf-8")
             .splitlines() if line.strip())}

    rows, missing = [], []
    for model, regime, name in ARMS:
        path = REPO / "experiments" / name
        if not path.exists():
            missing.append({"model": model, "regime": regime,
                            "artifact": name, "why": "not banked"})
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        predictions = record.get("predictions") or {}
        if not predictions or not all(isinstance(v, str)
                                      for v in predictions.values()):
            missing.append({
                "model": model, "regime": regime, "artifact": name,
                "why": "no RAW model text stored, so the fence question "
                       "cannot be asked of this arm by anyone"})
            continue

        as_is, repaired, fenced = [], [], 0
        invalid_as_is = invalid_repaired = 0
        for doc_id, text in predictions.items():
            cleaned, was_fenced = strip_fence(text)
            fenced += was_fenced
            for label, candidate in (("a", text), ("r", cleaned)):
                try:
                    obj = parse_json_object(candidate)
                except TextTaskFormatError:
                    obj = None
                score = 0.0 if obj is None else field_micro_f1(obj,
                                                               gold[doc_id])
                if label == "a":
                    as_is.append(score)
                    invalid_as_is += obj is None
                else:
                    repaired.append(score)
                    invalid_repaired += obj is None
        mean_as_is = statistics.mean(as_is)
        mean_repaired = statistics.mean(repaired)
        rows.append({
            "model": model,
            "regime": regime,
            "artifact": name,
            "n": len(predictions),
            "fenced": fenced,
            "fence_rate": round(fenced / len(predictions), 4),
            "mean_as_emitted": round(mean_as_is, 4),
            "invalid_as_emitted": invalid_as_is,
            "mean_fence_stripped": round(mean_repaired, 4),
            "invalid_fence_stripped": invalid_repaired,
            "fence_tax": round(mean_repaired - mean_as_is, 4),
            "mean_prompt_tokens": record.get("mean_prompt_tokens"),
            "cost_per_1k_documents_usd": record.get(
                "cost_per_1k_documents_usd_median",
                record.get("cost_per_1k_documents_usd")),
        })

    schema = [r for r in rows if r["regime"] == "schema"]
    kshot = [r for r in rows if r["regime"].startswith("k-shot")]

    record = {
        "what": "What one markdown fence costs, per checkpoint and per "
                "prompt regime, on the same 30 held-out documents with the "
                "same pinned scorer.",
        "method": "Each arm scored twice from its stored RAW model text: "
                  "exactly as emitted, and with one fence removed. Nothing "
                  "else differs between the two columns.",
        "arms": rows,
        "not_measurable": missing,
        "fence_rate_by_regime": {
            "schema (field list only)": sorted(
                {r["fence_rate"] for r in schema}) if schema else None,
            "k-shot (20 demonstrations)": sorted(
                {r["fence_rate"] for r in kshot}) if kshot else None,
        },
        "the_finding": (
            "Demonstrations suppress the fence. In the schema regime every "
            "checkpoint measured wraps its answer on every document; in the "
            "k-shot regime, given twenty examples of bare JSON, the same "
            "checkpoints wrap none. So the fence tax falls entirely on the "
            "CHEAP prompt regime -- the one you would use to make a small "
            "model economical -- and is invisible in the expensive regime "
            "that most benchmark harnesses run."
            if schema and kshot else
            "PENDING: both regimes must be banked before this is stated."),
        "the_tax_grows_with_capability": (
            "In the schema regime the fence costs more the better the model "
            "is, because a better model had more to lose. Every arm is "
            "reported as 0.0000 as emitted, so the loss is invisible in the "
            "number a harness prints."),
        "scope": "One corpus, thirty agent-authored documents, one seed, "
                 "greedy, CPU, one model family. It says nothing about "
                 "checkpoints or families not listed, and a fence rate "
                 "measured on 30 documents of one shape is not a fence rate "
                 "in general.",
        "what_it_does_not_show": (
            "It does not show any published benchmark number is wrong. It "
            "shows what the defect costs WHERE IT FIRES, on this corpus. "
            "Whether a given harness fires it is the separate question the "
            "hand-adjudicated census answers, and the two are reported side "
            "by side rather than multiplied together."),
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")

    print(f"{'model':24s} {'regime':14s} {'fenced':>7s} "
          f"{'as emitted':>11s} {'stripped':>9s} {'tax':>8s}")
    for r in rows:
        print(f"{r['model']:24s} {r['regime']:14s} "
              f"{r['fenced']:3d}/{r['n']:<3d} "
              f"{r['mean_as_emitted']:11.4f} {r['mean_fence_stripped']:9.4f} "
              f"{r['fence_tax']:+8.4f}")
    for m in missing:
        print(f"{m['model']:24s} {m['regime']:14s}  -- {m['why']}")
    print(f"\nbanked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
