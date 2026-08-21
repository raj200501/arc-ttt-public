#!/usr/bin/env python3
"""Rebuild the waybill field-audit page from the recorded run artifacts.

The page is a viewer, not a document: every score, every field and every
model output in it is read out of the paired-baseline run at build time.
That is the point. A hand-written before/after page is a marketing
asset; one that regenerates from artifacts is checkable, and if a number
in the run changes the page changes with it or this script fails.

    python3 scripts/build_field_audit.py \
        --run-dir <challenge-rehearsal dir> \
        --artifact experiments/blind_rehearsal_baseline_2026-08-21.json \
        --out demo/waybill_field_audit.html

Inputs, all produced by the run itself:
  <run-dir>/package/holdout.jsonl          the documents, text only
  <run-dir>/package/gold_holdout.jsonl     the challenger's withheld gold
  <run-dir>/baseline-kshot-greedy/predictions.jsonl
  <run-dir>/baseline-adapted-greedy/predictions.jsonl

The template lives beside this script so the prose and the data cannot
drift apart in a copy-paste.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "demo" / "waybill_field_audit.html"
MARKER = "const DOCS = "


def read_jsonl(path: pathlib.Path, key: str) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = row.get(key)
    return out


def tier(doc_id: str) -> str:
    if doc_id.startswith("h-"):
        return "hard"
    if doc_id.startswith("m-"):
        return "medium"
    if doc_id.startswith("x-"):
        return "mixed"
    return "easy"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--out", default=str(TEMPLATE))
    args = parser.parse_args()

    run = pathlib.Path(args.run_dir)
    texts = read_jsonl(run / "package" / "holdout.jsonl", "text")
    gold = read_jsonl(run / "package" / "gold_holdout.jsonl", "gold")
    prompted = read_jsonl(
        run / "baseline-kshot-greedy" / "predictions.jsonl", "prediction")
    adapted = read_jsonl(
        run / "baseline-adapted-greedy" / "predictions.jsonl", "prediction")
    scores = json.loads(pathlib.Path(args.artifact).read_text())["per_doc"]

    missing = set(scores) - set(texts)
    if missing:
        sys.exit(f"artifact scores documents not in the holdout: {sorted(missing)}")

    docs = [{
        "id": doc_id,
        "tier": tier(doc_id),
        "text": texts[doc_id],
        "gold": gold[doc_id],
        "prompted": prompted.get(doc_id),
        "adapted": adapted.get(doc_id),
        "sp": scores[doc_id]["baseline"],
        "sa": scores[doc_id]["adapted"],
    } for doc_id in sorted(texts)]

    out = pathlib.Path(args.out)
    page = out.read_text(encoding="utf-8") if out.exists() else None
    if page is None or MARKER not in page:
        sys.exit(f"{out} is missing the '{MARKER}' data marker — the template "
                 "must exist and carry it; this script only swaps the data.")

    start = page.index(MARKER) + len(MARKER)
    end = page.index(";\n", start)
    rebuilt = (page[:start]
               + json.dumps(docs, separators=(",", ":"))
               + page[end:])
    out.write_text(rebuilt, encoding="utf-8")

    wins = sum(1 for d in docs if d["sa"] > d["sp"])
    losses = sum(1 for d in docs if d["sa"] < d["sp"])
    print(f"rebuilt {out} from {len(docs)} recorded documents "
          f"({wins}W/{losses}L/{len(docs) - wins - losses}T)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
