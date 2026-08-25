#!/usr/bin/env python3
"""Where does the gap to the hosted model actually live? Post-hoc.

Addendum K set out to close the gap with document grounding and returned
reading (c). The reason, found by re-classifying the same banked
predictions, is that the errors are not the ones grounding fixes. This
script does that classification for every arm, so the claim rests on a
countable decomposition rather than on the four examples the Addendum K
row generalised from -- which is exactly the mistake that cost a day.

Every wrong field lands in one of three buckets, decided by the document
and the tenant's gold, with no judgement call left to the author:

  ASSIGNMENT   the emitted value IS the gold value of a DIFFERENT field
               of the same document. The model read the document
               correctly and put the answer under the wrong key. This is
               schema knowledge, not reading.
  TRUNCATION   not assignment, but every word of the value occurs in the
               document. The model was in the right place and stopped
               early, or picked the wrong span.
  INVENTION    the value contains a word the document does not. The
               model confabulated or copied scanner damage through.

Span snapping can only ever help the third bucket, and partly the
second. If the first dominates, the gap is a capability gap and no
amount of document grounding touches it -- which is what Addendum K
measured the hard way.

**This is post-hoc and it is not a gate.** It reads the holdout, whose
gold is published, after the fact. It is a decomposition of a result
already decided, not evidence for a new one, and nothing here is
preregistered.

    PYTHONPATH=src python3 scripts/error_mode_decomposition.py
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib

from arcttt.grounding import (
    infer_copy_fields,
    is_document_grounded,
    repair_ocr,
    snap_to_document,
)
from arcttt.scoring import normalize_value

REPO = pathlib.Path(__file__).resolve().parent.parent
RAW = REPO / "experiments" / "blind_rehearsal_2026-08-20_raw"


def _rows(name: str) -> list[dict]:
    return [json.loads(line) for line
            in (RAW / name).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def classify(key: str, got: object, gold_object: dict, document: str) -> str:
    """Which bucket does one wrong field fall into?"""
    if got is None:
        return "missing"
    text = str(got)
    for other_key, other_value in gold_object.items():
        if other_key == key:
            continue
        if normalize_value(text) == normalize_value(str(other_value)):
            return "assignment"
    return "truncation" if is_document_grounded(text, document) else "invention"


def decompose(predictions: dict[str, dict], documents: dict[str, str],
              gold: dict[str, dict], copy_fields: set[str]) -> dict:
    counts: collections.Counter[str] = collections.Counter()
    # The number that explains Addendum K, counted rather than argued:
    # how many of these errors could document grounding fix AT ALL? An
    # error is grounding-addressable if the field is copy-type and
    # snapping the model's own value lands on gold.
    addressable: collections.Counter[str] = collections.Counter()
    examples: dict[str, list[str]] = collections.defaultdict(list)
    unparseable = 0
    for doc_id, gold_object in gold.items():
        prediction = predictions.get(doc_id)
        if prediction is None:
            unparseable += 1
            continue
        for key, value in gold_object.items():
            got = prediction.get(key)
            if got is not None and normalize_value(str(got)) == normalize_value(
                    str(value)):
                continue
            bucket = classify(key, got, gold_object, documents[doc_id])
            counts[bucket] += 1
            if key in copy_fields and got is not None:
                snapped, _ = snap_to_document(
                    repair_ocr(str(got)), documents[doc_id])
                if normalize_value(snapped) == normalize_value(str(value)):
                    addressable[bucket] += 1
            if len(examples[bucket]) < 4:
                examples[bucket].append(
                    f"[{doc_id}] {key}: {got!r} (gold {value!r})")
    total = sum(counts.values())
    return {
        "field_errors": total,
        "unparseable_documents": unparseable,
        "buckets": dict(counts),
        "grounding_addressable": dict(addressable),
        "grounding_addressable_total": sum(addressable.values()),
        "share": {k: round(v / total, 3) for k, v in counts.items()} if total
        else {},
        "examples": dict(examples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(
        REPO / "experiments" / "waybill_error_modes_2026-08-22.json"))
    args = parser.parse_args()

    documents = {row["id"]: row["text"] for row in _rows("holdout.jsonl")}
    gold = {row["id"]: row["gold"] for row in _rows("gold_holdout.jsonl")}

    copy_fields = set(infer_copy_fields(
        [(r["text"], r["gold"]) for r in _rows("train.jsonl")]))

    arms: dict[str, dict] = {}
    for label, filename in (("ours_adapted", "predictions_adapted_greedy.jsonl"),
                            ("ours_prompted",
                             "predictions_prompted_greedy.jsonl")):
        rows = _rows(filename)
        arms[label] = decompose(
            {r["id"]: r["prediction"] for r in rows
             if isinstance(r.get("prediction"), dict)}, documents, gold,
            copy_fields)

    for path in sorted(glob.glob(str(
            REPO / "experiments"
            / "waybill_market_baseline_*matchedturns*.json"))):
        record = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        predictions = {}
        for result in record["results"]:
            try:
                parsed = json.loads(result["prediction"])
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, dict):
                predictions[result["id"]] = parsed
        arms[f"hosted:{pathlib.Path(path).stem[-14:]}"] = decompose(
            predictions, documents, gold, copy_fields)

    adapted = arms["ours_adapted"]
    assignment = adapted["buckets"].get("assignment", 0)
    invention = adapted["buckets"].get("invention", 0)
    record = {
        "what": "Decomposition of every wrong field on the 30 held-out "
                "waybills, by whether the model read the document wrong or "
                "filed a correct read under the wrong key.",
        "status": "POST-HOC. Not a gate, not preregistered, and not "
                  "evidence for a new claim -- a decomposition of a result "
                  "already decided (Addendum K, reading (c)).",
        "buckets": {
            "assignment": "the value IS another field's gold on the same "
                          "document: read right, filed wrong",
            "truncation": "every word occurs in the document, but the span "
                          "is wrong",
            "invention": "contains a word the document does not have",
        },
        "arms": arms,
        "reading": (
            f"Of our adapted arm's {adapted['field_errors']} field errors, "
            f"{assignment} are ASSIGNMENT (read right, filed wrong), "
            f"{adapted['buckets'].get('truncation', 0)} are TRUNCATION and "
            f"{invention} are INVENTION. **The number that explains Addendum "
            f"K is the last column: only "
            f"{adapted['grounding_addressable_total']} of "
            f"{adapted['field_errors']} are reachable by document grounding "
            "at all.** Assignment errors are unreachable by construction -- "
            "the value grounding would snap to is already in the document. "
            "Most inventions are unreachable too, but for a different "
            "reason: the model picked the wrong ENTITY (`Tampa` for "
            "`Tallahassee`, `Cincinnati` for `Council Bluffs`) or the wrong "
            "DIGITS (`188.60` for `1088.60`), and a document that contains "
            "several names and several numbers cannot tell a snapper which "
            "one belongs in this field. The common thread across the "
            "buckets is SELECTION, not transcription, and that is a "
            "capability gap."),
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")

    width = max(len(k) for k in arms)
    print(f"{'arm'.ljust(width)}  errors  assignment truncation invention"
          f"  grounding-fixable")
    for label, block in arms.items():
        bucket = block["buckets"]
        print(f"{label.ljust(width)}  {block['field_errors']:6d}  "
              f"{bucket.get('assignment', 0):10d} "
              f"{bucket.get('truncation', 0):10d} "
              f"{bucket.get('invention', 0):9d}"
              f"  {block['grounding_addressable_total']:16d}")
    print(f"\n{record['reading']}")
    print(f"\nbanked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
