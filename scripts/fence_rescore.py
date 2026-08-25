#!/usr/bin/env python3
"""Was that zero a capability failure, or a markdown fence?

Addendum O's 3B schema-only arm scored **0.0000 with 30 of 30 outputs
unparseable**, which read as a clean survival of the kill gate. The
outputs are correct extractions wrapped in ```json fences.

`run_market_baseline_waybills.py` already knew about this and wrote the
rule down: strip the fence, but publish the un-repaired mean beside the
repaired one, and say plainly that OUR arms -- produced by
`run_challenge.py`, which does a bare `json.loads` -- never received the
repair. Granting a repair to one arm alone is an asymmetry. The
scale-rung runner did not carry that rule forward, so it reported the
un-repaired number alone, and the un-repaired number is the one that
flatters this project.

**This reader exists to state both, on every arm that stored its raw
output, and to make the flattering reading impossible to take by
accident.** It is deliberately a separate script rather than a change to
the runner: arms were still generating when the defect was found, and
editing the generator mid-experiment would have made the finished arms
and the unfinished ones different experiments.

A fence is three lines of code in any real deployment. Treating it as a
capability failure -- on a rival's arm, in a comparison this project
would lose without it -- is not a defensible reading and this repository
will not publish one.

    PYTHONPATH=src python3 scripts/fence_rescore.py experiments/waybill_scale_rung_*.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
RAW = REPO / "experiments" / "blind_rehearsal_2026-08-20_raw"


def strip_fence(text: str) -> tuple[str, bool]:
    """Remove one markdown fence. Returns (text, was_fenced)."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped, False
    body = stripped.split("\n", 1)[-1]
    if "```" in body:
        body = body.rsplit("```", 1)[0]
    return body.strip(), True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+")
    parser.add_argument("--out", default=str(
        REPO / "experiments" / "fence_rescore.json"))
    args = parser.parse_args()

    from arcttt.scoring import field_micro_f1, parse_json_object
    from arcttt.text_task import TextTaskFormatError

    gold = {r["id"]: r["gold"] for r in
            (json.loads(line) for line in
             (RAW / "gold_holdout.jsonl").read_text(encoding="utf-8")
             .splitlines() if line.strip())}

    rows = []
    for name in args.artifacts:
        path = pathlib.Path(name)
        record = json.loads(path.read_text(encoding="utf-8"))
        predictions = record.get("predictions")
        if not isinstance(predictions, dict) or not predictions:
            rows.append({"artifact": path.name,
                         "status": "NO STORED PREDICTIONS -- cannot be "
                                   "re-scored at all, by anyone, including "
                                   "us. This is the ARITHMETIC level on the "
                                   "coverage map and it is why that map "
                                   "exists."})
            continue
        if not all(isinstance(v, str) for v in predictions.values()):
            rows.append({"artifact": path.name,
                         "status": "predictions are parsed objects, not raw "
                                   "model output; the fence question cannot "
                                   "be asked of them"})
            continue

        as_is, repaired, fenced, invalid_as_is, invalid_repaired = \
            [], [], [], 0, 0
        for doc_id, text in predictions.items():
            for label, candidate in (("as_is", text),
                                     ("repaired", strip_fence(text)[0])):
                try:
                    obj = parse_json_object(candidate)
                except TextTaskFormatError:
                    obj = None
                score = 0.0 if obj is None else field_micro_f1(
                    obj, gold[doc_id])
                if label == "as_is":
                    as_is.append(score)
                    invalid_as_is += obj is None
                else:
                    repaired.append(score)
                    invalid_repaired += obj is None
            if strip_fence(text)[1]:
                fenced.append(doc_id)

        mean_as_is = statistics.mean(as_is)
        mean_repaired = statistics.mean(repaired)
        rows.append({
            "artifact": path.name,
            "model": record.get("model"),
            "mode": record.get("mode"),
            "banked_mean_micro_f1": record.get("mean_micro_f1"),
            "mean_as_is": round(mean_as_is, 4),
            "invalid_as_is": invalid_as_is,
            "mean_fence_stripped": round(mean_repaired, 4),
            "invalid_fence_stripped": invalid_repaired,
            "documents_fenced": len(fenced),
            "delta_from_fence_alone": round(mean_repaired - mean_as_is, 4),
            "reading": (
                "THE ZERO WAS A FENCE, NOT A FAILURE. The un-repaired "
                "number must not be cited as a capability result."
                if mean_as_is < 0.05 <= mean_repaired else
                "The fence is not what decided this arm."),
        })

    record = {
        "what": "Every scale-rung arm that stored raw model output, scored "
                "twice: exactly as emitted, and with one markdown fence "
                "removed.",
        "why": "An arm scored 0.0000 with 30 of 30 outputs unparseable, and "
               "the outputs were correct extractions inside ```json fences. "
               "The un-repaired number is the one that flatters this "
               "project, and it is the one the runner reported alone.",
        "the_asymmetry_that_must_be_stated": (
            "OUR arms were produced by run_challenge.py, which does a bare "
            "json.loads and records a null on failure -- they never received "
            "this repair either. The adapted model is trained on the output "
            "format and does not fence, so in practice the repair costs it "
            "nothing; but that is an empirical claim about our arm and not a "
            "reason the rival's arm should be denied a repair any deployment "
            "would perform. Where a comparison turns on the fence, BOTH "
            "numbers are published and the fence-stripped one governs."),
        "rows": rows,
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")
    for row in rows:
        if "status" in row:
            print(f"{row['artifact']}: {row['status']}")
            continue
        print(f"{row['artifact']}\n"
              f"    as emitted     {row['mean_as_is']:.4f}  "
              f"({row['invalid_as_is']} invalid)\n"
              f"    fence stripped {row['mean_fence_stripped']:.4f}  "
              f"({row['invalid_fence_stripped']} invalid)  "
              f"[{row['documents_fenced']}/30 fenced]\n"
              f"    {row['reading']}")
    print(f"\nbanked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
