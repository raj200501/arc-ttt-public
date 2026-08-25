#!/usr/bin/env python3
"""Read Addendum K under its frozen bar. Written before it was run.

K asks whether the gap between our adapted 0.5B (0.8833) and a hosted
tier (0.9708-1.0000) on the 30 held-out waybills is a CAPABILITY gap or
an IMPLEMENTATION one. The diagnosis said implementation: of 28 field
errors, only one was a clean role swap; the rest were confabulation,
un-asked rewriting, and OCR damage copied through. So the model was
being asked to do by generation what the document already determines.

`arcttt.grounding` does that part deterministically. This reads the
result, and it is committed BEFORE the holdout is scored with it, for
the same reason the bar was frozen first: a reader written after seeing
the number can be shaped by it.

**The one thing this reader refuses to do is grant grounding to our arm
alone.** Grounding is an inference-time component of a serving stack,
not a scoring adjustment -- it reads the document and the model's own
output, never gold -- so anyone can adopt it, including the hosted
model. Scoring our grounded arm against their un-grounded one would
repeat exactly the asymmetry this project already had to correct once,
when a markdown-fence repair was granted to the hosted arm and not to
ours. Every arm here is grounded by the same code, and the comparison
that decides anything is grounded-against-grounded.

    PYTHONPATH=src python3 scripts/addendum_k_grounding.py
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import pathlib
import statistics

from arcttt.grounding import ground, infer_copy_fields
from arcttt.scoring import field_micro_f1

REPO = pathlib.Path(__file__).resolve().parent.parent
RAW = REPO / "experiments" / "blind_rehearsal_2026-08-20_raw"

# The frozen bar, transcribed from VERDICT.md's Addendum K row.
BAR_RESTORES = 0.95
BAR_MATERIAL = 0.90
# Banked comparisons, so this page and VERDICT.md cannot drift.
HOSTED_RANGE = (0.9708, 1.0000)
HOSTED_MEAN = 0.9865


def _rows(name: str) -> list[dict]:
    return [json.loads(line) for line
            in (RAW / name).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sign_test(deltas: list[float], label: str) -> dict:
    """Sign test naming its own direction, as Addendum I forced."""
    wins = sum(1 for d in deltas if d > 1e-12)
    losses = sum(1 for d in deltas if d < -1e-12)
    ties = len(deltas) - wins - losses
    n = wins + losses

    def tail(k: int) -> float:
        return (sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n
                if n else 1.0)

    observed = ("ours > theirs" if wins > losses else
                "theirs > ours" if losses > wins else "tied")
    return {"compared": label, "wins": wins, "losses": losses, "ties": ties,
            "observed_direction": observed,
            "p_value_ours_greater": tail(wins),
            "p_value_theirs_greater": tail(losses),
            "p_value_in_observed_direction": (
                tail(wins) if wins > losses else
                tail(losses) if losses > wins else 1.0)}


def _score_arm(predictions: dict[str, dict], documents: dict[str, str],
               gold: dict[str, dict], copy_fields: list[str]) -> dict:
    """Score one arm before and after grounding, per document."""
    before: dict[str, float] = {}
    after: dict[str, float] = {}
    changed_documents = 0
    damaged: list[dict] = []
    for doc_id, gold_object in gold.items():
        prediction = predictions.get(doc_id)
        if prediction is None:
            # An arm that emitted unparseable output scores zero on that
            # document; grounding cannot rescue what was never an object,
            # and pretending otherwise would invent a number.
            before[doc_id] = after[doc_id] = 0.0
            continue
        grounded, changes = ground(prediction, documents[doc_id], copy_fields)
        before[doc_id] = field_micro_f1(prediction, gold_object)
        after[doc_id] = field_micro_f1(grounded, gold_object)
        if changes:
            changed_documents += 1
        if after[doc_id] < before[doc_id] - 1e-12:
            damaged.append({"id": doc_id,
                            "before": round(before[doc_id], 4),
                            "after": round(after[doc_id], 4),
                            "changes": changes})
    return {
        "mean_before": round(statistics.mean(before.values()), 4),
        "mean_after": round(statistics.mean(after.values()), 4),
        "documents_grounding_touched": changed_documents,
        "documents_grounding_damaged": len(damaged),
        "damage": damaged,
        "per_document_after": after,
        "per_document_before": before,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(
        REPO / "experiments" / "waybill_grounded_2026-08-22.json"))
    args = parser.parse_args()

    # The banked artifact records the SHA-256 of the recipe that produced
    # it. Addendum K's holdout was scored once, under the recipe frozen at
    # commit 48ee9df; three tokenizer defects were found afterwards by
    # inspecting the damage, so a repaired recipe's numbers on these same
    # 30 documents are holdout-informed and are not the addendum's result.
    # Letting a re-run overwrite the frozen artifact in place would erase
    # that distinction silently, which is the one thing this file exists
    # to prevent -- so it refuses, and names the flag that says you meant
    # it.
    recipe = hashlib.sha256(
        (REPO / "src" / "arcttt" / "grounding.py").read_bytes()).hexdigest()
    out_path = pathlib.Path(args.out)
    if out_path.exists():
        banked = json.loads(out_path.read_text(encoding="utf-8"))
        if banked.get("recipe_sha256") not in (None, recipe):
            print(f"REFUSING to overwrite {out_path.name}: it was produced "
                  f"by a different grounding recipe\n  banked "
                  f"{banked['recipe_sha256'][:12]} != current "
                  f"{recipe[:12]}\nAddendum K's holdout is scored ONCE. Pass "
                  f"--out <other path> to score a repaired recipe; its "
                  f"number is holdout-informed and is not the addendum's "
                  f"result.")
            return 1

    documents = {row["id"]: row["text"] for row in _rows("holdout.jsonl")}
    gold = {row["id"]: row["gold"] for row in _rows("gold_holdout.jsonl")}

    # Copy-type fields come from the TRAIN split. The holdout's gold is
    # never consulted to build the recipe -- only to score it.
    train = _rows("train.jsonl")
    copy_fields = infer_copy_fields([(r["text"], r["gold"]) for r in train])

    arms: dict[str, dict] = {}
    for label, filename in (("ours_adapted", "predictions_adapted_greedy.jsonl"),
                            ("ours_prompted",
                             "predictions_prompted_greedy.jsonl")):
        rows = _rows(filename)
        predictions = {r["id"]: r["prediction"] for r in rows
                       if isinstance(r.get("prediction"), dict)}
        arms[label] = _score_arm(predictions, documents, gold, copy_fields)

    # Every hosted run, grounded by the same code. If grounding lifts them
    # too, our relative position is unchanged and this addendum has not
    # done what it set out to do -- and that has to be visible here.
    hosted_runs = []
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
        scored = _score_arm(predictions, documents, gold, copy_fields)
        scored["artifact"] = pathlib.Path(path).name
        hosted_runs.append(scored)

    ours = arms["ours_adapted"]
    grounded_mean = ours["mean_after"]

    if grounded_mean >= BAR_RESTORES:
        verdict = "(a) THE GAP WAS AN IMPLEMENTATION ARTIFACT"
        why = (
            f"Grounded, our adapted 0.5B scores {grounded_mean:.4f} on the "
            f"30 held-out waybills, at or above the frozen {BAR_RESTORES} "
            f"bar, against the hosted tier's {HOSTED_RANGE[0]:.4f}-"
            f"{HOSTED_RANGE[1]:.4f}. In the words frozen before this ran: "
            "**a properly-served 0.5B reaches the hosted tier's range on "
            "realistic documents, and the original claim is restored.**")
    elif grounded_mean >= BAR_MATERIAL:
        verdict = "(b) MATERIALLY CLOSED, NOT MATCHED"
        why = (
            f"Grounded, our adapted 0.5B scores {grounded_mean:.4f} -- above "
            f"the frozen {BAR_MATERIAL} bar and below {BAR_RESTORES}. Per "
            "the frozen reading, **the claim is stated at exactly the "
            f"number reached, {grounded_mean:.4f}, and not at the hosted "
            "tier's range.** The gap narrowed; it did not close.")
    else:
        verdict = "(c) NOT AN IMPLEMENTATION ARTIFACT"
        why = (
            f"Grounded, our adapted 0.5B scores {grounded_mean:.4f}, below "
            f"the frozen {BAR_MATERIAL} bar. **The gap is not an "
            "implementation artifact and the Addendum I scoping stands "
            "unchanged**, in the words frozen before this ran.")

    # Grounded-against-grounded is the only comparison that decides
    # anything, because grounding is available to both sides.
    hosted_after = ([statistics.mean(
        [r["per_document_after"][d] for d in gold]) for r in hosted_runs]
        if hosted_runs else [])
    paired = []
    for run in hosted_runs:
        deltas = [ours["per_document_after"][d] - run["per_document_after"][d]
                  for d in gold]
        paired.append({
            "artifact": run["artifact"],
            "hosted_mean_grounded": run["mean_after"],
            "mean_delta_ours_minus_theirs": round(
                statistics.mean(deltas), 4),
            "sign_test": sign_test(deltas, "ours grounded - hosted grounded"),
        })

    record = {
        "addendum": "K",
        "recipe_sha256": recipe,
        "what": "Is the gap to the hosted model a capability gap or an "
                "implementation one?",
        "preregistered": "VERDICT.md Addendum K, frozen before any component "
                         "was built. Recipe frozen by commit before the "
                         "holdout was scored with it.",
        "reader_written": "before the holdout was scored, deliberately",
        "recipe": {
            "mechanisms": ["span snapping", "OCR repair by character class"],
            "reads_gold": False,
            "copy_fields_source": "TRAIN split only",
            "copy_fields": copy_fields,
        },
        "bar": {"restores_claim": BAR_RESTORES,
                "materially_closed": BAR_MATERIAL},
        "ours_adapted": {k: v for k, v in ours.items()
                         if not k.startswith("per_document")},
        "ours_prompted": {k: v for k, v in arms["ours_prompted"].items()
                          if not k.startswith("per_document")},
        "hosted_runs_grounded": [
            {k: v for k, v in r.items() if not k.startswith("per_document")}
            for r in hosted_runs],
        "hosted_ungrounded_banked": {"mean": HOSTED_MEAN,
                                     "range": list(HOSTED_RANGE)},
        "hosted_grounded_range": ([round(min(hosted_after), 4),
                                   round(max(hosted_after), 4)]
                                  if hosted_after else None),
        "paired_grounded_vs_grounded": paired,
        "verdict": verdict,
        "why": why,
        "fairness_note": "Grounding is an inference-time component of a "
                         "serving stack, not a scoring adjustment: it reads "
                         "the document and the model's own output and never "
                         "reads gold. So it is available to the hosted model "
                         "too, and every arm above is grounded by the same "
                         "code. Citing our grounded score against their "
                         "un-grounded one would repeat the fence-strip "
                         "asymmetry this project already had to correct.",
        "disclosure": "The holdout error taxonomy was inspected during "
                      "diagnosis (it was published by the challenger on "
                      "2026-08-20), so this is not a blind design. The "
                      "mechanisms are justified from TRAIN statistics and "
                      "the published taxonomy, the recipe was frozen by "
                      "commit before the holdout was scored, and the "
                      "holdout was scored once -- but a reader should "
                      "discount accordingly, and the blind-holdout offer "
                      "remains the only thing that retires the objection.",
    }
    out_path.write_text(json.dumps(record, indent=2) + "\n",
                        encoding="utf-8")

    print(f"copy-type fields (TRAIN-derived): {copy_fields}")
    for label in ("ours_adapted", "ours_prompted"):
        block = arms[label]
        print(f"  {label:14s} {block['mean_before']:.4f} -> "
              f"{block['mean_after']:.4f}   "
              f"touched={block['documents_grounding_touched']:2d} "
              f"damaged={block['documents_grounding_damaged']:2d}")
    for run in hosted_runs:
        print(f"  {'hosted':14s} {run['mean_before']:.4f} -> "
              f"{run['mean_after']:.4f}   "
              f"touched={run['documents_grounding_touched']:2d} "
              f"damaged={run['documents_grounding_damaged']:2d}  "
              f"{run['artifact']}")
    print(f"\nVERDICT: {verdict}\n{why}")
    if paired:
        print("\ngrounded against grounded (the comparison that decides):")
        for block in paired:
            test = block["sign_test"]
            print(f"  vs {block['artifact']}: "
                  f"delta={block['mean_delta_ours_minus_theirs']:+.4f} "
                  f"{test['wins']}W/{test['losses']}L/{test['ties']}T "
                  f"p={test['p_value_in_observed_direction']:.2e} "
                  f"({test['observed_direction']})")
    print(f"\nbanked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
