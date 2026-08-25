#!/usr/bin/env python3
"""Read Addendum J under its frozen rule. Written before the data landed.

J asks whether the hosted model actually NEEDS the 20 demonstrations that
make our payload-asymmetry argument work. If it does not, its input cost
collapses and that argument dies.

Committed while the sweep was still running, for the same reason the bar
was: a reader written after seeing the numbers can be shaped by them.

Two things this reader refuses to do:

1. **Read the bare k=0 arm as support for us.** k=0 with no schema
   information at all is a strawman -- no deployment would send it, and
   its failure mode here is invalid JSON (prose), not wrong field names,
   which is the format problem a free constrained decoder fixes. The
   decisive arm is k=0 WITH the tenant's field list declared, which is
   what a real deployment would send. If that arm is missing, the reader
   returns INCOMPLETE rather than a verdict.
2. **Return a verdict on a partial sweep**, for the reason Addendum I
   established: one hosted-API run is not a measurement.

    python3 scripts/addendum_j_summary.py
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics

REPO = pathlib.Path(__file__).resolve().parent.parent
# The banked k=20 comparison, from the Addendum I matched-turn runs.
K20_RANGE = (0.9708, 1.0000)
K20_MEAN = 0.9865
K20_PROMPT_TOKENS_PER_1K = 4_332_000
OUR_ADAPTED = 0.8833
HOLD_WITHIN = 0.02  # reading (a)'s frozen tolerance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(
        REPO / "experiments" / "waybill_market_kshot_summary_2026-08-22.json"))
    args = parser.parse_args()

    arms: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(str(
            REPO / "experiments" / "waybill_market_kshot_k*_run*.json"))):
        record = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        k = record["n_demonstrations"]
        label = f"k={k}+schema" if record.get("schema_declared") else f"k={k}"
        arms.setdefault(label, []).append({
            "artifact": pathlib.Path(path).name,
            "mean_micro_f1": record["hosted_mean_micro_f1"],
            "invalid_json": record["hosted_invalid_json"],
            "prompt_tokens": record["api_usage_tokens"]["prompt"],
            "n": record["n"],
        })

    summary = {}
    for label, runs in sorted(arms.items()):
        means = [r["mean_micro_f1"] for r in runs]
        n_docs = runs[0]["n"]
        summary[label] = {
            "runs": len(runs),
            "means": means,
            "mean_of_runs": round(statistics.mean(means), 4),
            "range": [min(means), max(means)],
            "invalid_json_worst": max(r["invalid_json"] for r in runs),
            "prompt_tokens_per_1k_docs": round(
                runs[0]["prompt_tokens"] / n_docs * 1000),
        }

    schema_arm = summary.get("k=0+schema")
    if schema_arm is None or schema_arm["runs"] < 3:
        verdict, why = "INCOMPLETE", (
            "The decisive arm -- k=0 with the tenant's field list declared "
            "-- has "
            f"{0 if schema_arm is None else schema_arm['runs']} of 3 runs. "
            "No verdict is returned without it. The bare k=0 arm is a "
            "strawman: no deployment sends zero schema information, and "
            "its failure here is invalid JSON rather than wrong field "
            "names, which is a format problem a free constrained decoder "
            "fixes. Reading the sweep without the schema arm would be "
            "reading the arm that flatters us.")
    else:
        best = schema_arm["mean_of_runs"]
        savings = (K20_PROMPT_TOKENS_PER_1K
                   / schema_arm["prompt_tokens_per_1k_docs"])
        if best >= K20_MEAN - HOLD_WITHIN:
            verdict, why = "(a) OUR ARGUMENT IS WITHDRAWN", (
                f"With only its field list declared and no demonstrations, "
                f"the hosted model scores {best:.4f} against {K20_MEAN:.4f} "
                f"at k=20 -- within the frozen {HOLD_WITHIN} tolerance -- on "
                f"{savings:.0f}x fewer prompt tokens. **It never needed the "
                "20 demonstrations, only a schema line. The payload "
                "asymmetry is not structural and our cost argument on this "
                "corpus is withdrawn**, in the words frozen before this arm "
                "ran.")
        elif best >= OUR_ADAPTED:
            verdict, why = "(c) ASYMMETRY REAL, BUYS US NOTHING ON QUALITY", (
                f"The schema-declared arm scores {best:.4f}: below k=20's "
                f"{K20_MEAN:.4f} but still at or above our adapted "
                f"{OUR_ADAPTED}. The demonstrations are worth something, so "
                f"the payload asymmetry is real ({savings:.0f}x fewer prompt "
                "tokens without them) -- but a buyer can drop them, stay "
                "above us on quality, and pay far less. The claim narrows to "
                "cost at a quality we do not have.")
        else:
            verdict, why = "(b) THE DEMONSTRATIONS ARE LOAD-BEARING", (
                f"The schema-declared arm scores {best:.4f}, below our "
                f"adapted {OUR_ADAPTED} and well below k=20's "
                f"{K20_MEAN:.4f}. A schema line does not substitute for the "
                "demonstrations on this corpus, so the hosted model must "
                "carry them, and the payload asymmetry stands as published.")

    # The frozen readings key on k=0. The sweep also runs k=2 and k=5, and
    # if a small non-zero k already holds quality then the asymmetry is
    # gone in substance even though reading (b) applies by the letter.
    # Preregistration is a commitment, not a shield: report both, and name
    # the discrepancy rather than quoting the wording that flatters us.
    cheapest_holding = None
    for label, block in summary.items():
        if label == "k=0":
            continue
        if block["mean_of_runs"] >= K20_MEAN - HOLD_WITHIN:
            if (cheapest_holding is None
                    or block["prompt_tokens_per_1k_docs"]
                    < summary[cheapest_holding]["prompt_tokens_per_1k_docs"]):
                cheapest_holding = label
    letter_note = None
    if cheapest_holding and not verdict.startswith("(a)"):
        block = summary[cheapest_holding]
        letter_note = (
            f"**The letter of the frozen readings and their substance "
            f"disagree, and the substance is against us.** The readings key "
            f"on k=0, where the model scores 0.0000, so reading (b) applies "
            f"by the wording. But {cheapest_holding} already scores "
            f"{block['mean_of_runs']:.4f} -- within the frozen "
            f"{HOLD_WITHIN} tolerance of k=20 -- on "
            f"{block['prompt_tokens_per_1k_docs']:,} prompt tokens per 1,000 "
            f"documents against {K20_PROMPT_TOKENS_PER_1K:,}, a "
            f"{K20_PROMPT_TOKENS_PER_1K / block['prompt_tokens_per_1k_docs']:.1f}x "
            f"reduction. The hosted model does not need 20 demonstrations; "
            f"it needs a couple. A preregistration is a commitment, not a "
            f"shield, so this is stated rather than left to the wording "
            f"that favours us.")

    record = {
        "addendum": "J",
        "what": "Does the hosted model need the 20 demonstrations that our "
                "payload-asymmetry cost argument depends on?",
        "preregistered": "VERDICT.md Addendum J, with the schema-declared "
                         "arm added after the first k=0 cell and BEFORE the "
                         "rest of the sweep -- an ordering that makes it an "
                         "experiment rather than a rescue.",
        "reader_written": "before the sweep finished, deliberately",
        "comparison_k20": {"mean": K20_MEAN, "range": list(K20_RANGE),
                           "prompt_tokens_per_1k_docs":
                               K20_PROMPT_TOKENS_PER_1K},
        "our_adapted_0_5b": OUR_ADAPTED,
        "arms": summary,
        "verdict": verdict,
        "why": why,
        "cheapest_arm_holding_quality": cheapest_holding,
        "letter_versus_substance": letter_note,
        "bare_k0_note": "The bare k=0 arm is reported but is NOT evidence "
                        "for us. Its failure mode is invalid JSON, not "
                        "wrong field names -- the model answers in prose "
                        "when told nothing, which is the format problem a "
                        "free constrained decoder fixes. Citing it as proof "
                        "that demonstrations are load-bearing would be "
                        "beating a strawman.",
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")

    for label, block in summary.items():
        print(f"  {label:14s} runs={block['runs']} "
              f"mean={block['mean_of_runs']:.4f} "
              f"range={block['range']} "
              f"worst_invalid={block['invalid_json_worst']:2d} "
              f"prompt_tok/1k={block['prompt_tokens_per_1k_docs']:,}")
    print(f"  {'k=20 (banked)':14s} mean={K20_MEAN:.4f} "
          f"prompt_tok/1k={K20_PROMPT_TOKENS_PER_1K:,}")
    print(f"\nVERDICT: {verdict}\n{why}")
    if letter_note:
        print(f"\n{letter_note}")
    print(f"\nbanked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
