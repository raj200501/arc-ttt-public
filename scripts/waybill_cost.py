#!/usr/bin/env python3
"""Cost of the hosted arm ON THE WAYBILL CORPUS. Measured, finally.

Every dollar figure in this repository was measured on the *synthetic*
novel-schema corpus, at a different demonstration count and a different
token profile. On 2026-08-22 the Addendum I write-up carried one of them
onto the waybill corpus -- the cross-corpus conflation this project has
now had to correct three times -- and the claim was withdrawn because
nothing here had ever measured cost on realistic documents.

Both halves are now measured on THIS corpus, and the result is against
us. The k=20 price this page first published (~$1.55/1k) was the price
of a CHOICE, not of the workload: Addendum J's sweep shows the hosted
tier holds 0.9722 at k=2 -- inside the frozen 0.02 tolerance of its own
k=20 run -- for **~$0.40 per 1,000 documents**, against our measured
~$0.89 at 0.8833. **The payload-asymmetry cost argument is withdrawn and
its direction reverses**: on this corpus we are dearer and worse, not
cheaper and worse. Every demonstration count that was run is priced
below, so the cheapest arm holding quality is visible rather than only
the one that flattered the argument.

Rates are EXTERNAL QUOTES of a stated date, not measurements. They are
the same rates VERDICT.md already carries for this tier so the two pages
cannot drift.

    python3 scripts/waybill_cost.py
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent

# External list prices for the cheap hosted tier, quoted 2026-08-19 and
# already used in VERDICT.md's cache-state row. Not measured by us.
RATE_IN_PER_M = 0.30
RATE_OUT_PER_M = 2.50
RATE_DATE = "2026-08-19"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(
        REPO / "experiments" / "waybill_cost_2026-08-22.json"))
    args = parser.parse_args()

    runs = []
    for path in sorted(glob.glob(str(
            REPO / "experiments" / "waybill_market_baseline_*.json"))):
        record = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        usage = record.get("api_usage_tokens")
        if not usage or not usage.get("prompt"):
            continue
        n = record["n"]
        runs.append({
            "artifact": pathlib.Path(path).name,
            "n_documents": n,
            "prompt_tokens": usage["prompt"],
            "output_tokens": usage["output"],
            "prompt_tokens_per_1k_docs": round(usage["prompt"] / n * 1000),
            "output_tokens_per_1k_docs": round(usage["output"] / n * 1000),
        })
    if not runs:
        print("no run records API usage tokens yet")
        return 0

    # Read our own side from its artifact rather than restating it, so the
    # two cannot drift. Absent = still unmeasured, and it says so.
    ours_path = REPO / "experiments" / "waybill_cost_ours_2026-08-22.json"
    if ours_path.exists():
        ours = json.loads(ours_path.read_text(encoding="utf-8"))
        ours_cost = ours["cost_per_1k_documents_usd"]
        ours_note = ("Measured. See waybill_cost_ours_2026-08-22.json for "
                     "the rate, the correction applied to it, and the "
                     "quality caveat -- being cheaper is not being better, "
                     "and Addendum I is where the quality comparison lives.")
    else:
        ours_cost = None
        ours_note = ("UNMEASURED. Our throughput numbers come from the "
                     "synthetic corpus's document-only serving config and "
                     "do not transfer. Until it is measured this repository "
                     "has no cost comparison on realistic documents and "
                     "must not imply one.")

    per_k_in = sum(r["prompt_tokens_per_1k_docs"] for r in runs) / len(runs)
    per_k_out = sum(r["output_tokens_per_1k_docs"] for r in runs) / len(runs)
    cold = per_k_in / 1e6 * RATE_IN_PER_M + per_k_out / 1e6 * RATE_OUT_PER_M

    # The k=20 price above is the price of a CHOICE, not of the workload,
    # and Addendum J's sweep shows the choice is not forced. Price every
    # demonstration count that was actually run, so the cheapest arm that
    # still holds quality is visible instead of only the one that makes
    # our payload argument work.
    kshot = {}
    for path in sorted(glob.glob(str(
            REPO / "experiments" / "waybill_market_kshot_k*_run*.json"))):
        record = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        usage = record.get("api_usage_tokens") or {}
        if not usage.get("prompt"):
            continue
        k = record["n_demonstrations"]
        label = f"k={k}+schema" if record.get("schema_declared") else f"k={k}"
        n = record["n"]
        kshot.setdefault(label, {"means": [], "in": [], "out": []})
        kshot[label]["means"].append(record["hosted_mean_micro_f1"])
        kshot[label]["in"].append(usage["prompt"] / n * 1000)
        kshot[label]["out"].append(usage["output"] / n * 1000)
    kshot_priced = {}
    for label, block in kshot.items():
        tin = sum(block["in"]) / len(block["in"])
        tout = sum(block["out"]) / len(block["out"])
        kshot_priced[label] = {
            "runs": len(block["means"]),
            "mean_micro_f1": round(sum(block["means"]) / len(block["means"]), 4),
            "prompt_tokens_per_1k_docs": round(tin),
            "cost_per_1k_documents_usd_cold": round(
                tin / 1e6 * RATE_IN_PER_M + tout / 1e6 * RATE_OUT_PER_M, 3),
        }

    record = {
        "what": "Hosted-tier cost per 1,000 documents ON THE WAYBILL "
                "CORPUS, from measured token usage.",
        "status": "The TOKEN COUNTS are measured. The PRICES are external "
                  "list quotes of a stated date and are not a measurement."
                  + ("" if ours_cost is None else
                     " Our own side is measured too -- see "
                     "waybill_cost_ours_2026-08-22.json, including the "
                     "correction applied to it."),
        "rates": {"input_per_million_usd": RATE_IN_PER_M,
                  "output_per_million_usd": RATE_OUT_PER_M,
                  "quoted": RATE_DATE, "source": "external list price"},
        "runs": runs,
        "tokens_per_1k_documents": {"prompt": round(per_k_in),
                                    "output": round(per_k_out)},
        "hosted_cost_per_1k_documents_usd_cold": round(cold, 3),
        "why_the_prompt_is_large": "Each request carries all 20 "
                                   "demonstrations, matching what our own "
                                   "k-shot arm receives. That is the "
                                   "cold-cache case; a warm shared-prefix "
                                   "cache would cut the input side "
                                   "substantially and is NOT measured here.",
        "hosted_cost_by_demonstration_count": kshot_priced,
        "cheapest_hosted_arm_holding_quality": (
            "k=2 scores 0.9722 against k=20's 0.9865 -- inside Addendum J's "
            "frozen 0.02 tolerance -- on 498,000 prompt tokens per 1,000 "
            "documents against 4,332,000, an 8.7x reduction. The k=20 price "
            "is the price of a CHOICE no deployment is forced to make."),
        "our_cost_on_this_corpus": ours_cost,
        "our_cost_note": ours_note,
        "reading": (
            "The accuracy comparison on this corpus is decided and "
            "published. The COST comparison is half-measured: the hosted "
            "side is here, our side is not, and the gap is stated rather "
            "than closed with a number from a different corpus."
            if ours_cost is None else
            "**The cost argument this repository published is WITHDRAWN, "
            "and the direction reverses.** It priced the hosted arm at "
            "~$1.55/1k by assuming it must carry all 20 demonstrations in "
            "every request -- that assumption was the whole payload-"
            "asymmetry argument, and Addendum J's sweep refutes it. At k=2 "
            "the hosted tier scores 0.9722, inside the frozen 0.02 "
            "tolerance of its own k=20 run, for **~$0.40 per 1,000 "
            "documents**. Against our measured ~$0.89 at 0.8833, a buyer "
            "who can use a hosted API pays **less than half** what we cost "
            "and gets a better score. We are not cheaper-and-worse on this "
            "corpus; we are dearer-and-worse. What survives is what "
            "survived Addendum I and nothing more: workloads where a "
            "hosted API is not an option, where the comparison is not "
            "available at any price."),
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")

    print(f"runs with usage recorded : {len(runs)}")
    print(f"tokens per 1k documents  : {round(per_k_in):,} in / "
          f"{round(per_k_out):,} out")
    print(f"hosted cost per 1k docs  : ~${cold:.2f} cold "
          f"(at ${RATE_IN_PER_M}/M in + ${RATE_OUT_PER_M}/M out, {RATE_DATE})")
    for label, block in sorted(kshot_priced.items()):
        print(f"  hosted {label:12s} f1={block['mean_micro_f1']:.4f} "
              f"~${block['cost_per_1k_documents_usd_cold']:.2f}/1k "
              f"({block['prompt_tokens_per_1k_docs']:,} prompt tok/1k)")
    if ours_cost is None:
        print("our cost on this corpus  : UNMEASURED — stated, not filled in")
    else:
        print(f"our cost on this corpus  : ~${ours_cost:.2f} per 1k "
              f"(measured; at 0.8833 quality against their 0.9708-1.0000)")
    print(f"\nbanked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
