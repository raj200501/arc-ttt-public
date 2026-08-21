#!/usr/bin/env python3
"""Run the whole thing on YOUR documents, and print the paired before/after.

The repo could be recomputed but not USED. Everything a reader could run
verified our numbers on our data; nothing pointed the machinery at
theirs. This is the missing command:

    python3 scripts/try_your_documents.py --docs mydocs.jsonl

One JSONL, one line per document:

    {"id": "inv-001", "text": "<the document text>", "gold": {...}}

It splits your documents, runs BOTH arms at a matched decode -- the same
base model prompted with your training pairs (baseline), and the same
base model adapted on them (the product) -- scores both against your own
gold with the pinned scorer, and prints the paired delta with a sign
test. That is the same comparison every gate row in VERDICT.md is built
on, run on your data instead of ours.

WHAT THIS IS NOT, stated here rather than in a footnote, because a number
this script prints could otherwise be mistaken for a gate result:

  * It is NOT blind. Your gold is on this machine and this process reads
    it to score. Nothing is withheld from anyone.
  * It has NO preregistered bar. The gates in VERDICT.md fixed their
    thresholds before their data existed; this fixes nothing, and you can
    re-run it with different settings until you like the answer. So can
    we. That is exactly the freedom preregistration gives up on purpose.
  * It is ONE corpus, ONE schema, ONE run. A data point, not a benchmark.
  * A good mean can hide a bad tier, so per-document scores print too.

If you want the version that IS evidence -- blind, single submission,
gold you keep, published either way -- that is the standing offer in
CHALLENGES.md, and its terms are in docs/research/CHALLENGE_TERMS.md.

Needs torch (it adapts weights). The verification scripts do not; this
one does, and the split/score halves still run stdlib-only.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent


def run(cmd: list[str], label: str) -> None:
    print(f"\n[{label}] {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=REPO)
    if result.returncode != 0:
        sys.exit(f"[{label}] failed with exit code {result.returncode}")


def score_arm(pred: pathlib.Path, gold: pathlib.Path) -> dict[str, float]:
    """Score one arm with the pinned scorer, returning per-document F1."""
    sys.path.insert(0, str(REPO / "src"))
    from arcttt.text_ttt import score_text_output

    gold_by_id = {}
    for line in gold.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            gold_by_id[row["id"]] = row["gold"]

    preds: dict = {}
    for line in pred.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            preds[row["id"]] = row.get("prediction")

    scores = {}
    for doc_id, gold_obj in gold_by_id.items():
        prediction = preds.get(doc_id)
        if prediction is None:
            scores[doc_id] = 0.0
            continue
        score = score_text_output(json.dumps(prediction), json.dumps(gold_obj))
        scores[doc_id] = score.micro_f1 if score.valid_json else 0.0
    return scores


def sign_test(deltas: list[float]) -> tuple[int, int, int, float]:
    """Exact one-sided binomial sign test, ties dropped."""
    import math

    wins = sum(1 for d in deltas if d > 1e-12)
    losses = sum(1 for d in deltas if d < -1e-12)
    ties = len(deltas) - wins - losses
    n = wins + losses
    if not n:
        return wins, losses, ties, 1.0
    total = sum(math.comb(n, k) for k in range(wins, n + 1))
    return wins, losses, ties, total / (2 ** n)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", required=True,
                        help="your labeled JSONL (id/text/gold per line)")
    parser.add_argument("--out-dir", default="my-run")
    parser.add_argument("--train-k", type=int, default=20,
                        help="how many documents to adapt on (rest held out)")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--samples", type=int, default=1,
                        help="decode pool for BOTH arms (1 = greedy). Kept "
                             "matched on purpose: running the adapted arm "
                             "with a richer decode than its baseline credits "
                             "adaptation with the decode difference.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out = pathlib.Path(args.out_dir)
    pkg = out / "package"
    run([sys.executable, str(HERE / "make_challenge.py"), "split",
         "--docs", args.docs, "--train-k", str(args.train_k),
         "--seed", str(args.seed), "--out-dir", str(pkg)], "split")

    common = ["--train", str(pkg / "train.jsonl"),
              "--holdout", str(pkg / "holdout.jsonl"),
              "--model", args.model, "--samples", str(args.samples),
              "--seed", "1", "--allow-unpinned"]
    if args.device:
        common += ["--device", args.device]

    run([sys.executable, str(HERE / "run_challenge.py"), *common,
         "--out-dir", str(out / "baseline"), "--kshot"], "baseline arm")
    run([sys.executable, str(HERE / "run_challenge.py"), *common,
         "--out-dir", str(out / "adapted")], "adapted arm")

    gold = pkg / "gold_holdout.jsonl"
    base = score_arm(out / "baseline" / "predictions.jsonl", gold)
    adapted = score_arm(out / "adapted" / "predictions.jsonl", gold)

    ids = sorted(set(base) & set(adapted))
    deltas = [adapted[i] - base[i] for i in ids]
    mean_base = sum(base[i] for i in ids) / len(ids)
    mean_adapted = sum(adapted[i] for i in ids) / len(ids)
    wins, losses, ties, p_value = sign_test(deltas)

    print("\n" + "=" * 68)
    print("PAIRED RESULT ON YOUR DOCUMENTS")
    print("=" * 68)
    print(f"{'document':<28}{'prompted':>10}{'adapted':>10}{'delta':>12}")
    for doc_id in ids:
        d = adapted[doc_id] - base[doc_id]
        print(f"{str(doc_id)[:28]:<28}{base[doc_id]:>10.4f}"
              f"{adapted[doc_id]:>10.4f}{d:>+12.4f}")
    print("-" * 68)
    print(f"{'mean micro-F1':<28}{mean_base:>10.4f}{mean_adapted:>10.4f}"
          f"{mean_adapted - mean_base:>+12.4f}")
    print(f"\nsign test: {wins}W / {losses}L / {ties}T over {len(ids)} "
          f"held-out documents (one-sided p = {p_value:.4g})")
    print(f"adapted on {args.train_k} of your documents; decode matched at "
          f"samples={args.samples} on both arms")

    print("\nHOW TO READ THIS, honestly:")
    print("  - Not blind: your gold is on this machine and was read to score.")
    print("  - No preregistered bar: nothing here was fixed in advance, and")
    print("    you can re-run until you like the number. So could we.")
    print("  - One corpus, one schema, one run: a data point, not a benchmark.")
    if mean_base >= 0.98:
        # The most useful thing this script can tell some readers is that
        # they do not need it. A saturated baseline means plain prompting
        # already solves their extraction, and no adaptation delta is
        # available to win -- saying anything else here would be selling.
        print("\n  Your prompted baseline is already at "
              f"{mean_base:.4f}. Plain prompting solves these documents,")
        print("  so there is no headroom for adaptation to win and nothing")
        print("  here for you to buy. That is the honest read, and it is the")
        print("  same one our own results table gives: on our synthetic")
        print("  corpus a frontier model and the cheapest API tier both")
        print("  score 1.00 by prompting. Adaptation is worth measuring where")
        print("  the prompted baseline is NOT saturated -- messier documents,")
        print("  more fields, tighter output-format requirements.")
    elif mean_adapted - mean_base <= 0:
        print("\n  Your delta is <= 0: adaptation did not help on your data.")
        print("  That is a real result and we would rather you saw it here")
        print("  than after a purchase. If you send it to us we will publish")
        print("  it -- see CHALLENGES.md.")
    print("\n  The version that is evidence -- blind, single submission, gold")
    print("  you keep, published either way -- is CHALLENGES.md, terms at")
    print("  docs/research/CHALLENGE_TERMS.md.")
    print(f"\nartifacts: {out}/  (predictions, adapter, manifests per arm)")
    print("  NOTE: these runs pass --allow-unpinned, so their manifests are")
    print("  NOT valid challenge deliverables. A real challenge pins the base")
    print("  checkpoint by immutable revision + sha256, and the runner refuses")
    print("  to emit an unpinned manifest without that flag. This is a")
    print("  self-run on your own data; nothing here is a submission.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
