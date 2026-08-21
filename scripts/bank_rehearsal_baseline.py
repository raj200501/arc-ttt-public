#!/usr/bin/env python3
"""Bank the blind-rehearsal PAIRED BASELINE against its preregistered bar.

The rehearsal published an adapted number (0.8792) with nothing to
subtract from it, so it could not say what ADAPTATION contributed -- the
one comparison every gate row in VERDICT.md is built on. This banks the
missing pair.

The bar was frozen and published in VERDICT.md BEFORE these arms ran:
paired seed-mean delta >= +5.0 micro-F1 with the sign test agreeing,
publishing either way including negative. This script applies that rule
and does not decide anything else.

Both arms run at a MATCHED greedy decode. The banked 0.8792 used voted
decode (samples=5) and the k-shot arm cannot run voted -- 20
demonstration pairs inflate the prompt past memory -- so pairing a voted
adapted arm against a greedy baseline would have credited adaptation
with the decode difference. 0.8792 stays reported separately as the
voted number it is.

    python3 scripts/bank_rehearsal_baseline.py \
        --baseline <dir>/predictions.jsonl \
        --adapted  <dir>/predictions.jsonl \
        --gold     <dir>/gold_holdout.jsonl \
        --out experiments/blind_rehearsal_baseline_2026-08-21.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

BAR = 0.05  # +5.0 micro-F1, frozen in VERDICT.md before these arms ran


def load_gold(path: pathlib.Path) -> dict:
    gold = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            gold[row["id"]] = row["gold"]
    return gold


def score_arm(path: pathlib.Path, gold: dict) -> tuple[dict, int]:
    from arcttt.text_ttt import score_text_output

    preds: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            preds[row["id"]] = row.get("prediction")

    scores, invalid = {}, 0
    for doc_id, gold_obj in gold.items():
        prediction = preds.get(doc_id)
        if prediction is None:
            scores[doc_id] = 0.0
            invalid += 1
            continue
        score = score_text_output(json.dumps(prediction), json.dumps(gold_obj))
        scores[doc_id] = score.micro_f1 if score.valid_json else 0.0
        invalid += 0 if score.valid_json else 1
    return scores, invalid


def sign_test(deltas: list[float]) -> dict:
    wins = sum(1 for d in deltas if d > 1e-12)
    losses = sum(1 for d in deltas if d < -1e-12)
    ties = len(deltas) - wins - losses
    n = wins + losses
    p = 1.0
    if n:
        p = sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)
    return {"wins": wins, "losses": losses, "ties": ties, "p_value": p}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--adapted", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    gold = load_gold(pathlib.Path(args.gold))
    base, base_invalid = score_arm(pathlib.Path(args.baseline), gold)
    adapt, adapt_invalid = score_arm(pathlib.Path(args.adapted), gold)

    ids = sorted(set(base) & set(adapt))
    deltas = [adapt[i] - base[i] for i in ids]
    mean_base = sum(base[i] for i in ids) / len(ids)
    mean_adapt = sum(adapt[i] for i in ids) / len(ids)
    mean_delta = mean_adapt - mean_base
    st = sign_test(deltas)

    sd = math.sqrt(sum((d - mean_delta) ** 2 for d in deltas) / (len(deltas) - 1))
    sys.path.insert(0, str(REPO / "scripts"))
    from novel_schema_summary import t95  # the one estimator in this repo
    half = t95(len(deltas) - 1) * sd / math.sqrt(len(deltas))

    clears_bar = mean_delta >= BAR
    sign_agrees = st["p_value"] < 0.05 and st["wins"] > st["losses"]
    verdict = "PASS" if (clears_bar and sign_agrees) else "FAIL"

    record = {
        "what": "Blind-rehearsal PAIRED BASELINE — the arm the rehearsal "
                "lacked. Bar frozen and published in VERDICT.md before "
                "these arms ran.",
        "rule": {"bar_delta": BAR, "requires": "seed-mean delta >= bar AND "
                                               "sign test agrees",
                 "publishes_either_way": True},
        "decode": {"both_arms": "greedy (samples=1), matched",
                   "note": "the banked 0.8792 used voted decode (samples=5); "
                           "the k-shot arm cannot run voted, so pairing "
                           "voted-vs-greedy would credit adaptation with the "
                           "decode difference"},
        "baseline_kshot_greedy": {"mean_micro_f1": round(mean_base, 4),
                                  "invalid_or_missing": base_invalid},
        "adapted_greedy": {"mean_micro_f1": round(mean_adapt, 4),
                           "invalid_or_missing": adapt_invalid},
        "paired": {"mean_delta": round(mean_delta, 4),
                   "ci95": [round(mean_delta - half, 4),
                            round(mean_delta + half, 4)],
                   "n": len(ids), "sign_test": st},
        "verdict": verdict,
        "why": {
            "clears_mean_bar": clears_bar,
            "sign_test_agrees": sign_agrees,
            "rule": "both required; the two-statistics rule exists so a mean "
                    "carried by a few documents cannot pass alone",
            "ties": st["ties"],
            "ties_both_perfect": sum(
                1 for i in ids
                if abs(adapt[i] - base[i]) < 1e-12 and base[i] == 1.0),
            "baseline_zero_docs": sum(1 for i in ids if base[i] == 0.0),
            "share_of_delta_from_baseline_zero_docs": round(
                sum(adapt[i] - base[i] for i in ids if base[i] == 0.0)
                / sum(adapt[i] - base[i] for i in ids), 4)
            if abs(sum(adapt[i] - base[i] for i in ids)) > 1e-12 else None,
        },
        "per_doc": {i: {"baseline": round(base[i], 4),
                        "adapted": round(adapt[i], 4),
                        "delta": round(adapt[i] - base[i], 4)} for i in ids},
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")

    print(f"baseline (k-shot, greedy): {mean_base:.4f}  "
          f"({base_invalid} invalid/missing)")
    print(f"adapted  (greedy)        : {mean_adapt:.4f}  "
          f"({adapt_invalid} invalid/missing)")
    print(f"paired delta             : {mean_delta:+.4f}  "
          f"CI95 [{mean_delta - half:+.4f}, {mean_delta + half:+.4f}]")
    print(f"sign test                : {st['wins']}W/{st['losses']}L/"
          f"{st['ties']}T  p={st['p_value']:.4g}")
    print(f"bar                      : +{BAR:.2f} (frozen before the arms)")
    print(f"\nVERDICT: {verdict}")
    if verdict == "FAIL":
        print("Published per the preregistered rule: this arm did not clear "
              "its bar, and that is the result.")
    print(f"banked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
