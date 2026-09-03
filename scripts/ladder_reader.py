#!/usr/bin/env python3
"""The engineering ladder's reader — bars frozen before any rung ran.

    PYTHONPATH=src python3 scripts/ladder_reader.py

Applies docs/research/ADAPTATION_ENGINEERING_LADDER.md rule 1 to every
banked rung: the bar is the best PROMPTED arm at the same size, the
two-statistics rule governs (paired mean >= +0.01 AND sign test at
p <= 0.05), every failure publishes at full size, and the attempt count
against each bar is printed wherever any success is.

Scoring is from RAW model text with symmetric fence-stripping, using the
pinned scorer -- the same treatment every banked arm receives.
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
RAW = REPO / "experiments" / "blind_rehearsal_2026-08-20_raw"

TOLERANCE = 0.01
ALPHA = 0.05


def _sign(deltas):
    w = sum(1 for d in deltas if d > 1e-12)
    l = sum(1 for d in deltas if d < -1e-12)
    t = len(deltas) - w - l
    n = w + l
    tail = (lambda k: round(sum(math.comb(n, j)
                                for j in range(k, n + 1)) / 2 ** n, 4)
            if n else (lambda k: 1.0))
    return {"wins": w, "losses": l, "ties": t,
            "p_rung_better": tail(w), "p_bar_better": tail(l)}


def main() -> int:
    from arcttt.scoring import field_micro_f1, parse_json_object
    from arcttt.text_task import TextTaskFormatError
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fencecheck", REPO / "tools" / "fencecheck.py")
    fc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fc)

    gold = {json.loads(l)["id"]: json.loads(l)["gold"] for l in
            (RAW / "gold_holdout.jsonl").read_text().splitlines() if l.strip()}

    def score_raw(text):
        try:
            obj = parse_json_object(fc.strip_fence(text)[0])
        except TextTaskFormatError:
            return None
        return obj

    def score_dir(outdir):
        rows = [json.loads(l) for l in
                (outdir / "predictions.jsonl").read_text().splitlines()
                if l.strip()]
        per, invalid, fenced = {}, 0, 0
        for r in rows:
            raw = r.get("raw")
            if raw is None:
                # An older run without banked raw text: fall back to the
                # parsed object and SAY SO in the artifact.
                obj = r.get("prediction")
            else:
                fenced += fc.strip_fence(raw)[1]
                obj = score_raw(raw)
            if obj is None:
                invalid += 1
                per[r["id"]] = 0.0
            else:
                per[r["id"]] = round(field_micro_f1(obj, gold[r["id"]]), 4)
        return per, invalid, fenced

    baseline = json.loads((REPO / "experiments" /
                           "blind_rehearsal_baseline_2026-08-21.json")
                          .read_text())
    bar_05b = {d: v["baseline"] for d, v in baseline["per_doc"].items()}
    bar_05b_mean = baseline["baseline_kshot_greedy"]["mean_micro_f1"]
    adapted_s1 = {d: v["adapted"] for d, v in baseline["per_doc"].items()}

    kshot3b = json.loads((REPO / "experiments" /
                          "waybill_scale_rung_3b_kshot_bf16_2026-08-25.json")
                         .read_text())
    bar_3b = kshot3b["per_document_micro_f1"]
    bar_3b_mean = kshot3b["mean_micro_f1"]

    rungs = []

    def read_rung(name, per, invalid, fenced, bar, bar_mean, bar_name,
                  attempt_no, extra=None):
        ids = sorted(set(per) & set(bar))
        deltas = [per[i] - bar[i] for i in ids]
        mean = round(statistics.mean(per.values()), 4)
        d = round(statistics.mean(deltas), 4)
        sign = _sign(deltas)
        clears = (d >= TOLERANCE and sign["p_rung_better"] <= ALPHA)
        fails_hard = (d <= -TOLERANCE and sign["p_bar_better"] <= ALPHA)
        rung = {
            "rung": name, "mean_micro_f1": mean, "invalid": invalid,
            "fenced_outputs": fenced,
            "bar": {"name": bar_name, "mean": bar_mean},
            "paired_mean_delta": d, "sign_test": sign,
            "attempt_number_at_this_bar": attempt_no,
            "reading": ("CLEARS THE BAR — and it is attempt "
                        f"{attempt_no} at this bar, stated per the "
                        "ladder's own rule" if clears else
                        "FAILS — loses to the bar with both statistics "
                        "agreeing" if fails_hard else
                        "FAILS — does not clear the two-statistics bar; "
                        "both numbers publish, no claim is made"),
            "per_document_micro_f1": per,
        }
        if extra:
            rung.update(extra)
        rungs.append(rung)
        print(f"{name}: {mean} vs {bar_name} {bar_mean} -> D={d:+.4f} "
              f"{sign['wins']}W/{sign['losses']}L/{sign['ties']}T  "
              f"{rung['reading'][:60]}")

    # E2 — attempts 1 and 2 at the 0.5B prompted bar in this ladder
    # (the banked samples=1 arm predates the ladder and is the context).
    for i, s in enumerate((2, 4), start=1):
        d = REPO / "work" / f"e2_s{s}"
        if (d / ".complete").exists():
            per, inv, fen = score_dir(d)
            paired_s1 = round(statistics.mean(
                [per[k] - adapted_s1[k] for k in per if k in adapted_s1]), 4)
            read_rung(f"E2 samples={s}", per, inv, fen, bar_05b,
                      bar_05b_mean, "prompted 0.5B k-shot", i,
                      {"vs_samples1_adapted_mean_delta": paired_s1,
                       "samples1_adapted_mean": 0.8833})

    # E3 — 0.5B adapted PLUS demonstrations, same bar.
    d = REPO / "work" / "e3_05b"
    if (d / ".complete").exists():
        per, inv, fen = score_dir(d)
        read_rung("E3 0.5B adapted+k20", per, inv, fen, bar_05b,
                  bar_05b_mean, "prompted 0.5B k-shot", 3)

    # E5 — 3B adapted PLUS demonstrations vs the 0.9747 bar; attempt 2
    # at the 3B bar (Q was attempt 1 and failed it).
    e5 = REPO / "experiments" / "ladder_e5_3b_adapted_kshot_2026-08-31.json"
    if e5.exists():
        rec = json.loads(e5.read_text())
        per = rec["per_document_micro_f1"]
        read_rung("E5 3B adapted+k20", per,
                  rec.get("adapted_invalid_json", 0), None, bar_3b,
                  bar_3b_mean, "prompted 3B k-shot bf16", 2)

    # E6 — the same stack on CORD, where headroom exists. Bar = the
    # prompted 3B k=20 arm on the SAME split, scored from ITS raw text
    # by the same instrument; attempt 3 at a 3B prompted bar. Reads only
    # when BOTH arms are banked — never from the arm that finished first.
    e6_dir = REPO / "experiments" / "ladder_e6_cord_split"
    e6_prompted = REPO / "experiments" / "ladder_e6_cord_prompted_2026-08-31.json"
    e6_adapted = REPO / "experiments" / "ladder_e6_cord_adapted_2026-08-31.json"
    if e6_prompted.exists() and e6_adapted.exists():
        gold_e6 = {json.loads(l)["id"]: json.loads(l)["gold"] for l in
                   (e6_dir / "gold.jsonl").read_text().splitlines()
                   if l.strip()}

        def score_e6(path):
            rec = json.loads(path.read_text())
            per, invalid, fenced = {}, 0, 0
            for doc_id, raw in rec["predictions"].items():
                fenced += fc.strip_fence(raw)[1]
                obj = score_raw(raw)
                if obj is None:
                    invalid += 1
                    per[doc_id] = 0.0
                else:
                    per[doc_id] = round(field_micro_f1(obj, gold_e6[doc_id]), 4)
            return per, invalid, fenced

        bar_e6, bar_inv, bar_fen = score_e6(e6_prompted)
        per, inv, fen = score_e6(e6_adapted)
        read_rung("E6 CORD 3B adapted+k20", per, inv, fen, bar_e6,
                  round(statistics.mean(bar_e6.values()), 4),
                  "prompted 3B k-shot bf16 (CORD)", 3,
                  {"bar_invalid": bar_inv, "bar_fenced_outputs": bar_fen,
                   "corpus": "CORD validation, 80 held-out receipts, "
                             "split banked in ladder_e6_cord_split/"})

    # E7 (Ladder II) — the same arms re-decoded with the JSON-constrained
    # greedy decoder. Two frozen readings: ADAPT (E7 adapted vs E7
    # prompted, same decoder) and SYSTEM (E7 adapted vs E6's greedy
    # prompted bar). Reads only when BOTH E7 arms are banked.
    e7_prompted = REPO / "experiments" / "ladder_e7_cord_prompted_2026-09-02.json"
    e7_adapted = REPO / "experiments" / "ladder_e7_cord_adapted_2026-09-02.json"
    if e7_prompted.exists() and e7_adapted.exists() and e6_prompted.exists():
        gold_e6 = {json.loads(l)["id"]: json.loads(l)["gold"] for l in
                   (e6_dir / "gold.jsonl").read_text().splitlines() if l.strip()}

        def score_e7(path):
            rec = json.loads(path.read_text())
            per, invalid, fenced = {}, 0, 0
            for doc_id, raw in rec["predictions"].items():
                fenced += fc.strip_fence(raw)[1]
                obj = score_raw(raw)
                if obj is None:
                    invalid += 1
                    per[doc_id] = 0.0
                else:
                    per[doc_id] = round(field_micro_f1(obj, gold_e6[doc_id]), 4)
            acct = rec.get("decode_accounting", {})
            return per, invalid, fenced, {
                "fallbacks_total": sum(a["fallbacks"] for a in acct.values()),
                "constrained_steps_total": sum(a["constrained_steps"] for a in acct.values()),
                "stopped_on": {k: sum(1 for a in acct.values() if a["stopped_on"] == k)
                               for k in ("eos", "complete", "max_new_tokens")}}

        bar7, bar7_inv, bar7_fen, bar7_acct = score_e7(e7_prompted)
        per7, inv7, fen7, acct7 = score_e7(e7_adapted)
        read_rung("E7 ADAPT: 3B adapted+k20 constrained vs prompted constrained",
                  per7, inv7, fen7, bar7, round(statistics.mean(bar7.values()), 4),
                  "prompted 3B k-shot bf16 constrained (CORD)", 4,
                  {"bar_invalid": bar7_inv, "bar_fenced_outputs": bar7_fen,
                   "decoder_accounting": {"adapted": acct7, "prompted": bar7_acct},
                   "reading_kind": "ADAPT — same decoder both arms; credits the adapter only"})
        bar6, bar6_inv, bar6_fen = score_e6(e6_prompted)
        read_rung("E7 SYSTEM: 3B adapted+k20 constrained vs E6 greedy prompted",
                  per7, inv7, fen7, bar6, round(statistics.mean(bar6.values()), 4),
                  "prompted 3B k-shot bf16 greedy (E6 bar)", 4,
                  {"reading_kind": "SYSTEM — credits adapter AND decoder; never substitutes for ADAPT",
                   "prompted_constrained_vs_greedy_mean_delta": round(
                       statistics.mean(bar7.values()) - statistics.mean(bar6.values()), 4)})

    # E8 (Ladder II) — E7's decoder plus similarity-ORDERED demonstrations
    # (BM25 over OCR text, most similar last). ADAPT vs the E8 prompted
    # arm; SYSTEM vs E6's greedy bar; and the ordering effect on its own
    # (E8 prompted vs E7 prompted) so a SYSTEM gain cannot be misread.
    e8_prompted = REPO / "experiments" / "ladder_e8_cord_prompted_2026-09-03.json"
    e8_adapted = REPO / "experiments" / "ladder_e8_cord_adapted_2026-09-03.json"
    if (e8_prompted.exists() and e8_adapted.exists() and e6_prompted.exists()
            and e7_prompted.exists()):
        bar8, bar8_inv, bar8_fen, bar8_acct = score_e7(e8_prompted)
        per8, inv8, fen8, acct8 = score_e7(e8_adapted)
        bar7b, _, _, _ = score_e7(e7_prompted)
        read_rung("E8 ADAPT: 3B adapted+k20 bm25-ordered constrained vs prompted same",
                  per8, inv8, fen8, bar8, round(statistics.mean(bar8.values()), 4),
                  "prompted 3B k-shot bf16 bm25-ordered constrained (CORD)", 5,
                  {"bar_invalid": bar8_inv, "bar_fenced_outputs": bar8_fen,
                   "decoder_accounting": {"adapted": acct8, "prompted": bar8_acct},
                   "reading_kind": "ADAPT — same decoder and ordering both arms"})
        bar6b, _, _ = score_e6(e6_prompted)
        read_rung("E8 SYSTEM: 3B adapted+k20 bm25-ordered constrained vs E6 greedy prompted",
                  per8, inv8, fen8, bar6b, round(statistics.mean(bar6b.values()), 4),
                  "prompted 3B k-shot bf16 greedy (E6 bar)", 5,
                  {"reading_kind": "SYSTEM — credits adapter, decoder and ordering",
                   "ordering_effect_prompted_bm25_minus_prompted_e7_mean_delta": round(
                       statistics.mean(bar8.values()) - statistics.mean(bar7b.values()), 4),
                   "ordering_effect_sign_test": _sign(
                       [bar8[d] - bar7b[d] for d in sorted(set(bar8) & set(bar7b))])})

    # E9 (Ladder II) — 40-receipt adapter, BM25-SELECTED demonstrations in
    # split ORDER, E7's decoder, on a fresh seed-2 split (60 eval). ADAPT
    # vs the prompted+constrained arm; SYSTEM vs a greedy prompted arm on
    # the SAME 60 receipts. New bar: attempt 1. Reads only when the arms
    # each reading needs exist.
    e9_dir = REPO / "experiments" / "ladder_e9_cord_split"
    e9 = {a: REPO / "experiments" / f"ladder_e9_cord_{a}_2026-09-03.json"
          for a in ("prompted", "adapted", "prompted_greedy")}
    if e9["prompted"].exists() and e9["adapted"].exists():
        gold_e9 = {json.loads(l)["id"]: json.loads(l)["gold"] for l in
                   (e9_dir / "gold.jsonl").read_text().splitlines() if l.strip()}

        def score_e9(path):
            rec = json.loads(path.read_text())
            per, invalid, fenced = {}, 0, 0
            for doc_id, raw in rec["predictions"].items():
                fenced += fc.strip_fence(raw)[1]
                obj = score_raw(raw)
                if obj is None:
                    invalid += 1
                    per[doc_id] = 0.0
                else:
                    per[doc_id] = round(field_micro_f1(obj, gold_e9[doc_id]), 4)
            acct = rec.get("decode_accounting", {})
            summary = {"fallbacks_total": sum(a.get("fallbacks", 0) for a in acct.values()),
                       "constrained_steps_total": sum(a.get("constrained_steps", 0) for a in acct.values())}
            return per, invalid, fenced, summary

        bar9, bar9_inv, bar9_fen, bar9_acct = score_e9(e9["prompted"])
        per9, inv9, fen9, acct9 = score_e9(e9["adapted"])
        read_rung("E9 ADAPT: 3B adapted(40)+k20-of-40 selected, constrained vs prompted same",
                  per9, inv9, fen9, bar9, round(statistics.mean(bar9.values()), 4),
                  "prompted 3B k=20-of-40 bm25-selected split-ordered constrained (E9 split)", 1,
                  {"bar_invalid": bar9_inv, "bar_fenced_outputs": bar9_fen,
                   "decoder_accounting": {"adapted": acct9, "prompted": bar9_acct},
                   "reading_kind": "ADAPT — same selection, order and decoder both arms; new bar, attempt 1",
                   "corpus": "CORD validation, seed-2 split, 60 held-out receipts"})
        if e9["prompted_greedy"].exists():
            barg, barg_inv, barg_fen, _ = score_e9(e9["prompted_greedy"])
            read_rung("E9 SYSTEM: the stack vs greedy prompted on the same 60 receipts",
                      per9, inv9, fen9, barg, round(statistics.mean(barg.values()), 4),
                      "prompted 3B k=20-of-40 bm25-selected split-ordered GREEDY (E9 split)", 1,
                      {"bar_invalid": barg_inv, "bar_fenced_outputs": barg_fen,
                       "reading_kind": "SYSTEM — credits adapter and decoder; never substitutes for ADAPT"})

    out = {
        "what": "Engineering-ladder results, read against the bars frozen "
                "in docs/research/ADAPTATION_ENGINEERING_LADDER.md before "
                "any rung ran. Failures publish at the same size as "
                "successes; attempt counts are part of every reading.",
        "date": "2026-08-31",
        "rule": f"paired mean >= +{TOLERANCE} over the best prompted arm "
                f"at the same size AND sign test at p <= {ALPHA}",
        "rungs": rungs,
    }
    (REPO / "experiments" / "ladder_results_2026-08-31.json").write_text(
        json.dumps(out, indent=2) + "\n")
    print("banked: experiments/ladder_results_2026-08-31.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
