#!/usr/bin/env python3
"""Addendum I: a hosted model against our adapted 0.5B, on REAL-SHAPED documents.

Every delta in this repository is a within-model comparison: an adapted
0.5B against the same 0.5B's own prompt. **No buyer has ever faced that
choice.** The comparison a buyer would actually make -- us against the
hosted model they would otherwise deploy -- has been run exactly twice,
and both times on the synthetic novel-schema corpus, where we already
knew it comes out saturated at 1.00 (`novel_cheaptier_baseline_2026-08-19`,
`novel_frontier_baseline_2026-08-16`).

The one realistic corpus we have -- 30 held-out freight waybills, gold
published, scorer pinned -- **has never seen a hosted model.** That is
the single measurement most likely to decide whether there is a product,
it costs under a dollar, and it was noticed by an outside reader rather
than by us.

This script runs it. Same 30 documents, same 20 training pairs as the
in-context demonstrations, same pinned scorer, temperature 0, one sample
per document, raw predictions stored.

The API key is read from a file OUTSIDE the repository and is never
stored in the artifact, never logged, and never placed in an argument
that would reach a process list.

    PYTHONPATH=src python3 scripts/run_market_baseline_waybills.py \\
        --key-file ~/.gemini_key --model gemini-3.5-flash-lite \\
        --out experiments/waybill_market_baseline_<model>_<date>.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
import urllib.error
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

RAW = REPO / "experiments" / "blind_rehearsal_2026-08-20_raw"
# Banked, matched-greedy, same 30 documents. From
# experiments/blind_rehearsal_baseline_2026-08-21.json.
OUR_PROMPTED = 0.7836
OUR_ADAPTED = 0.8833


def load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_contents(train: list[dict], document: str, matched_turns: bool,
                   schema_fields: list[str] | None = None):
    """The same demonstrations our k-shot arm receives.

    Deliberately NOT a hand-tuned prompt. Giving the hosted model a
    better prompt than our own baseline got would make the comparison
    flatter us; giving it a worse one would be the more tempting error.

    ``matched_turns=True`` mirrors ``arcttt.text_ttt.text_task_to_messages``
    exactly: one user turn per demonstration document, one assistant turn
    per gold object, then the test document as a final user turn -- the
    structure our own k-shot arm was given, with no added instruction.
    ``matched_turns=False`` packs the same pairs into a single turn with a
    one-line instruction. Both are run and both are banked, because a
    reader is entitled to ask whether the turn structure, rather than the
    model, produced the difference.
    """
    if schema_fields:
        # Addendum J's fourth arm: what a real deployment would actually
        # send instead of 20 demonstrations -- the tenant's field list, in
        # one short instruction. ~50 tokens against ~4,300.
        instruction = ("Extract these fields from the document and return "
                       "them as a single JSON object with exactly these "
                       "keys, no others:\n"
                       + "\n".join(f"- {f}" for f in schema_fields))
        contents = []
        for pair in train:
            contents.append({"role": "user",
                             "parts": [{"text": pair["text"]}]})
            contents.append({"role": "model", "parts": [
                {"text": json.dumps(pair["gold"], sort_keys=True)}]})
        contents.append({"role": "user",
                         "parts": [{"text": instruction + "\n\n" + document}]})
        return contents
    if matched_turns:
        contents = []
        for pair in train:
            contents.append({"role": "user",
                             "parts": [{"text": pair["text"]}]})
            contents.append({"role": "model", "parts": [
                {"text": json.dumps(pair["gold"], sort_keys=True)}]})
        contents.append({"role": "user", "parts": [{"text": document}]})
        return contents
    parts = ["Extract the fields as JSON. Follow the examples exactly.", ""]
    for pair in train:
        parts.append(pair["text"])
        parts.append(json.dumps(pair["gold"], sort_keys=True))
        parts.append("")
    parts.append(document)
    return [{"role": "user", "parts": [{"text": "\n".join(parts)}]}]


def call_gemini(key: str, model: str, contents: list, attempts: int = 6) -> str:
    """One completion, with backoff on rate limits.

    A 429 partway through must not become a missing document: an arm
    scored on 24 of 30 documents because the sixth call was throttled
    would be a silently truncated comparison, which is exactly the defect
    class this repository keeps finding in its own numbers. Every document
    is retried until it succeeds or the attempts are exhausted, and an
    exhausted document raises rather than scoring zero -- a zero from a
    network error is not a model failure and must never be recorded as one.
    """
    body = json.dumps({
        "contents": contents,
        "generationConfig": {"temperature": 0, "maxOutputTokens": 1024},
    }).encode()
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    delay = 4.0
    for attempt in range(attempts):
        request = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json",
                     "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.loads(response.read())
            break
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or \
                    attempt == attempts - 1:
                raise
            print(f"    HTTP {error.code}, retry in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay *= 2
        except urllib.error.URLError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2
    try:
        return payload["candidates"][0]["content"]["parts"][0]["text"], payload
    except (KeyError, IndexError):
        # The docstring above promises a network or safety failure is never
        # recorded as a model failure. Returning "" broke that promise by
        # scoring it as a zero. Raise instead, and say why.
        reason = ""
        try:
            reason = payload["candidates"][0].get("finishReason", "")
        except (KeyError, IndexError):
            reason = json.dumps(payload)[:200]
        raise RuntimeError(
            f"no completion returned (finishReason={reason!r}); refusing to "
            "score this as a model zero")


def sign_test(deltas: list[float], label: str) -> dict:
    """Sign test with an EXPLICIT direction.

    `bank_rehearsal_baseline.py`'s version returns P(wins >= observed),
    which is the right tail when the arm of interest is ours. Here the
    hypothesis is reversed -- the hosted model is expected to win -- so
    that tail returns 1.0 beside 0W/14L, which a reader will read as "no
    significant difference" and get the opposite of the truth. Both tails
    are reported, and the one in the observed direction is named.
    """
    wins = sum(1 for d in deltas if d > 1e-12)
    losses = sum(1 for d in deltas if d < -1e-12)
    ties = len(deltas) - wins - losses
    n = wins + losses
    tail = lambda k: (sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n
                      if n else 1.0)  # noqa: E731
    p_ours = tail(wins)
    p_theirs = tail(losses)
    observed = ("ours > theirs" if wins > losses else
                "theirs > ours" if losses > wins else "tied")
    return {"wins": wins, "losses": losses, "ties": ties,
            "compared": label,
            "observed_direction": observed,
            "p_value_ours_greater": p_ours,
            "p_value_theirs_greater": p_theirs,
            "p_value_in_observed_direction": (
                p_theirs if losses > wins else p_ours)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", required=True,
                        help="path OUTSIDE the repo holding the API key")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sleep", type=float, default=4.0)
    parser.add_argument("--k", type=int, default=None,
                        help="number of demonstrations to include "
                             "(default: all 20). Addendum J sweeps this.")
    parser.add_argument("--declare-schema", action="store_true",
                        help="Addendum J fourth arm: send the tenant's "
                             "field list in one instruction instead of "
                             "relying on demonstrations")
    parser.add_argument("--matched-turns", action="store_true",
                        help="mirror text_task_to_messages exactly: one "
                             "turn per demonstration, no added instruction")
    args = parser.parse_args()

    key_path = pathlib.Path(args.key_file).expanduser()
    if REPO in key_path.resolve().parents:
        raise SystemExit("refusing: the key file is inside the repository")
    key = key_path.read_text(encoding="utf-8").strip()

    from arcttt.text_ttt import score_text_output

    train = load_jsonl(RAW / "train.jsonl")
    schema_fields = None
    if args.declare_schema:
        # The tenant's field names, taken from the gold the challenger
        # published -- exactly what a tenant would hand a vendor.
        first = json.loads((RAW / "gold_holdout.jsonl")
                           .read_text(encoding="utf-8").splitlines()[0])
        schema_fields = sorted(first["gold"])
    if args.k is not None:
        if not 0 <= args.k <= len(train):
            raise SystemExit(f"--k must be in [0, {len(train)}]")
        train = train[:args.k]
    holdout = load_jsonl(RAW / "holdout.jsonl")
    gold = {r["id"]: r["gold"] for r in load_jsonl(RAW / "gold_holdout.jsonl")}
    ours = json.loads(
        (REPO / "experiments" /
         "blind_rehearsal_baseline_2026-08-21.json").read_text("utf-8"))
    per_doc = ours["per_doc"]
    banked_adapted = ours["adapted_greedy"]["mean_micro_f1"]
    banked_prompted = ours["baseline_kshot_greedy"]["mean_micro_f1"]
    if (banked_adapted, banked_prompted) != (OUR_ADAPTED, OUR_PROMPTED):
        raise SystemExit(
            f"our banked arms have moved ({banked_prompted} / "
            f"{banked_adapted}) but this script still hardcodes "
            f"{OUR_PROMPTED} / {OUR_ADAPTED}. Update the constants "
            "deliberately rather than publishing a stale comparison.")

    rows, invalid, fenced, prompt_tokens, output_tokens = [], 0, [], 0, 0
    for item in holdout:
        doc_id = item["id"]
        text = item["text"]
        raw, payload = call_gemini(
            key, args.model,
            build_contents(train, text, args.matched_turns, schema_fields))
        usage = payload.get("usageMetadata", {})
        prompt_tokens += usage.get("promptTokenCount", 0)
        output_tokens += usage.get("candidatesTokenCount", 0)

        # A markdown fence around otherwise-valid JSON. We strip it -- but
        # `run_challenge.py`, which produced OUR arms, does a bare
        # json.loads and records a null on failure, so our own arms never
        # received this repair. Granting it to the hosted arm alone would
        # be an asymmetry in the hosted model's favour, so every fenced
        # document is named and the un-repaired mean is banked beside the
        # repaired one. An outside reviewer found this.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            fenced.append(doc_id)
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        score = score_text_output(cleaned, json.dumps(gold[doc_id]))
        unrepaired = score_text_output(raw.strip(), json.dumps(gold[doc_id]))
        f1 = score.micro_f1 if score.valid_json else 0.0
        invalid += 0 if score.valid_json else 1
        rows.append({"id": doc_id, "micro_f1": f1,
                     "valid_json": bool(score.valid_json),
                     # canonical-JSON equality, the repo's secondary metric.
                     # micro-F1 folds case; this does not, and the two
                     # disagree on this corpus.
                     "exact": bool(score.exact_match),
                     "micro_f1_without_fence_strip": (
                         unrepaired.micro_f1 if unrepaired.valid_json else 0.0),
                     "prediction": raw})
        print(f"  {doc_id} f1={f1:.3f}"
              f"{'' if score.exact_match else '  (not exact-match)'}",
              flush=True)
        time.sleep(args.sleep)

    mean = sum(r["micro_f1"] for r in rows) / len(rows)
    missing = [r["id"] for r in rows if r["id"] not in per_doc]
    if missing:
        raise SystemExit(
            f"{len(missing)} documents have no banked paired score "
            f"({missing[:3]}); a paired comparison over a shrunken "
            "denominator while n still reports 30 is exactly the silent "
            "truncation this repository keeps finding. Refusing.")
    deltas_adapted = [per_doc[r["id"]]["adapted"] - r["micro_f1"]
                      for r in rows]
    deltas_prompted = [per_doc[r["id"]]["baseline"] - r["micro_f1"]
                       for r in rows]
    assert len(deltas_adapted) == len(rows) == len(deltas_prompted)

    record = {
        "addendum": "I",
        "what": "A hosted model on the 30 held-out freight waybills -- the "
                "comparison a BUYER would make, on the only real-shaped "
                "corpus we have. Every other comparison in this repository "
                "is our 0.5B against its own prompt.",
        "preregistered": "VERDICT.md Addendum I row, committed before this "
                         "arm ran. Readings (a)/(b)/(c) frozen there.",
        "status": "CONTEXT ARM, not a preregistered GATE of the adaptation "
                  "recipe -- it measures the market, not our bar. It is "
                  "nonetheless preregistered because its readings change "
                  "what this company claims.",
        "model": args.model,
        "corpus": "experiments/blind_rehearsal_2026-08-20_raw (30 holdout "
                  "documents, gold published, agent-authored)",
        "n_demonstrations": len(train),
        "schema_declared": bool(schema_fields),
        "declared_fields": schema_fields,
        "demonstration_format": ("matched chat turns, mirroring "
                                 "arcttt.text_ttt.text_task_to_messages"
                                 if args.matched_turns else
                                 "all 20 pairs packed into one user turn "
                                 "with a one-line instruction"),
        "protocol": "the same 20 demonstration pairs our k-shot arm "
                    "received, same order, same documents, temperature 0, "
                    "one sample per document, scored with "
                    "arcttt.text_ttt.score_text_output -- the pinned "
                    "scorer used by every arm in this repository",
        "n": len(rows),
        "hosted_mean_micro_f1": round(mean, 4),
        "hosted_invalid_json": invalid,
        "hosted_exact_match": f"{sum(1 for r in rows if r['exact'])}/{len(rows)}",
        "exact_match_note": "micro-F1 is the primary metric and folds letter "
                            "case; canonical-JSON exact match does not. They "
                            "disagree on this corpus, so BOTH are reported "
                            "and 'every document exactly right' is not a "
                            "statement micro-F1 alone supports.",
        "fence_stripped_documents": fenced,
        "mean_without_fence_strip": round(
            sum(r["micro_f1_without_fence_strip"] for r in rows) / len(rows), 4),
        "fence_strip_note": "OUR arms were produced by run_challenge.py, "
                            "which does a bare json.loads and records a null "
                            "on failure -- they never received this repair. "
                            "Where mean_without_fence_strip differs from the "
                            "headline mean, the headline is repair-dependent "
                            "and must not be cited as the primary number.",
        "api_usage_tokens": {"prompt": prompt_tokens, "output": output_tokens,
                             "note": "recorded so the cost comparison on THIS "
                                     "corpus can be computed; it never had "
                                     "been, and the dollar figures elsewhere "
                                     "in this repository are from the "
                                     "synthetic corpus and do not transfer"},
        "our_prompted_0_5b": OUR_PROMPTED,
        "our_adapted_0_5b": OUR_ADAPTED,
        "hosted_minus_our_adapted": round(mean - OUR_ADAPTED, 4),
        "paired_our_adapted_minus_hosted": {
            "mean_delta": round(sum(deltas_adapted) / len(deltas_adapted), 4),
            "sign_test": sign_test(deltas_adapted, "our adapted minus hosted"),
        },
        "paired_our_prompted_minus_hosted": {
            "mean_delta": round(sum(deltas_prompted) / len(deltas_prompted), 4),
            "sign_test": sign_test(deltas_prompted,
                                   "our prompted minus hosted"),
        },
        "results": rows,
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")
    print(f"\nhosted {args.model}: {mean:.4f}  ({invalid} invalid JSON)")
    print(f"our adapted 0.5B     : {OUR_ADAPTED:.4f}")
    print(f"our prompted 0.5B    : {OUR_PROMPTED:.4f}")
    print(f"paired (ours - hosted): "
          f"{record['paired_our_adapted_minus_hosted']['mean_delta']:+.4f}  "
          f"{record['paired_our_adapted_minus_hosted']['sign_test']}")
    print(f"\nbanked: {args.out}")
    print("Read it against the frozen readings in VERDICT.md Addendum I.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
