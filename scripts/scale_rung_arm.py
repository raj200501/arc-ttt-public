#!/usr/bin/env python3
"""One rung of the open-model scale ladder, run as a committed script.

Addenda M and N answered the question that matters most to whether there
is a company here -- *does an on-prem buyer need our adaptation, or just a
bigger checkpoint?* -- and both were run from ad-hoc code that was never
committed. The artifacts store their numbers; nothing stores the way the
numbers were produced. `scripts/verification_coverage.py` calls that class
of row ARITHMETIC at best, and for the two rows the remaining claim rests
on, that is not good enough: a reader who wants to re-run them cannot.

This is that code, written once and parameterised over the rung, so every
future rung is produced by the same path as the ones already banked and no
rung can quietly drift from another. It re-runs M and N as a
**reproduction check** before it is trusted on anything new.

Two arms, matching the two already on the page:

  --mode kshot    20 demonstrations in the prompt, no adaptation. Built
                  through `run_challenge.build_task` and
                  `text_task_to_messages`, the same constructors that fed
                  our own arms, so the prompts cannot diverge.
  --mode schema   no demonstrations; the tenant's field list in one
                  instruction. The wording is lifted VERBATIM from
                  `run_market_baseline_waybills.py`'s `--declare-schema`
                  path, which is the arm that took a hosted model to
                  0.8930 and killed our cost argument in Addendum J.

Raw predictions are always stored, so every rung produced here is
PRIMARY-verifiable on the coverage map -- unlike M and N themselves.

    PYTHONPATH=src python3 scripts/scale_rung_arm.py \\
        --model Qwen/Qwen2.5-3B-Instruct --mode schema \\
        --out experiments/waybill_scale_rung_3b_schema_2026-08-25.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import platform
import statistics
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

RAW = REPO / "experiments" / "blind_rehearsal_2026-08-20_raw"

# The same external instance quote every other cost row on this page
# uses, so the pages cannot drift. A price, not a measurement.
INSTANCE_USD_PER_HOUR = 0.290
RATE_DATE = "2026-08-19"

# Banked comparators. Hardcoded deliberately and checked against the
# artifacts at run time: a comparison that silently goes stale is the
# defect this repository keeps finding in its own numbers.
OUR_ADAPTED = 0.8833
OUR_PROMPTED = 0.7836
HOSTED_K0_SCHEMA = 0.8930          # Addendum J, the same instruction
BANKED_RUNGS = {
    # (model, mode) -> (mean_micro_f1, invalid_json, artifact)
    ("Qwen/Qwen2.5-1.5B-Instruct", "kshot"): (
        0.8804, 0, "waybill_scale_rung_1.5b_2026-08-25.json"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "schema"): (
        0.0, 30, "waybill_scale_rung_1.5b_schema_2026-08-25.json"),
}


def _rows(name: str) -> list[dict]:
    return [json.loads(line) for line
            in (RAW / name).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sign_test(deltas: list[float]) -> dict:
    """Both tails, with the observed direction named.

    A one-tailed p reported without its direction reads as "no difference"
    when the arm of interest is the one losing. That has happened here
    before, so both tails are always banked.
    """
    wins = sum(1 for d in deltas if d > 1e-12)
    losses = sum(1 for d in deltas if d < -1e-12)
    ties = len(deltas) - wins - losses
    n = wins + losses

    def tail(k: int) -> float:
        if not n:
            return 1.0
        return sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n

    return {
        "wins_ours": wins, "losses_ours": losses, "ties": ties,
        "observed_direction": ("ours > theirs" if wins > losses
                               else "theirs > ours" if losses > wins
                               else "tied"),
        "p_value_ours_greater": round(tail(wins), 4),
        "p_value_theirs_greater": round(tail(losses), 4),
        "p_value_in_observed_direction": round(
            tail(losses) if losses > wins else tail(wins), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="an OPEN checkpoint an on-prem buyer could run")
    parser.add_argument("--mode", required=True,
                        choices=("kshot", "schema", "schema_kshot"),
                        help="schema_kshot is Addendum R's discriminator: "
                             "the field list AND k demonstrations, so a "
                             "short prompt that still contains an example")
    parser.add_argument("--k", type=int, default=20,
                        help="demonstrations, kshot mode only")
    parser.add_argument("--batch", type=int, default=1,
                        help="batch 1 matches the banked M and N arms. "
                             "Larger batches amortise cost and are the "
                             "fair-comparison arm, but change per-document "
                             "LATENCY, so both are reported.")
    parser.add_argument("--dtype", default="float32",
                        choices=("float32", "bfloat16"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke-test on the first N documents. An arm "
                             "run with this set is marked SMOKE and must "
                             "never be cited as a result.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from arcttt.scoring import field_micro_f1, parse_json_object
    from arcttt.text_task import TextTaskFormatError
    from arcttt.text_ttt import text_task_to_messages

    sys.path.insert(0, str(REPO / "scripts"))
    from run_challenge import build_task  # noqa: E402

    torch.set_num_threads(4)
    train, holdout = _rows("train.jsonl"), _rows("holdout.jsonl")
    gold = {r["id"]: r["gold"] for r in _rows("gold_holdout.jsonl")}
    if args.limit:
        holdout = holdout[:args.limit]

    # Guard the comparators the same way run_market_baseline_waybills.py
    # does: if our banked arms have moved, refuse rather than publish a
    # comparison against a number that is no longer on the page.
    banked = json.loads((REPO / "experiments" /
                         "blind_rehearsal_baseline_2026-08-21.json")
                        .read_text(encoding="utf-8"))
    if (banked["adapted_greedy"]["mean_micro_f1"],
            banked["baseline_kshot_greedy"]["mean_micro_f1"]) != (
            OUR_ADAPTED, OUR_PROMPTED):
        raise SystemExit(
            "our banked arms have moved but this script still hardcodes "
            f"{OUR_ADAPTED} / {OUR_PROMPTED}. Update deliberately.")
    per_doc_ours = banked["per_doc"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # ---- prompts -------------------------------------------------------
    if args.mode == "schema_kshot":
        # Addendum R. The k-shot and schema prompts differ in three ways at
        # once -- examples, length, turn count -- so "demonstrations
        # suppress the fence" is a causal claim with three confounds in it.
        # This cell holds length and turn count near the schema arm while
        # adding examples, which is the only way to tell them apart.
        first = json.loads((RAW / "gold_holdout.jsonl")
                           .read_text(encoding="utf-8").splitlines()[0])
        instruction = ("Extract these fields from the document and return "
                       "them as a single JSON object with exactly these "
                       "keys, no others:\n"
                       + "\n".join(f"- {f}" for f in sorted(first["gold"])))
        demos = []
        for pair in train[:args.k]:
            demos.append({"role": "user", "content": pair["text"]})
            demos.append({"role": "assistant",
                          "content": json.dumps(pair["gold"],
                                                sort_keys=True)})
        prompt_texts = [
            tokenizer.apply_chat_template(
                demos + [{"role": "user",
                          "content": instruction + "\n\n" + r["text"]}],
                tokenize=False, add_generation_prompt=True)
            for r in holdout]
        prompt_note = (f"the tenant's field list AND {args.k} "
                       "demonstration(s). Addendum R's discriminator cell: "
                       "short prompt, examples present.")
    elif args.mode == "kshot":
        task = build_task(train[:args.k], holdout)
        prompt_texts = [
            tokenizer.apply_chat_template(
                [{"role": t.role, "content": t.content}
                 for t in text_task_to_messages(task, i, include_demos=True)],
                tokenize=False, add_generation_prompt=True)
            for i in range(len(holdout))]
        prompt_note = (f"{args.k} demonstrations as chat turns, built through "
                       "run_challenge.build_task and "
                       "arcttt.text_ttt.text_task_to_messages -- the same "
                       "constructors that fed our own k-shot arm, so the "
                       "prompts cannot diverge. No adaptation.")
    else:
        # VERBATIM from run_market_baseline_waybills.py --declare-schema.
        # If that wording ever changes, these two arms stop being the same
        # experiment, so it is asserted rather than trusted.
        first = json.loads((RAW / "gold_holdout.jsonl")
                           .read_text(encoding="utf-8").splitlines()[0])
        schema_fields = sorted(first["gold"])
        instruction = ("Extract these fields from the document and return "
                       "them as a single JSON object with exactly these "
                       "keys, no others:\n"
                       + "\n".join(f"- {f}" for f in schema_fields))
        runner = (REPO / "scripts" / "run_market_baseline_waybills.py"
                  ).read_text(encoding="utf-8")
        for fragment in ("Extract these fields from the document and return ",
                         "them as a single JSON object with exactly these ",
                         "keys, no others:"):
            if fragment not in runner:
                raise SystemExit(
                    "the schema instruction has drifted from "
                    "run_market_baseline_waybills.py; this arm would no "
                    "longer be the same experiment as Addendum J's")
        prompt_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction + "\n\n" + r["text"]}],
                tokenize=False, add_generation_prompt=True)
            for r in holdout]
        prompt_note = ("no demonstrations; the tenant's field list in one "
                       "instruction, wording asserted identical to "
                       "run_market_baseline_waybills.py --declare-schema, "
                       "which is the arm that took the hosted tier to "
                       f"{HOSTED_K0_SCHEMA} in Addendum J")

    encoded = [tokenizer(text, return_tensors="pt").input_ids[0]
               for text in prompt_texts]
    mean_prompt_tokens = statistics.mean(int(t.shape[0]) for t in encoded)

    print(f"{args.model}  mode={args.mode}  dtype={args.dtype}  "
          f"batch={args.batch}", flush=True)
    print(f"mean prompt tokens: {mean_prompt_tokens:.0f}", flush=True)

    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.eval()

    # ---- generate ------------------------------------------------------
    texts: list[str] = []
    seconds = 0.0
    # ---- per-document checkpointing ------------------------------------
    # The execution environment suspends between agent turns and resumes
    # with a fresh process tree, so a run that banks only at completion
    # loses everything at every suspend: the 1.5B bf16 control reached
    # 2/30 twice and died twice, paying for the same documents both
    # times. Generation is greedy and per-document independent, so
    # completed documents checkpoint to a sidecar and are skipped on
    # relaunch -- idempotent per DOCUMENT, not just per arm. Sidecar
    # times were taken live and replay as measurements; wall clock
    # across suspends is meaningless and is not banked.
    ckpt_path = pathlib.Path(str(args.out) + ".ckpt.jsonl")
    ckpt: dict[str, dict] = {}
    # Every row carries the arm's configuration; a checkpoint written
    # under a different model/mode/dtype/k/decode budget is REFUSED, not
    # silently replayed -- resuming after a config change would bank old-
    # config outputs under new-config metadata, which is exactly the
    # dtype mixing the Q contingency forbids. A torn final line (the
    # write was cut mid-row, the one failure the sidecar exists for) is
    # dropped with a notice instead of killing every future resume.
    config_key = (f"{args.model}|{args.mode}|{args.dtype}|k={args.k}|"
                  f"mnt={args.max_new_tokens}")
    if ckpt_path.exists():
        torn = 0
        for line in ckpt_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                # fencecheck: ignore -- a checkpoint line this script
                # wrote itself, not model text; a torn tail is counted.
                row = json.loads(line)
            except json.JSONDecodeError:
                torn += 1
                continue
            if row.get("config", config_key) != config_key:
                raise SystemExit(
                    f"checkpoint {ckpt_path} was written under a different "
                    f"configuration ({row['config']!r} vs {config_key!r}). "
                    "Delete it deliberately; it will not be replayed under "
                    "this arm's metadata.")
            ckpt[row["id"]] = row
        if torn:
            print(f"  checkpoint: dropped {torn} torn line(s)", flush=True)
        print(f"  checkpoint: {len(ckpt)}/{len(encoded)} documents "
              "already generated; skipping them", flush=True)
    replayed_documents = len(ckpt)
    ckpt_handle = ckpt_path.open("a", encoding="utf-8")
    # Per-batch wall clock, kept so a contaminated cost figure can be
    # detected instead of averaged in. The 3B schema arm's mean was
    # inflated 20% by two batches that overlapped an unrelated test suite
    # on the same four cores, and only the log revealed it -- the
    # artifact banked the mean alone. A cost number whose contamination
    # is invisible in its own artifact is not a measurement.
    batch_seconds: list[float] = []
    for start in range(0, len(encoded), args.batch):
        chunk = encoded[start:start + args.batch]
        chunk_rows = holdout[start:start + args.batch]
        if all(r["id"] in ckpt for r in chunk_rows):
            for r in chunk_rows:
                texts.append(ckpt[r["id"]]["text"])
                batch_seconds.append(ckpt[r["id"]]["seconds"])
                seconds += ckpt[r["id"]]["seconds"]
            continue
        width = max(int(t.shape[0]) for t in chunk)
        input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, ids in enumerate(chunk):
            input_ids[row, width - ids.shape[0]:] = ids
            attention[row, width - ids.shape[0]:] = 1
        began = time.monotonic()
        with torch.no_grad():
            out = model.generate(input_ids=input_ids,
                                 attention_mask=attention,
                                 max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=pad_id)
        elapsed = time.monotonic() - began
        seconds += elapsed
        # PER-DOCUMENT, always: replayed checkpoint rows contribute
        # per-document times, so live batches must too, or the two
        # populations sit in batch_seconds in different units and the
        # median (and the cost derived from it) is corrupted at any
        # batch size above 1.
        batch_seconds.extend([elapsed / len(chunk)] * len(chunk))
        texts.extend(
            tokenizer.decode(out[r][width:], skip_special_tokens=True).strip()
            for r in range(len(chunk)))
        done = min(start + args.batch, len(encoded))
        for offset, row_meta in enumerate(chunk_rows):
            ckpt_handle.write(json.dumps({
                "id": row_meta["id"],
                "text": texts[start + offset],
                "seconds": elapsed / len(chunk_rows),
                "config": config_key,
            }) + "\n")
        ckpt_handle.flush()
        print(f"  {done}/{len(encoded)}  {elapsed:6.1f}s", flush=True)

    # ---- score ---------------------------------------------------------
    scores, invalid, predictions, per_document = [], 0, {}, {}
    for row, text in zip(holdout, texts):
        try:
            obj = parse_json_object(text)
        except TextTaskFormatError:
            obj, invalid = None, invalid + 1
        score = 0.0 if obj is None else field_micro_f1(obj, gold[row["id"]])
        scores.append(score)
        per_document[row["id"]] = round(score, 4)
        # Raw text, not the parsed object: a reader must be able to see
        # what the model actually emitted on the 30 invalid documents,
        # which a parsed null would hide.
        predictions[row["id"]] = text
    mean = statistics.mean(scores)
    per_doc_seconds = seconds / len(encoded)

    # Paired against our adapted arm on the same documents.
    paired_ids = [r["id"] for r in holdout if r["id"] in per_doc_ours]
    if len(paired_ids) != len(holdout):
        raise SystemExit(
            f"{len(holdout) - len(paired_ids)} documents have no banked "
            "paired score; a paired test over a shrunken denominator while "
            "n still reads 30 is silent truncation. Refusing.")
    deltas = [per_doc_ours[i]["adapted"] - per_document[i] for i in paired_ids]

    # ---- reproduction check -------------------------------------------
    key = (args.model, args.mode)
    reproduction = None
    if key in BANKED_RUNGS and not args.limit and args.batch == 1 \
            and args.dtype == "float32":
        want_mean, want_invalid, artifact = BANKED_RUNGS[key]
        agrees = (abs(round(mean, 4) - want_mean) <= 0.0001
                  and invalid == want_invalid)
        reproduction = {
            "checked_against": artifact,
            "banked_mean_micro_f1": want_mean,
            "banked_invalid_json": want_invalid,
            "this_run_mean_micro_f1": round(mean, 4),
            "this_run_invalid_json": invalid,
            "reproduces": agrees,
            "why_this_matters": (
                "Addenda M and N were run from code that was never "
                "committed. If this runner does not land on their numbers, "
                "either it is not the same experiment or theirs was not "
                "deterministic -- and in both cases the two rows the "
                "remaining claim rests on are weaker than the page says. "
                "Published either way."),
        }
        print(f"\nreproduction check vs {artifact}: "
              f"{'AGREES' if agrees else 'DISAGREES'} "
              f"({mean:.4f} vs banked {want_mean})", flush=True)

    record = {
        "what": f"{args.model}, {args.mode} arm, on the 30 held-out freight "
                "waybills. An OPEN checkpoint an on-prem buyer could run, "
                "with NO adaptation -- the rival explanation for everything "
                "this project claims.",
        "model": args.model,
        "mode": args.mode,
        # `schema_kshot` DOES consume --k: it builds the prompt from
        # `train[:args.k]` exactly as `kshot` does, and only the schema
        # block differs. Recording 0 for it made the artifact under-report
        # its own configuration, and it did so in the one place that
        # inverts the reading -- Addendum R's whole question is what
        # ONE demonstration does to a schema prompt, and the artifact
        # answering it said there had been no demonstrations.
        #
        # A run whose banked metadata contradicts the prompt it actually
        # sent is not a weaker result, it is a wrong one, and nobody
        # reading the file could have caught it without the token count
        # sitting next to it (196 for pure schema against 401 here).
        "n_demonstrations": 0 if args.mode == "schema" else args.k,
        "prompt_construction": prompt_note,
        "dtype": args.dtype,
        "batch_size": args.batch,
        "max_new_tokens": args.max_new_tokens,
        "n": len(holdout),
        "smoke_only": bool(args.limit),
        "mean_micro_f1": round(mean, 4),
        "invalid_json": invalid,
        "mean_prompt_tokens": round(mean_prompt_tokens, 1),
        "seconds_per_document_amortised": round(per_doc_seconds, 2),
        "wall_clock_seconds_total": round(seconds, 1),
        "cost_per_1k_documents_usd": round(
            per_doc_seconds * 1000 / 3600 * INSTANCE_USD_PER_HOUR, 4),
        "seconds_per_batch": [round(s, 2) for s in batch_seconds],
        # batch_seconds holds PER-DOCUMENT times (live batches are
        # divided by their size at append; replayed rows already are),
        # so the median is per-document as-is. The old /args.batch here
        # divided replayed entries twice at any batch above 1.
        "seconds_per_document_median": round(
            statistics.median(batch_seconds), 2),
        "cost_per_1k_documents_usd_median": round(
            statistics.median(batch_seconds)
            * 1000 / 3600 * INSTANCE_USD_PER_HOUR, 4),
        "timing_integrity": (
            "Per-batch wall clock is banked so a contaminated cost figure "
            "is visible in the artifact rather than only in a log. This "
            "box has four cores; anything else running on them inflates "
            "the mean. Where the mean and the median disagree materially, "
            "the run shared the machine and THE MEDIAN IS THE HONEST "
            "FIGURE -- the mean is inflated, which for a rival's arm is "
            "the direction that flatters this project."),
        "cost_basis": f"${INSTANCE_USD_PER_HOUR}/hour CPU instance, external "
                      f"list price quoted {RATE_DATE}; not a measurement. "
                      "The same quote every cost row on this page uses.",
        "comparators": {
            "our_adapted_0.5b": OUR_ADAPTED,
            "our_prompted_0.5b": OUR_PROMPTED,
            "hosted_k0_plus_schema_addendum_J": HOSTED_K0_SCHEMA,
        },
        "paired_ours_minus_theirs": {
            "mean_delta": round(statistics.mean(deltas), 4),
            "sign_test": sign_test(deltas),
            "note": "positive mean_delta means our adapted 0.5B scored "
                    "higher on average; the sign test is the second "
                    "statistic and can disagree with it.",
        },
        "per_document_micro_f1": per_document,
        "predictions": predictions,
        "resumed_from_checkpoint": replayed_documents > 0,
        "replayed_documents": replayed_documents,
        "resume_note": (
            "This arm resumed from a per-document checkpoint after the "
            "environment reclaimed its process: the replayed documents' "
            "outputs and per-document timings were taken live in an "
            "earlier process and reloaded from the sidecar. Quality is "
            "unaffected (greedy decode, per-document independent). "
            "wall_clock_seconds_total sums timings across processes and "
            "is NOT one machine-session's wall clock."
            if replayed_documents else
            "Single uninterrupted process; the checkpoint sidecar was "
            "written but never replayed."),
        "scope": f"One rung ({args.model}), one corpus, thirty "
                 "agent-authored documents, one seed, "
                 f"batch {args.batch}, CPU, greedy. It says nothing about "
                 "rungs it did not run, and it does not license 'bigger "
                 "models beat adaptation' any more than the reverse.",
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__,
                        "threads": torch.get_num_threads()},
        "reproduction_check": reproduction,
        "verification_level": "PRIMARY -- raw model output is stored for "
                              "every document, so a reader can re-score "
                              "every one against the published gold.",
    }
    ckpt_handle.close()
    out = pathlib.Path(args.out)
    # The sidecar dies with the successful bank. Left in place, the
    # documented reproduction command would replay all 30 documents
    # without ever loading the model and bank a fresh-looking artifact
    # from old outputs -- a vacuous reproduction, which is worse than
    # none because it reads as one.
    ckpt_path.unlink(missing_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(f"\n{args.model} {args.mode}: {mean:.4f}  "
          f"({invalid} invalid JSON)  "
          f"${record['cost_per_1k_documents_usd']}/1k")
    print(f"our adapted 0.5B      : {OUR_ADAPTED}")
    print(f"paired (ours - theirs): "
          f"{record['paired_ours_minus_theirs']['mean_delta']:+.4f}  "
          f"{record['paired_ours_minus_theirs']['sign_test']}")
    print(f"\nbanked: {out}")
    print("Read it against the frozen readings in VERDICT.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
