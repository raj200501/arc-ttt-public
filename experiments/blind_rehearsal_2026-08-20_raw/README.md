# Blind-rehearsal raw material — re-score it yourself

Everything needed to reproduce the paired-baseline result in
`../blind_rehearsal_baseline_2026-08-21.json` without trusting any
summary we wrote:

| file | what it is |
|---|---|
| `train.jsonl` | the 20 labelled pairs both arms were given |
| `holdout.jsonl` | the 30 held-out documents, **text and id only** |
| `gold_holdout.jsonl` | the withheld gold labels |
| `predictions_prompted_greedy.jsonl` | raw output, prompted arm |
| `predictions_adapted_greedy.jsonl` | raw output, adapted arm |

Both arms are the same base checkpoint (`Qwen/Qwen2.5-0.5B-Instruct`,
CPU/fp32) at a **matched greedy decode**. The prompted arm carries the
20 training pairs as in-context demonstrations; the adapted arm was
trained on those same 20 and carries the same prompt.

Re-score either arm with the pinned scorer:

```bash
python3 scripts/make_challenge.py score \
    --pred experiments/blind_rehearsal_2026-08-20_raw/predictions_prompted_greedy.jsonl \
    --gold experiments/blind_rehearsal_2026-08-20_raw/gold_holdout.jsonl
```

Expected: prompted **0.7836** (3 of 30 unparseable, scored 0), adapted
**0.8833** (0 unparseable). Recompute the whole verdict, including the
sign test that made this a FAIL, with:

```bash
python3 scripts/bank_rehearsal_baseline.py \
    --baseline experiments/blind_rehearsal_2026-08-20_raw/predictions_prompted_greedy.jsonl \
    --adapted  experiments/blind_rehearsal_2026-08-20_raw/predictions_adapted_greedy.jsonl \
    --gold     experiments/blind_rehearsal_2026-08-20_raw/gold_holdout.jsonl \
    --out /tmp/recheck.json
```

## What this corpus is, and is not

The 50 waybills were **authored by an adversarial AI agent that we
operated ourselves**, on the same host, in the same working session. The
gold was withheld in process and hash-committed before the submission —
procedural blindness, **not third-party custody**. This is a
transfer signal and a protocol rehearsal. It is not a customer's
documents and it is not an independent replication.

It is published in full anyway, because a result nobody can re-score is
a claim, not evidence. `gold_holdout.jsonl` sha256
`13e9cc7f9955bff22997b7d95802c5b65531713f3f881e631828665b8797c2ef`.
