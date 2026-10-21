# Addendum Y — does the decoder reproduce, and when does it stop? (preregistration)

**Frozen 2026-09-17, before any arm ran.** Addendum X withheld itself
because its determinism gate failed on `microsoft/Phi-3-mini-4k-instruct`:
re-decoding the first five documents, the `model.generate` comparator did
not reproduce the text banked on 2026-09-03 on two of them (the same two
in both of two runs), and the constrained decoder differed from Addendum
V's banked text on a third in one run and matched it in the next. In a
fresh process that third document reproduced twice out of two, and still
did after one intervening `generate` call. So the failure is neither
random nor simple. This addendum measures it directly instead of leaving
it as a gate verdict, and it measures it on a second family so the result
can say whether the dtype is the cause or merely the coincidence.

**Every Phi-3 cell this repository has banked was decoded in bfloat16**
(Addendum T's protocol, line 37; the `decode` string inside the T cell
says `float32` and is the hard-coded metadata T's own erratum already
disclosed). The other four families were decoded in float32 and re-decode
byte-identically. bfloat16 keeps eight bits of mantissa, so a change in
the order of a floating-point reduction — which on this CPU can follow
from thread count, memory layout or kernel selection — flips low bits
that float32 would absorb. That is the hypothesis. It is written here so
it can be wrong in public.

## Design — three (family, dtype) pairs, the same ten documents, one process each

The first **10** receipts of the Addendum V/X holdout, in order
(`cord-000` … `cord-009`), the Addendum S `SCHEMA_INSTRUCTION` verbatim
through the family's own chat template, `max_new_tokens=512`, no clock
in either template (both were checked and banked in
`experiments/generation_configs_2026-09-16.json`).

| pair | why it is here | banked cells to compare against |
|---|---|---|
| **Phi-3-mini, bfloat16** | the family that failed the gate | V's constrained cell; T's generate cell |
| **Qwen2.5-0.5B, bfloat16** | a family known to reproduce 0-of-10 in float32, run in the suspect dtype | none (no bfloat16 cell exists) |
| **Qwen2.5-0.5B, float32** | the control: the known-good pair, same design | V's constrained cell; S's generate cell |

Within ONE process per pair, in this fixed order, every decode
checkpointed and keyed by (document, pass):

1. **Pass A — immediate repeat.** For each document in order, decode with
   the constrained decoder **twice, back to back** (`a1`, `a2`).
2. **Pass B — state dependence.** For each document in order, decode with
   the constrained decoder once more (`b`), now after twenty decodes have
   run in the process.
3. **Pass GA / GB — the plain path.** For each document, `model.generate`
   with the checkpoint's own defaults, exactly the comparator call
   Addenda S/T used (`g1`); then the whole pass again (`g2`).
4. **Thread count** (Phi-3 bfloat16 only, two documents: `cord-004`, the
   one that flaked, and `cord-000`, which never did): the constrained
   decoder at `torch.set_num_threads(4)` and again at `1`, same process.

## Quantities — per pair, counts of 10

- `Y1` — documents where `a1 ≠ a2` (immediate repeat).
- `Y2` — documents where `b ≠ a1` (state dependence).
- `Y3` — documents where `g2 ≠ g1` (the plain path, state dependence).
- `then_vs_now` — documents where `a1` ≠ the banked constrained cell, and
  where `g1` ≠ the banked generate cell (only where a banked cell exists).
- `Y5` — documents (of 2) where the 4-thread and 1-thread decodes differ.

All comparisons are byte-for-byte on the decoded text after the shipped
`tools/fencecheck.py` strip, symmetrically, as every addendum here does.
Per-document texts of every pass are banked, so a stranger can recount.

## Frozen readings — applied by arithmetic in the reader

Per pair:

- **REPRODUCES** — `Y1 = Y2 = Y3 = 0`.
- **IMMEDIATE-REPEAT UNSTABLE** — `Y1 ≥ 1`, whatever else.
- **STATE-DEPENDENT** — `Y1 = 0` and `Y2 + Y3 ≥ 1`.

Across pairs, the dtype question is decided by the two Qwen pairs alone,
because they hold the family fixed:

- **BFLOAT16 IS SUFFICIENT** — Qwen-bfloat16 is not REPRODUCES **and**
  Qwen-float32 is REPRODUCES.
- **BFLOAT16 IS NOT SUFFICIENT ON THIS FAMILY** — both Qwen pairs are
  REPRODUCES. Then Phi-3's instability is not explained by its dtype alone,
  and this addendum says so rather than keeping the hypothesis.
- **NOT A DTYPE STORY** — Qwen-float32 is not REPRODUCES. Then the float32
  reproduction Addendum X reported was a sample, and the sentence "four
  families reproduce" narrows by dated erratum to what this run shows.

Thread count: **THREAD-SENSITIVE** if `Y5 ≥ 1`, else **THREAD-INSENSITIVE
ON THIS SAMPLE**. This reading names a mechanism only if it fires; a null
here rules out one mechanism on two documents and nothing more.

**What may be said afterwards, in the non-flattering direction.** "This
repository's cells reproduce" may be said only about (family, dtype)
pairs that read REPRODUCES here, named as such, and never as an
unqualified sentence about the repository. A pair that fails keeps every
number it has already published and loses only the word *reproducible*.

## Predictions, written before the run

- `Y1 = 0` on all three pairs: back-to-back decodes in one process agree.
  A failure here would mean the decoder is non-deterministic in the plain
  sense, which nothing seen so far suggests.
- Phi-3-bfloat16 reads **STATE-DEPENDENT**: `Y2 + Y3 ≥ 1`. That is the
  gate's flake, reproduced under a design that can see it.
- Phi-3-bfloat16 is **THREAD-SENSITIVE**.
- The Qwen pairs are **not predicted**. If Qwen-bfloat16 reproduces, the
  dtype hypothesis is weakened in public and the reading says so.

## What this cannot show

Ten documents, one corpus, one CPU box, one `torch`/`transformers` pair
(banked per cell). It cannot say what changed between 2026-09-03 and now
for the two documents whose `generate` text differs stably, because the
cell that was banked then carries no environment record — that is
exactly why every cell since Addendum X carries one. Float32 Phi-3 is not
run: its weights alone exceed this box's memory. A REPRODUCES reading on
ten documents is a sample, not a proof.

Artifacts: `experiments/phi3_instability_cells/<family>_<dtype>_sequence.json`,
`experiments/phi3_instability_cells/phi3-mini_bfloat16_threads.json`,
`experiments/phi3_instability_2026-09-17.json` (the reading). Runner:
`scripts/phi3_instability.py`; the reader withholds until every
preregistered cell exists.
