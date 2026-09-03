# Adaptation engineering ladder II — architecture rungs

**Frozen 2026-09-02, before any rung ran.** Ladder I
(`ADAPTATION_ENGINEERING_LADDER.md`) ended with reading (b) on E6:
the stacked configuration (adapted 3B + k=20 demonstrations) does not
separate from demonstrations alone on CORD — D=+0.0109, 28W/27L/25T.
The founder's directive is that better engineering and architecture
can produce better numbers. This ladder is that attempt, under the
same rules: bars and readings frozen here, the two-statistics rule
(paired mean ≥ +0.01 AND sign test p ≤ 0.05), every failure published
at full size, attempt counters printed beside any success, raw text
banked for every arm, fence handling symmetric.

## What E6's outputs say the architecture should fix (measured before this was written)

Decomposition of the banked E6 arms (`experiments/ladder_e6_cord_*`),
run 2026-09-02 before any Ladder-II code existed:

- **No truncation.** Longest output 452 tokens; longest gold 323; the
  512 cap was never hit. Raising it would change nothing.
- **Invalid JSON is the stack's dominant loss.** The adapted arm's 8
  invalid outputs are all structural faults — single-quoted keys
  (`'nm':'...'`), an extra closing brace, a second object appended —
  none of them content errors. Four of those documents are ones the
  prompted arm parsed, and they carry **−1.885 of the stack's −3.443
  total loss mass (55%)**. The prompted arm has the same fault class
  at half the rate (4 invalid: three single-quote, one prose).
- **On valid outputs the stack already errs less.** Field-level
  mismatches, adapted vs prompted: `menu.nm` 90 vs 119, `menu.cnt` 81
  vs 102, `menu.price` 71 vs 90.

So the mechanism is specific: the stack knows the content better and
loses on syntax. A decoder that cannot emit a syntactically invalid
JSON object attacks exactly that loss. Whether it separates the arms is
what the rung measures; the decomposition licenses the rung, not the
result.

## Rule 1 — what a rung is measured against

Every rung has TWO readings, both frozen, both published:

- **(ADAPT) the adaptation reading:** the rung's stack arm vs the
  prompted 3B k=20 arm *with the same decoder*. Same size, same
  prompt, same decoding architecture; only the adapter differs. This is
  the only reading that can credit adaptation.
- **(SYSTEM) the system reading:** the rung's stack arm vs **E6's
  frozen prompted bar** (`ladder_e6_cord_prompted_2026-08-31.json`,
  greedy, 0.7166). This credits the whole engineered system — adapter
  plus decoder plus anything else the rung adds — against the plain
  baseline, and it is stated as such: a SYSTEM win with an ADAPT
  failure means the decoder did the work.

Attempt counters: at the 3B prompted-CORD bar this is attempt 4
(E6 was 3); each rung increments.

## Rungs

### E7 — JSON-constrained greedy decoding, both arms

**Architecture.** A constrained greedy decoder: at each step the
candidate tokens (top-k=16 by logit) are tried in order and the first
whose appended text keeps the output a **valid JSON prefix** is
emitted; if none qualifies the top-1 token is emitted (so the decoder
degrades to greedy, never stalls, and the fallback count is banked).
The prefix validator is a string-level incremental checker: string
state, escape state, bracket/brace stack, double-quote-only keys and
strings, no trailing content after the root closes. It does NOT know
the CORD schema — it enforces JSON syntax only, so both arms get an
identical, schema-blind decoder. Implemented in
`src/arcttt/constrained_json.py`, unit-tested on the exact fault
classes E6 exhibited (single-quoted key, extra closer, appended second
object).

**Cells.** Same E6 split, same adapter (`work/e6/adapter.pt`, banked
sentinel), same k=20 prompts, bfloat16, max_new_tokens 512, seed 1,
per-document checkpoints: (a) prompted 3B k=20 + constrained decoder;
(b) adapted 3B + k=20 + constrained decoder. ~9.5h CPU total.

**Frozen readings.**
- (a) ADAPT clears → the adaptation claim returns on public data, with
  the decoder named as the enabling architecture and attempt 4 printed.
- (b) ADAPT mean positive but unseparated → both numbers publish, no
  adaptation claim; the SYSTEM reading publishes on its own terms.
- (c) ADAPT at/below zero → adaptation adds nothing even with syntax
  removed as a factor; the stack's content advantage was illusory.
- SYSTEM readings are published independently with the same rule and
  never substitute for ADAPT.
- Banked beside the readings: invalid counts per arm (expected to fall
  to ~0), fallback-to-greedy counts, and per-document deltas vs E6's
  greedy arms — a decoder that changes valid outputs' content is a
  finding in itself.

### E8 — similarity-ordered demonstrations (on top of E7)

**Architecture.** Demonstration ORDER by lexical similarity to the test
receipt (BM25 over the OCR text, most similar demonstration last,
nearest the query). Same 20 demonstrations, same decoder as E7; both
arms. Cheap, and a known effect direction in in-context learning.

**Frozen readings.** ADAPT and SYSTEM as in E7, attempt 5. Additionally
the E8 prompted arm vs the E7 prompted arm publishes the ordering
effect on its own, so a SYSTEM gain cannot be misattributed.

### E9 — a larger adaptation set with retrieved demonstrations

**Architecture.** A new frozen split of the 100 CORD receipts: seed-2
shuffle, 40 train / 60 eval (SHA-banked before the arms run). The
adapter is trained on 40; each test receipt receives the 20 most
similar of the 40 as demonstrations (BM25). Prompted arm identical
minus the adapter. E7's decoder. Attempt counter resets to 1 because
the bar (prompted, 60-receipt eval, retrieved demos) is a new bar —
stated as such.

**Frozen readings.** ADAPT and SYSTEM as above on the new eval set;
no cross-split comparison to E6/E7/E8 numbers is made.

## Cost and order

E7 first (it is the mechanism the decomposition licenses), read and
banked before E8 runs; E9 only if E7 or E8 shows ADAPT ≥ +0.01 by mean
(otherwise the stack has no content advantage worth scaling, and the
ladder says so and stops). Total budget ~30h CPU across wakes,
checkpointed cells, sentinels, raw text banked, resume disclosed.

## What this ladder cannot do

It cannot move the bar. It cannot soften the two-statistics rule. It
cannot compare a decoded arm against an undecoded one and call that
adaptation. It cannot read a result before both arms of a cell exist.
Where its letter and its substance disagree, the reading that does
not flatter us governs.

## E9 protocol note — fixed 2026-09-03 after E8 read, before any E9 data

E8 cleared its ADAPT letter because BM25 ORDERING damaged the prompted
arm (−0.029, p=0.0016) while the adapted arm was unmoved; against the
strongest prompted arm the stack was +0.029 and unseparated (p=0.60).
So for E9 the demonstration ORDER is fixed to the split order for both
arms; BM25 is used only to SELECT the 20 nearest of the 40 training
receipts per test receipt. The E9 bar is therefore the strongest
prompted configuration available to it, not a degraded one. Split:
seed-2 shuffle of the 100 receipts, 40 train / 60 eval, SHA-banked
before the arms run (`experiments/ladder_e9_cord_split/`). Adapter:
trained on the 40 (same recipe as E6: r16/α32, one pass, bfloat16,
gradient checkpointing), durable sentinel. Decoder: E7's. Readings:
ADAPT (vs prompted, same selection, same order, same decoder) and
SYSTEM (vs a fresh greedy prompted arm on the SAME 60 receipts with
the same selected demonstrations — E6's bar is a different eval set
and cannot serve). Attempt counter resets to 1 for this new bar, as
frozen above. Cost: one 3B adaptation on 40 receipts (~25 min) plus
three 3B arms of 60 receipts.

## Ladder II closed — 2026-09-03

| rung | stack | ADAPT bar | D | W/L/T | p | reading |
|---|---|---|---|---|---|---|
| E7 decoder | 0.7995 | 0.7650 | +0.0345 | 32/27/21 | 0.30 | (b) |
| E8 + ordered demos | 0.7943 | 0.7361 (ordering-damaged) | +0.0582 | 42/18/20 | 0.0013 | letter (a); substance: order-robustness, not adaptation-beats-prompting (vs strongest prompted arm +0.0293, p=0.60, exploratory) |
| E9 40-receipt adapter + retrieved demos, new split | 0.8450 | 0.8007 | +0.0444 | 26/18/16 | 0.146 | (b) |

SYSTEM readings: E7 +0.0830 (p=0.096), E8 +0.0777 (p=0.40), E9 +0.0521
(p=0.087) — none clears. What the ladder established, at its size: the
constrained decoder removes every invalid output on both arms and lifts
every configuration; retrieval selection lifts prompting; the adapted
model is robust to demonstration order where the prompted model is not;
and the stack's advantage over the best prompting available to it is a
mean of +0.03 to +0.06 that never separates by sign test across three
architectures and a doubled adaptation set. The ladder stops here by
its own rule: no rung is left that the decompositions license, and the
readings that would let a worse bar clear have been named rather than
used.
