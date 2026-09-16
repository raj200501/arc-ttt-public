# Addendum X — the decoder, isolated (preregistration)

**Frozen 2026-09-16, before the plain arm ran on any family.** Addendum V
ends with a sentence against itself: *"The design that would isolate the
decoder — one loop, constraint toggled, everything else equal — has not
been run."* This is that design.

V compared a constrained arm decoded by `src/arcttt/constrained_json.py`
against comparator cells decoded by `model.generate`. Two things differ
between those arms, not one: the **constraint**, and the **decoding
path** — and `model.generate` applies a checkpoint's `generation_config`
defaults even with `do_sample=False`, so on one family a third thing
differs as well.

**V could only attribute the differences BETWEEN its arms; it could not
measure them. Its within-arm quantities do not depend on that, and this
addendum cannot reinstate them.** The constrained arm on its own left
**7, 8, 5, 7 and 8 of 50 outputs invalid** (Qwen2.5-0.5B, SmolLM2,
Granite, Phi-3, Falcon3), every one stopped at the token cap, with 0
fallbacks. That is what failed V's `I_c ≤ 1` REMOVES threshold and
withdrew the drop-in claim, and no comparator enters that test. What
rests on an attribution is narrower and is exactly what X measures: which
of the three differences produced the between-arm effects V reported —
its regression count, its byte-identity observations, and the erratum it
put on Ladder II's SYSTEM rows.

This addendum separates the three.

## Arms — same five families, the same fifty documents, the same prompts

The Addendum V cells exactly: the schema-only regime, the first 50
receipts in the comparator cell's document order (`cord-000` … `cord-049`),
the Addendum S `SCHEMA_INSTRUCTION` verbatim as one user turn through the
family's own chat template, `max_new_tokens=512`, and each family's own
dtype (Phi-3 bfloat16, the rest float32).

| arm | decoder | status |
|---|---|---|
| **C** — constrained | `constrained_greedy_generate`, validator on, top-k 16, leading fence tolerated | V's banked cells, **reused under a determinism gate** — except where the prompt is not reproducible, where it is **re-run** |
| **P** — plain, path-matched | **the same function, `enforce=False`**: top-1 always taken, EOS never suppressed, same cap, same fence tolerance, same stop rule | **new** |
| **G** — generate, defaults neutralised | the checkpoint's own `generation_config` loaded, with **only its modifiers** set to neutral values, `do_sample=False` | **new, where the scope record says it runs** |
| **G₀** — generate, defaults in force | `model.generate(do_sample=False)`, config untouched | V's comparator cells, **reused under the same determinism gate** |

**C and P are the same function.** Not two loops asserted to match — one
loop with one boolean. That is the whole point of the addendum, and it is
why the flag is added to the shipped module rather than to a copy of it.
`enforce=True` behaves exactly as the module did before the flag existed,
which is pinned by test rather than asserted here.

**Arm G starts from the checkpoint's config and changes only the
modifiers**, so everything else the checkpoint ships — EOS token lists,
pad, BOS — survives. Building a fresh config instead would drop those and
make arm G differ from arm G₀ in more than the thing under test.

## Scope is decided by reading the checkpoints, not the results

Each family's `generation_config` and chat template are inspected and
banked **before any arm runs**, in
`experiments/generation_configs_2026-09-16.json`. Two properties decide
which arms a family gets, and both are properties of the checkpoint:

1. **Does it ship a generation modifier?** Measured against the library's
   own defaults, because `GenerationConfig.to_dict()` fills those in and a
   naive read reports every key as set. `top_k` alone does not count: with
   `do_sample=False` no sampling warper runs, so it cannot change greedy
   output. Measured 2026-09-16: **`Qwen/Qwen2.5-0.5B-Instruct` carries
   `repetition_penalty` 1.1, `temperature` 0.7, `top_p` 0.8, `top_k` 20 and
   `do_sample=True`; SmolLM2, Granite, Phi-3 and Falcon3 carry none.**
2. **Does its chat template read the clock?** See below. Measured
   2026-09-16: **`ibm-granite/granite-3.1-2b-instruct` calls
   `strftime_now`; no other family calls anything.**

A family gets **arm G** if either is true. It keeps the **reused arm C**
only if the second is false.

## The prompt that changed every day

**Granite's chat template builds its own system message and puts
`strftime_now('%B %d, %Y')` inside it.** So the prompt this repository has
been sending that checkpoint was different on every day it ran. Addendum
V's constrained cell and Addendum S/T's comparator cell for Granite were
produced five days apart and therefore **did not carry the same prompt** —
a confound inside the family V reported as byte-identical between its arms.
Found by adversarial audit of this protocol before it was frozen, and
disclosed here rather than in a later erratum.

Consequences, all taken in the direction that costs us work:

- Every Addendum X arm renders that family's system message from a
  **frozen string** pinned to `September 16, 2026`, so X's own prompts are
  identical across its arms and reproducible by a stranger on any future
  day. The frozen string is checked at run time to reproduce the
  template's own output exactly, differing only in the date; if the
  template changes, the run refuses to start rather than quietly sending a
  different prompt.
- **Granite's arm C is re-run, not reused**, because V's cell carries a
  prompt this run cannot reproduce.
- **Granite gets arm G**, because its banked arm G₀ carries a third,
  different prompt and cannot serve as the path comparator.
- Granite's determinism gate reports NOT APPLICABLE and says why, rather
  than passing a check it cannot perform.
- Every cell this addendum banks carries a **sha256 of the rendered
  prompt per document**, and the reader **checks prompt identity between
  the arms it compares** instead of assuming it. V's cells carry no such
  digest, which is why this was invisible.

## The determinism gate on the reused arms

Reusing banked cells assumes the decoder is deterministic across
processes and container rebuilds. That is assumed everywhere in this
repository and has never been tested. Before the reading, **both reused
arms** — the constrained arm C and the generate comparator G₀ — are
re-run on the **first five documents of every family**, fixed here before
the run, and every one must be **byte-identical** to the banked text. If
any differs, the reuse is void, **the whole addendum withholds**, and the
non-determinism publishes as the result instead, because a decoder that
does not reproduce is a larger finding than anything else this addendum
could return.

## The three contrasts

- **X1 — the constraint, isolated.** C against P. One loop, one boolean,
  everything else equal by construction. *Does the constraint change what
  the decoder emits, and does it remove invalid JSON?*
- **X2 — the implementation path.** P against G where arm G runs, else
  against G₀ (where G ≡ G₀ by construction). *Does our loop reproduce
  `generate` when neither is constrained and neither has defaults in play?*
- **X3 — the defaults.** G against G₀, on the family that ships modifiers.
  *How much of what V saw was the `generation_config`?* This is the
  contrast that puts V's own published attribution at risk.

**The stop rules differ between our loop and `generate`, and X2 is
measured so that difference cannot masquerade as divergence.** Our loop
stops as soon as the root closes and parses; `generate` stops at EOS or
the cap. So X2 does **not** compare final strings. It asks whether the
**plain arm's** generated token sequence is a **prefix** of the generate
arm's, and where it is not, records the **first index at which they
differ**. Arms P and G bank their generated token ids for that comparison.
Arm G₀'s banked cells carry text only, so on those families the prefix
test is applied to the decoded text — a weaker test, and named as weaker
in the artifact rather than silently equated with the token test.

Two cases the bare prefix rule would score in our favour, and which the
rule therefore excludes: **the generate arm stopping first** is never
expected and counts as a divergence at its own length; and **the plain arm
stopping on EOS** means the model asked to stop on logits the generate arm
also saw, so there the two must end in the same place and any further
generate tokens count as a divergence. Without that second clause an empty
plain arm against a long generate arm would read as perfect agreement.

## Quantities

Both arms of every contrast are classified by the SHIPPED
`tools/fencecheck.py` strip and the fail-closed
`arcttt.scoring.parse_json_object`, symmetrically, exactly as V did:
**invalid** (does not parse to one JSON object after the strip), **fenced**,
and field-level micro-F1 against CORD gold as context only. Decode
accounting — constrained steps, fallbacks, stop reasons, token counts —
and the run environment (library versions, thread count, platform) are
banked per cell, so a difference between two cells can be attributed
rather than guessed.

## Frozen readings — applied by arithmetic in the reader

**X1, per family.** `I_p` and `I_c` are the invalid counts of 50 under P
and C; `R` is the number of documents valid under P and invalid under C.
Evaluated in this order, first match wins:

1. **CONSTRAINT INERT** — the arms are byte-identical on all 50 documents.
2. **CONSTRAINT HURTS** — `I_c > I_p`, or `R ≥ 3`.
3. **UNTESTABLE AT SIZE** — `I_p ≤ 1`. Nothing to remove, so the family is
   counted neither way. This is Addendum V's own guard, and it is here
   because without it `I_c ≤ I_p // 2` is satisfied by `0 ≤ 0` and a family
   where the constraint removed nothing reads as REMOVES.
4. **CONSTRAINT REMOVES** — `I_c ≤ I_p // 2` and `R = 0`.
5. **CONSTRAINT HELPS** — `I_c < I_p` and `R = 0`.
6. **CONSTRAINT MIXED** — `I_c < I_p` and `1 ≤ R ≤ 2`.
7. **CONSTRAINT NEUTRAL** — anything else (`I_c = I_p`, arms not identical).

HURTS is evaluated before UNTESTABLE on purpose: a constraint that breaks
a valid output is a finding even on a family that had nothing to fix.

Combined, in the non-flattering direction: *the constraint removes invalid
JSON* may be said only if **CONSTRAINT REMOVES** fires in at least **4 of
5** families and no family fires HURTS. If a family fires HURTS **and**
REMOVES fires nowhere, there is no sentence to narrow and **the claim is
withdrawn rather than scoped**. Otherwise a HURTS family is named at full
size and the sentence becomes *on N of the 5 families tested*, with N the
count that actually fired. Neither INERT nor UNTESTABLE is a success,
neither is ever counted toward the 4, and the denominator stays 5.

**X2, per family.** `D_path` is the number of documents where the prefix
rule above finds a divergence.

- **PATH REPRODUCES** — `D_path = 0`.
- **PATH DIVERGES** — `D_path ≥ 1`; the count publishes, with the first
  divergence index per document and the invalid-count difference between
  the two arms beside it.

*Our loop reproduces `generate` token-for-token* may be said only where
`D_path = 0` **on the token basis** — the families that run arm G, which
the banked scope record fixes as Qwen2.5-0.5B (modifiers) and Granite
(clock-dependent prompt). Where the basis is decoded text — SmolLM2,
Phi-3 and Falcon3, whose arm G₀ cells bank text only — the sentence is
only *our loop reproduces `generate`'s decoded text on these fifty
documents*, and the reading itself carries the words `WEAKER TEST:
decoded text, not token ids` so the two can never be quoted as the same
result. Nowhere else, and never as a general statement about the loop.

**X3, on the family that ships modifiers.** `D_def` is the number of
documents where G and G₀ differ. `R_V` is the regression count V published
for that family (valid under G₀, invalid under C), **read from V's own
artifact rather than retyped here** — this addendum can narrow one of V's
sentences, so the number being narrowed must have exactly one referent. It
is 5 at the time of freezing. The same quantity is **recomputed** here from
the same two banked cells; **if the recomputation disagrees with what V
published, that disagreement is read first, the attribution reading
withholds, and the disagreement publishes before anything else in X3.**
`E` is how many of those regressions are **also invalid under G** — i.e.
owed their validity to the defaults rather than lost to the constraint.

- **V'S ATTRIBUTION HOLDS** — `E = R_V`. V's sentence *"the regressions are
  the penalty difference, not the constraint"* stands as written.
- **V'S ATTRIBUTION IS TOO STRONG** — `E < R_V`. The sentence is narrowed
  by dated erratum to *"E of the R_V"*, and **each residual document is
  attributed by arithmetic on that same document**: if arms C and P differ
  there, the constraint owns it; if they agree, the difference is
  elsewhere in the path.

The second branch is the one that costs us something. It is not protected
by where it sits in the source — `E < R_V` and its negation are
complementary arithmetic, and HOLDS is the fall-through. What keeps it
from being reached around is that `E` and `R_V` are both **recomputed by
the reader from the banked cells** rather than read from prose, that the
reading is emitted by `x3_reading` rather than drafted after the numbers
are seen, that a recomputation disagreeing with V withholds the
attribution entirely, and that both branches are pinned by test at their
boundary.

## Two predictions, written before the run

**X1.** V banked **0 constrained steps** on every document of four
families and on 43 of Falcon3's 50, with 0 fallbacks everywhere. If that
accounting is correct, then arms C and P must be **byte-identical on every
document with 0 constrained steps** — a decoder that never overrode the
top-1 token cannot have produced different text — so X1 reads **INERT** on
Qwen2.5-0.5B, SmolLM2, Granite and Phi-3, and on Falcon3 the arms differ
on **exactly the seven documents where the validator fired**.

**This is checked per document, not per family.** Summing a family's
constrained steps would make the check vacuous wherever the validator
fired at all, and would have excused Falcon3's other 43 documents.

The check is symmetric, because the other direction is equally
impossible: a document where the validator DID fire cannot have identical
arms either, since firing means a top-1 token was rejected. Both lists are
banked.

**If any document with 0 banked constrained steps has non-identical arms —
or any document with steps has identical ones — the decoder's own
accounting is wrong, and that is the result** — a larger and worse one
than the contrast this addendum was built to make. It publishes in those
words.

**X2**, weaker and stated as weaker: PATH DIVERGES on Phi-3 (V saw 32 of
50 bodies differ at bfloat16) and PATH REPRODUCES on SmolLM2. Qwen,
Falcon3 and Granite are not predicted — Granite because its arms are being
compared under a prompt V never used.

## What this cannot show

Fifty receipts per family, one corpus, schema-only prompts, CPU, ≤ 4B, one
`transformers` version, one box. A valid object is not a correct one:
`invalid` counts syntax and the score is context. X3 is measured on the one
family that ships a modifier; it says nothing about how other checkpoints'
defaults behave. Arm C and arm G₀ are reused under a five-document
determinism gate per family, not re-run in full — the gate is a sample, and
a decoder could in principle be deterministic on five documents and not on
fifty. Granite's contrasts are run under a pinned prompt that **no earlier
addendum used**, so its numbers here are not comparable, document by
document, with the Granite numbers in V or T; that is the price of making
them reproducible at all, and it is stated rather than smoothed over.

Artifacts: `experiments/cord_decoder_isolation_cells/<family>_{plain,constrained,generate_neutral,determinism}.json`,
`experiments/cord_decoder_isolation_2026-09-16.json` (the reading),
`experiments/generation_configs_2026-09-16.json` (the scope, banked before
any arm). Runner: `scripts/cord_decoder_isolation.py`; driver:
`scripts/run_addendum_x.sh`. The reader withholds until every
preregistered cell exists and the determinism gate has passed.
