# Addendum V — does the constrained decoder remove invalid JSON on other families? (preregistration)

**Frozen 2026-09-08, before any constrained arm ran on any family but
Qwen2.5-3B.** Ladder II's rung E7 showed that a schema-blind,
JSON-constrained greedy decoder (`src/arcttt/constrained_json.py`)
removed every invalid output on both 3B arms on CORD. The roadmap names
the decoder as a drop-in for any HF causal LM; that is a claim about
other families, and it has been measured on one. Addendum T banked
schema-only cells on four other families in which 11–27 of 100
outputs are invalid JSON even after the fence is stripped. This
addendum re-decodes the same prompts with the constrained decoder and
asks whether the invalid outputs go away — and whether anything else
changes.

## Cells

Five families, the schema-only regime only (where the invalid outputs
concentrate; the k-shot cells have 1–14 invalid and are not re-run):

| family | checkpoint | dtype | greedy comparator cell |
|---|---|---|---|
| Qwen2.5 | `Qwen/Qwen2.5-0.5B-Instruct` | float32 | `cord_fence_tax_cells/0.5b_schema.json` |
| HuggingFace | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | float32 | `cord_fence_tax_families_cells/smollm2-1.7b_schema.json` |
| IBM | `ibm-granite/granite-3.1-2b-instruct` | float32 | `…/granite-2b_schema.json` |
| Microsoft | `microsoft/Phi-3-mini-4k-instruct` | bfloat16 | `…/phi3-mini_schema.json` |
| TII | `tiiuae/Falcon3-1B-Instruct` | float32 | `…/falcon3-1b_schema.json` |

Per family: the FIRST 50 receipts in the comparator cell's document
order (`cord-000` … `cord-049`), the identical prompt (the Addendum S
`SCHEMA_INSTRUCTION` verbatim, one user turn, the family's own chat
template), `max_new_tokens=512`, the same dtype as the comparator, and
E7's decoder unchanged: top-k = 16 candidates tried in logit order, the
first that keeps the decoded text a valid JSON prefix is emitted, a
leading fence tolerated, greedy fallback when no candidate qualifies
(counted), stop on EOS or when the root closes and parses. Fifty, not
one hundred, because the decoder is 1.5–2× slower than greedy on this
CPU box and the run must finish inside one day of restarts; the
comparator is restricted to the same fifty documents by id. Raw text,
per-document seconds, fallback counts, constrained-step counts and stop
reasons are banked per document; per-document checkpoints with the
config key; resume disclosed.

## Quantities — per family, on the same fifty documents

Both arms classified by the SHIPPED `tools/fencecheck.py` strip and the
fail-closed `parse_json_object`, symmetrically:

- **invalid** — outputs that do not parse to one JSON object after the
  strip, greedy vs constrained;
- **regressions** — documents that parsed under greedy and do not under
  constrained (a decoder that breaks valid output is a finding);
- **fenced** — outputs the strip flagged, greedy vs constrained (the
  decoder tolerates a leading fence; whether it changes the fence rate
  is banked, not predicted);
- **score** — field-level micro-F1 against CORD gold, paired per
  document, mean delta and sign test (context; the carried quantity is
  `invalid`);
- **decode accounting** — fallbacks, constrained steps, stop reasons.

## Frozen readings — applied by arithmetic in the reader

Per family, with `I_g` and `I_c` the invalid counts of 50:

- **REMOVES:** `I_c ≤ 1` and `regressions = 0`.
- **REDUCES:** `I_c ≤ I_g / 2` (integer floor) and not REMOVES.
- **NO EFFECT:** anything else, including any family with `I_g ≤ 1`
  (nothing to remove — named as untestable at size, not counted
  either way).

Combined, in the non-flattering direction: *the decoder removes
invalid JSON across families* may be said only if REMOVES fires in at
least **4 of 5** families and no family fires NO EFFECT; any NO EFFECT
family is a named exception at full size and the sentence becomes *on
N of the 5 families tested*; anything else publishes as mixed, all
counts, no headline. A family with `regressions ≥ 3` is published as a
regression finding regardless of its invalid counts. The score delta
publishes on its own terms (two-statistics rule: mean ≥ +0.01 and sign
test p ≤ 0.05 to call it a gain; a negative mean with p ≤ 0.05 is
published as a loss) and never substitutes for the invalid reading.

**One prediction, written before the run:** REMOVES on Qwen2.5-0.5B
(the mechanism was measured on the same family at 3B) and at least
REDUCES on the other four; fallback-to-greedy counts stay under 5% of
constrained steps. If a family's outputs are invalid for a reason the
prefix validator does not cover (a truncated document at the token cap
is the obvious one — the greedy cells' longest outputs are banked), the
residual is decomposed by cause at size.

## What this cannot show

Fifty receipts per family, one corpus, schema-only prompts, CPU,
≤ 4B. A valid object is not a correct one: `invalid` counts syntax,
the score is context. A decoder that removes invalid output changes
what a downstream parser sees; whether a team would deploy it is not
measured here.

Artifacts: `experiments/cord_constrained_families_cells/<family>_schema_constrained.json`
and `experiments/cord_constrained_families_2026-09-08.json` (the
reading). Runner: `scripts/cord_constrained_families.py`; driver:
`scripts/run_addendum_v.sh`. The reader withholds until all five cells
exist.

## Errata — 2026-09-09, after the reading, corrected the same day under review

1. **The invalid-reading prediction failed in every family.** Written
   before the run: REMOVES on Qwen2.5-0.5B, at least REDUCES on the
   other four. Measured: NO EFFECT in all five; Falcon3 14 → 8 misses
   REDUCES by one. The fallback clause (under 5% of constrained steps)
   held trivially: 0 fallbacks. Published as written.
2. **Cause, decomposed post-hoc.** Every output still invalid under the
   constraint stopped at the 512-token cap (two of Qwen's are repetition
   loops). The validator fired on 7 Falcon3 documents — exactly its
   non-truncation faults, four prose answers and three syntax faults —
   and six of those became valid; it fired 0 times in the other four
   families. A syntax constraint cannot close a document the model has
   not finished. Banked in the artifact's `post_hoc_added_2026-09-09`.
3. **A confound the protocol did not anticipate, on one family.** The
   comparator cells were decoded by `model.generate`, which applies a
   checkpoint's `generation_config` defaults even with `do_sample=False`;
   Qwen2.5-0.5B ships `repetition_penalty` 1.1 and is the only family
   here that does, while the constrained decoder is plain top-1 over raw
   logits. On Qwen the arms differ in 47 of 50 bodies with 0 constrained
   steps: four greedy-invalid outputs became valid by the path alone and
   five valid ones broke — the regressions are the penalty difference.
   Phi-3 diverges on 32 bodies with 0 steps and no penalty; attributed to
   bfloat16 numerics, untested. SmolLM2 and Granite are byte-identical
   (a genuine null, not a design artefact) and Falcon3 differs only where
   the validator fired. Ladder II's ADAPT readings held the decoder
   constant across arms and are unaffected; its SYSTEM readings compared
   against `generate` arms (Qwen2.5-3B ships 1.05) and carry an erratum.
   The design that would isolate the decoder — one loop, constraint
   toggled — has not been run.
4. **Consequence.** The roadmap's "drop-in for any HF causal LM" is
   withdrawn; `tools/jsongreedy.py` stays published with this addendum as
   its measured limit, Falcon3's six recoveries included.
5. **The regression rule was not applied by arithmetic** in the first
   banked artifact (it lived only in the VERDICT prose); the reader now
   appends "REGRESSION FINDING" to the per-family reading and names it
   in the combined finding. Qwen2.5-0.5B: 5 ≥ 3.

## Erratum — 2026-09-16, found by audit of the Addendum X preregistration

6. **Granite's two arms were not sent the same prompt.**
   `ibm-granite/granite-3.1-2b-instruct`'s chat template builds its own
   system message and puts `strftime_now('%B %d, %Y')` inside it, so a
   call with no explicit system turn embeds the wall-clock date. This
   addendum's constrained cell and the Addendum S/T comparator cell it was
   read against were produced five days apart, so they carried different
   prompts. Erratum 3 above lists Granite among the families whose arms
   are "byte-identical (a genuine null, not a design artefact)"; that
   sentence is now known to describe a pair of runs with different inputs,
   and it can no longer be read as evidence that the decoder had nothing
   to fix — it is evidence that the output did not move when both the
   prompt date and the decoder changed. **No Granite number here is
   withdrawn, and none is defended.** The affected claim is the inference,
   not the count. Addendum X pins that family's system message to a frozen
   date, re-runs its constrained arm rather than reusing this one, adds a
   `generate` arm because this comparator cannot serve as one, and banks a
   sha256 of the rendered prompt per document so a reader can check prompt
   identity between compared arms instead of assuming it. Found before the
   X protocol froze, not after a result depended on it.

## Erratum — 2026-09-16, from Addendum X's determinism gate

7. **Phi-3's cells do not reproduce, and erratum 3's Phi-3 sentence is
   upgraded from an attribution to a measurement.** Erratum 3 said Phi-3
   "diverges on 32 bodies with 0 constrained steps and no penalty;
   attributed to bfloat16 numerics, untested." Addendum X tested it, and
   the result is worse than the attribution. Re-decoding the first five
   documents of this family today: the `model.generate` comparator does
   not reproduce the text banked on 2026-09-03 on two of five documents —
   identically in two separate runs today, so a stable difference between
   then and now — and the constrained arm differed from this addendum's
   own banked text on a third document in one gate run and matched it in
   the next. In isolation that document is stable (a fresh process
   reproduces it twice), so the instability is order-dependent inside a
   process rather than random.

   **Consequence for this addendum.** Its Phi-3 row is not
   byte-reproducible: a stranger re-running the command does not get the
   banked text back on that family. The invalid counts are not withdrawn
   and the reading is not re-taken — nothing here is known to be wrong —
   but the 32-of-50 between-arm divergence this addendum reported for
   Phi-3 **needs no cross-arm explanation at all**, because the family
   does not reproduce against itself. The other four families re-decode
   0-of-10 mismatches on both arms, which is the first evidence this
   repository has that any of its cells reproduce.

   **Consequence for Addendum X.** Its own frozen protocol makes a failed
   gate terminal, so X withholds entirely: nothing it banked is read about
   the constraint, the decoding path or the generation defaults. The gate
   was written to be able to end the addendum and it did.

## Erratum — 2026-09-17, narrowing erratum 7, from Addendum Y

8. **"The family does not reproduce against itself" was too strong.**
   Addendum Y decoded the first ten Phi-3 documents in one unbroken
   process — the constrained decoder twice back to back, a third pass
   after twenty decodes, `model.generate` in two sweeps — and every
   comparison was byte-identical, including 10 of 10 against this
   addendum's own banked constrained cell. What Addendum X's gate caught
   is narrower and now has a mechanism: the flaking document (`cord-004`)
   decodes to different text at four threads and at one thread in the
   same process, i.e. a change in floating-point reduction order flips a
   token at bfloat16 precision; the other probed document does not. The
   dtype alone is not sufficient — Qwen2.5-0.5B reproduces in bfloat16 as
   well — so the correct sentence is: Phi-3's cells reproduce within a
   process at a fixed thread count and are not guaranteed to across a
   change of reduction order. Erratum 7's consequence for this addendum
   is unchanged: its Phi-3 row is not guaranteed reproducible by a
   stranger on another box, and its between-arm Phi-3 divergence needs no
   cross-arm explanation. The `generate` comparator banked on 2026-09-03
   still differs today on 4 of 10 documents, cause unrecorded.
