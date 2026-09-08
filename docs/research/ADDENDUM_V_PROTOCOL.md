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
