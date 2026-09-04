# Addendum T — does the fence tax hold across model families? (preregistration)

**Frozen 2026-09-03, before any arm ran.** Every fence-rate number this
project has is on one model family (Qwen2.5, 0.5B–3B). A reviewer's
objection, verbatim in spirit: *to be a category rather than a
curiosity, the tax needs to show up on other families at the prompt
regimes teams actually use — or you need to state that it doesn't.*
This addendum answers it the same way Addendum S answered the
one-corpus objection: same instrument, same corpus, same two regimes,
different checkpoints, thresholds frozen here.

## Families

Four instruction-tuned checkpoints from four organisations, chosen
BEFORE any ran, by one rule: ungated on the Hub (so the run is
reproducible without credentials) and ≤ 4B parameters (so it runs on
this CPU box in bfloat16 or float32). Gated families (Llama 3.2, Gemma
2) were probed and refused a download without a token; they are named
here as the ones this addendum could not test, not silently omitted.

| family | checkpoint | architecture |
|---|---|---|
| HuggingFace | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | Llama-class |
| IBM | `ibm-granite/granite-3.1-2b-instruct` | Granite |
| Microsoft | `microsoft/Phi-3-mini-4k-instruct` (3.8B) | Phi-3 |
| TII | `tiiuae/Falcon3-1B-Instruct` | Llama-class |

## Cells — identical to Addendum S

For each family: (i) **schema-only** — the Addendum S instruction
verbatim (`docs/research/ADDENDUM_S_PROTOCOL.md`), one user turn, all
100 CORD receipts; (ii) **k-shot, k=20** — the E6 train split as
demonstrations through `run_challenge.build_task` +
`text_task_to_messages`, the 80 non-demonstration receipts. Greedy,
`max_new_tokens=512`, `max_seq=8192`, each family's own chat template
via `apply_chat_template`, per-document checkpoints, raw text banked.
dtype: float32 for ≤ 2B, bfloat16 for Phi-3 (memory), stated in each
cell artifact. Classifier: the SHIPPED `tools/fencecheck.py`
`strip_fence`, leading-fence scope, the same instrument as R and S.

## Frozen readings — per family, then combined

Per family, with `f_schema` and `f_kshot` the fence rates:
- **(a) replicates:** `f_schema ≥ 0.50` and `f_kshot ≤ 0.10`.
- **(b) partial:** `f_schema ≥ 0.50` and `f_kshot > 0.10`.
- **(c) does not replicate:** `f_schema < 0.50`.

Combined, in the non-flattering direction:
- **HOLDS ACROSS FAMILIES** may be claimed only if (a) fires in **at
  least 3 of 4** families and no family fires (c).
- Any family firing **(c)** publishes as a named exception at full
  size, and the outbound sentence becomes *"on N of the 5 families
  tested"* with the exception named — never *"across model families"*.
- Anything else publishes as mixed: all eight rates, no headline.

Readings are applied by arithmetic in the runner, never re-derived
from the shape of the data. Artifact:
`experiments/cord_fence_tax_families_2026-09-03.json`; per-cell raw
outputs under `experiments/cord_fence_tax_families_cells/`. Scores are
banked as context only; the carried quantity is the fence RATE.

## What this cannot show

Four families at ≤ 4B on one public corpus. It says nothing about
hosted APIs or ≥ 7B checkpoints. Reading (c) in any family is
published, not explained away.

## Erratum — 2026-09-03, noticed while the Falcon3 cells ran, before any reading

The combined-readings section above writes the exception sentence as
*"on N of the 5 families tested"*. Four families are tabled and four
run; the "5" is a slip. The sentence the runner emits is *"on N of the
families tested"* with the exception named, and N is at most 4. The
frozen text is left as written; this note governs.

## Disclosure — 2026-09-04, before the T reader ran

Addendum U (`ADDENDUM_U_PROTOCOL.md`) ran the shipped parsers over
every banked raw output, including the six T cells banked so far, and
its per-family strict-parse loss on schema-only outputs — Granite 0/89,
SmolLM2 0/85, Falcon3 73/73 — is arithmetically the direction of this
addendum's per-family readings before `--read` has run on all eight
cells. Nothing in this protocol changes: the thresholds are frozen, the
reader withholds until the Phi-3 cells exist, and the reading it prints
is the one banked. It is stated here so nobody has to discover that the
T result was foreseeable from a sibling artifact.
