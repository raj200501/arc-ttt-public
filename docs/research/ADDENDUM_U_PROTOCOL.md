# Addendum U — what shipped parsers do to real model output (preregistration)

**Frozen 2026-09-04, before the corpus builder or the runner touched any
banked output.** The fence census counted CODE: 21 of 36 packages have a
path that reads a model's JSON without handling a markdown fence and
turns the parse failure into a zero or a silent drop; one of those paths
was later executed against a real model (autoevals `JSONDiff`). The
fence tax counted OUTPUTS: how often small models wrap JSON in a fence
under each prompt regime. Nobody has yet put the two together and
MEASURED what the shipped parsers do to the outputs — including what
the lenient "repair" parsers, the usual fix, do to them. This addendum
is that measurement, and the engineering that makes it re-runnable by
anyone with a saved-outputs file.

## The corpus — every raw output this project has banked

`tools/fence_corpus.py` builds `experiments/fence_corpus_2026-09-04.jsonl`
from the banked artifacts whose `predictions` map holds the model's RAW
text per document (no fence stripping, no repair — each artifact says
so in its own `what`/`fence_policy`/`verification_level` fields, and the
builder refuses any artifact whose text field is not a string). One
record per output: `id`, source artifact and its SHA-256, model id,
family, size, adapted (bool), corpus (`waybill` | `cord`), regime
(`schema` | `kshot` | `document_only`), k, decoder (`greedy` |
`constrained`), dtype, `text`. Labels come from the artifact's own
fields (`mode`/`regime`/`arm`/`k`/`n_demonstrations`/`decoder`/`dtype`)
where they exist; the registry in the builder supplies only family,
size, adapted and corpus, and the builder ABORTS if an artifact's fields
contradict the registry. The manifest banks per-slice counts and the
corpus SHA-256.

Registry (frozen; artifacts present at build time are included, absent
ones are named in the manifest as absent, never silently dropped):

| source | family | outputs |
|---|---|---|
| Addendum M/N/R scale rungs and dose arms, waybills (`waybill_scale_rung_*`, `waybill_fence_dose_*`) | Qwen2.5 0.5B/1.5B/3B, prompted | 30 each |
| Addendum Q / ladder E5, waybills (`waybill_adapted_3b`, `ladder_e5_3b_adapted_kshot`) | Qwen2.5 3B, adapted | 30 each |
| Ladder E6–E9, CORD (`ladder_e6..e9_cord_*`) | Qwen2.5 3B, prompted and adapted, greedy and constrained | 80/80/80/80/80/80/60/60/60 |
| Addendum S cells, CORD (`cord_fence_tax_cells/`) | Qwen2.5 0.5B/1.5B | 100 schema + 80 k-shot each |
| Addendum T cells, CORD (`cord_fence_tax_families_cells/`) | SmolLM2-1.7B, Granite-3.1-2B, Phi-3-mini, Falcon3-1B | 100 schema + 80 k-shot each, as banked |

The Phi-3 cells are running as this is frozen. Rule: the corpus is
built from what exists; the reading names the number of families it
covers; if the Phi-3 cells land before the reading is banked the corpus
is rebuilt and the manifest says so; the reading is taken ONCE, on the
manifest banked beside it.

## The reference labeler — the shipped instrument, not a new one

For each record the reference object is `parse_json_object(
strip_fence(text)[0])` from the SHIPPED `tools/fencecheck.py` and
`src/arcttt/scoring.py`: one leading fence removed (Addendum R's
documented undercount applies — prose before a fence is not handled,
so the reference under-credits, never over-credits), then a
FAIL-CLOSED parse that requires the whole text to be exactly one JSON
object. Records where this yields an object are `ref` records; the
rest are `noref`. The fenced flag is `strip_fence(text)[1]`.

## Parsers under test — the real functions, frozen list

A parser is a function `text -> object | None`. Each is the SHIPPED
callable where one exists; where the shipped site is an inline
`json.loads` inside a method that needs a live completion function,
its parse-and-except semantics are reproduced verbatim and the line is
cited. The panel:

| key | what is called | why it is here |
|---|---|---|
| `strict` | `json.loads(text)`; `except ValueError -> None`; non-dict -> None | the census's fail-open shape; verbatim the semantics of `evals/elsuite/basic/json_match.py:80` (`sampled_json = None` on failure) |
| `autoevals_validjson` | `autoevals.ValidJSON().valid_json(text) == 1`, then `json.loads` — the gate `JSONDiff` uses at `autoevals/json.py:162` | the census extension's live-CONFIRMED site |
| `autoevals_jsondiff_score` | `autoevals.JSONDiff().eval(output=text, expected=<reference object>).score` | not a parser: the SCORE the shipped scorer assigns to an output whose stripped parse is exactly the expected object — what the fence costs, in the scorer's own units |
| `langchain_parse_json_markdown` | `langchain_core.output_parsers.json.parse_json_markdown(text)`; exception -> None; non-dict -> None | the most-installed lenient parser |
| `json_repair` | `json_repair.loads(text)`; non-dict (including `""`/`{}` from empty input) -> None | the usual "just repair it" fix |
| `fencecheck` | `parse_json_object(strip_fence(text)[0])` | the reference; listed for completeness, zero loss by construction, never ranked |

Not attempted, named rather than omitted: deepeval, lighteval,
inspect-ai, ragas, opencompass, crfm-helm, langsmith, guardrails-ai,
llama-index, haystack, crewai, semantic-kernel, dspy, instructor,
marvin, litellm, smolagents — their census sites are behind heavy
installs or API-bound call paths that this CPU box does not run. The
runner's parser interface is one function per entry so any of them can
be added by a later addendum without touching the readings.

## Quantities — per parser, per slice, all published

Per parser and per slice (family × size × adapted × corpus × regime ×
decoder), with `n_ref` and `n_noref` the record counts:

- **lost** = #(ref record where the parser yields no object) / n_ref —
  a correct-as-far-as-syntax-goes output the parser threw away.
- **diverged** = #(ref record where the parser yields an object ≠ the
  reference object) / n_ref — the parser changed the content.
- **fabricated** = #(noref record where the parser yields an object) /
  n_noref — the parser produced structure the model did not.
- **hazard** = (diverged + fabricated) / n — the two ways a lenient
  parser can be wrong, together.
- For `autoevals_jsondiff_score`: the mean score on fenced ref records
  and on unfenced ref records, expected = the reference object.

Object equality is structural equality of the parsed Python values.
Every per-record status (`ok` / `lost` / `diverged` / `agree_none` /
`fabricated`) is banked so any rate can be recomputed from the artifact.

## Frozen readings — applied by arithmetic in the runner

**U1 — the strict reading (does a fail-open parse lose what the fence
tax predicts?).** Per family, on schema-only outputs, `strict.lost`:
≥ 0.50 → LOSES; < 0.10 → DOES NOT LOSE; else PARTIAL. Combined, in the
non-flattering direction: *fail-open parsing loses the majority of
schema-only outputs across families* may be said only if LOSES fires
in at least 3 families and no family fires DOES NOT LOSE; any DOES NOT
LOSE family is named as an exception at full size and the sentence
becomes *on N of the families tested*; anything else publishes as
mixed, all rates, no headline. The family count is the number present
in the manifest, stated.

**U2 — the lenient reading (is "just repair it" safe?).** Per lenient
parser (`langchain_parse_json_markdown`, `json_repair`): MATERIAL if
`hazard ≥ 0.05` on any slice with n ≥ 30; HARMLESS ON THIS CORPUS if
overall `hazard < 0.01` and no slice ≥ 0.05; else PRESENT, stated at
size. MATERIAL licenses the sentence *lenient repair changes or invents
content on real outputs*, with the slice named; HARMLESS is published
as the finding when it fires — the addendum does not get to hide it.

**U3 — the k-shot reading (does the loss vanish where the fence
does?).** `strict.lost` on k-shot (k = 20) outputs, pooled across
families: < 0.05 → the fence tax and the parser loss are the same
phenomenon at k = 20; ≥ 0.05 → strict parsing loses outputs the fence
does not explain, and the residual is decomposed by cause (invalid
JSON vs other) at size.

No reading may be re-derived from the shape of the data; the runner
prints them from the thresholds above and the reader banks what it
prints.

## What this cannot show

Two corpora, one agent-authored (waybills) and one public (CORD); open
checkpoints ≤ 4B on CPU; greedy or constrained decoding; no hosted
APIs. Six parsers, not twenty-one. A parse is not a score: `lost`
counts outputs the parser discarded, not fields the model got wrong.
The reference under-credits by Addendum R's documented undercount, so
`lost` is a floor for the fail-open parsers, never a ceiling.

Artifacts: `experiments/fence_corpus_2026-09-04.jsonl` (+ `.manifest.json`),
`experiments/parser_robustness_2026-09-04.json`. Runner:
`scripts/parser_robustness.py`. Builder: `tools/fence_corpus.py`.
Both idempotent; both refuse to read a reading from a partial corpus.

## Errata — 2026-09-04, after the run, from an adversarial review of the published row

1. **U1 precedence.** The frozen text licenses the *"on N of the
   families tested"* sentence as the reduced form of a headline that
   itself needs LOSES in at least 3 families. With 2 LOSES and 2
   exceptions the runner emitted the exception form alone; the
   non-flattering reading — no combined headline in either form — is
   the one that governs, and the runner's U1 text now says so
   explicitly when LOSES < 3. Thresholds unchanged.
2. **U2 HARMLESS.** The frozen text says HARMLESS needs "no slice ≥
   0.05" with no size filter; the first runner filtered to n ≥ 30. Now
   matches the letter. Did not fire on this corpus (every slice has
   n ≥ 30).
3. **U3 residual by cause.** The frozen text says "invalid JSON vs
   other"; the runner's buckets are `fenced` / `unfenced_other`, which
   is the same split stated from the reference's side. Did not fire.
4. **The substance check.** Added after the first run; its first
   version mislabelled leading-fenced-but-malformed bodies as "a fence
   after prose". Corrected the same day; see `CORRECTIONS.md`.
5. **Non-object returns.** The frozen rule "non-dict → None" for the
   lenient parsers hides the lists `json_repair.loads` returns (15
   records). The artifact now banks those returns and a second slice
   table with them counted; the frozen reading is applied to the
   preregistered rule.
6. **"Frozen before the runner existed."** True by commit order
   (7e0cf36 protocol at 22:10:47 UTC; c752d14 builder, runner, tests
   and the banked run at 22:14:04 UTC). The code was written in those
   minutes against the frozen thresholds; the order is what a reader
   can verify, and it is the claim.
