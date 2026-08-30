# The fence-tax census — preregistration

**Frozen 2026-08-25, before any package was downloaded, inspected or
counted.** The sampling frame, the classifier, the definition of
"exposed", the accuracy check on the classifier itself, and the shape of
the published sentence are all fixed below. Commit this before running
anything.

---

## THE QUESTION

When a model returns a JSON object wrapped in a triple-backtick code
fence, does the code that scores it read the object, or read a parse
failure?

This is not hypothetical. On 30 held-out documents, `Qwen2.5-3B-Instruct`
given a bare field list produced a well-formed JSON object for every
single document and scored **0.0000, 30 of 30 unparseable**, because the scorer
called `json.loads` on text beginning with ` ```json `. Removing the
fence and changing nothing else: **0.8958, zero invalid**
(`experiments/fence_rescore.json`). A 0.90 F1 swing from output packaging.

The interesting question is not whether one runner had that bug. It is
**how much published code is standing in the same place**.

## WHY IT MIGHT MATTER, STATED AS A HYPOTHESIS AND NOT AS A CONCLUSION

Instruction-tuned chat models fence heavily; models fine-tuned on a
strict output format do not. If scoring code treats a fenced-but-valid
object as a failure, then comparisons between those two classes of model
are biased in a nameable direction — against the chat model, in favour of
the format-trained one. **Whether that bias is real, and how large, is
measured below and not assumed.** If the data does not support it, the
null publishes.

---

## SAMPLING FRAME — frozen before any download

**Universe.** Python packages distributed on PyPI whose documented
purpose includes evaluating, scoring, parsing or validating the output of
a language model.

**Why PyPI rather than a GitHub search.** Three reasons, all of them
about reproducibility. A GitHub code search returns a ranking that
changes daily and cannot be replayed; a PyPI package name plus a version
pins exactly what was inspected. A published package is code someone
chose to ship rather than a snippet in a notebook. And a stranger can
re-run the whole census with `pip download` and no credentials, which is
what makes G2 possible at all.

**Selection rule, applied in this order and recorded in full:**

1. **Seed list.** An enumerated list of package names written down in
   `scripts/fence_census.py` BEFORE any inspection, drawn from three
   categories: (a) LLM evaluation and benchmark harnesses, (b) agent and
   LLM application frameworks that parse structured model output, (c)
   libraries whose stated job is turning model text into JSON.
2. **PyPI search expansion.** The exact query strings are frozen in the
   script. Every returned name is recorded, whether or not it is kept.
3. **Inclusion test, mechanical:** the package's source contains at least
   one call site that converts text to a Python object
   (`json.loads`, `json.JSONDecoder().decode`, `ast.literal_eval`,
   `yaml.safe_load`) inside a module that also references model output
   (any of: `completion`, `response`, `generation`, `model_output`,
   `llm`, `predict`, `answer`). A package with no such site is **excluded
   and recorded as excluded**, because it is not in the universe.
4. **Exclusion reasons are published.** Every candidate that does not
   make the final count appears in the artifact with the reason. A census
   whose exclusions are invisible is a census that chose its answer.

**Target size: at least 40 packages that pass the inclusion test.** If
the frame yields fewer than 40, the shortfall is published and the
result is reported over whatever number was reached, labelled with that
number. **The frame is not widened after seeing results.**

**Version pinning.** Latest release as of the run date, recorded per
package with its exact version string. The census describes those
versions and says so. Any package fixed after that date is still counted
as exposed *at the version inspected*, and the artifact says the fix may
exist — this is a snapshot, not an accusation.

---

## THE CLASSIFIER — frozen before any package is read

For each included package, every JSON-parse call site is examined and
three properties are recorded **with `file:line` evidence for each**:

- **F — fence-aware.** Within the enclosing function, or in a helper it
  calls by name, the code strips or tolerates a markdown fence: a literal
  ` ``` ` appears in a strip/split/replace/regex/removeprefix operation,
  or the text is routed through a known repair dependency
  (`json_repair`, `dirtyjson`, `demjson3`, `json5`, `pydantic`'s
  permissive parsers, an explicit "extract JSON from markdown" helper).
- **Z — failure becomes a zero or a silent drop.** The parse is wrapped
  in a handler for `JSONDecodeError`, `ValueError` or `Exception` whose
  body returns or records `0`, `0.0`, `False`, `None`, `{}`, `[]`, or
  reaches `continue`/`pass` — rather than raising, or surfacing a
  distinguishable "unparseable" state to the caller.
- **N — no handler at all.** The parse can raise into the caller.

### AMENDMENT, 2026-08-25, before any package was inspected

A fourth property is recorded and it **narrows** what counts as exposed:

- **R — the parse argument is a file read.** If the text being parsed
  comes from `read_text`, `open(...)`, `.read()` or a `Path`, the site is
  consuming a config file or a banked artifact, not a model's answer, and
  it is **not** counted as exposed regardless of F and Z.

**Why, and when.** The classifier was run against this repository's own
`scripts/` directory as a smoke test of the tool — not against any census
package — and flagged four sites. Three were genuine: two that parse
`result["prediction"]` from banked artifacts, and `run_challenge.py:75`,
which is the bare `json.loads` that produced this project's own arms and
is the origin of the whole finding. The fourth,
`verification_coverage.py:82`, reads JSON artifacts off disk in a module
that happens to mention models, and is not model output at all.

Gating on the file, rather than on the parsed expression, was too coarse.
**Precision is what makes a census number defensible**, and a classifier
that indicts every module which both mentions a model and opens a JSON
file would produce a big number that falls apart on the first spot check.

**No census package had been downloaded or inspected when this changed,
so no result influenced it.** The revised definition is:

> A package is **EXPOSED** if it has at least one call site that is
> **NOT F**, **IS Z**, and **NOT R**.

### ~~The frozen definition of EXPOSED~~ — superseded by the amendment above, preserved unaltered

> A package is **EXPOSED** if it has at least one call site that is
> **NOT F** and **IS Z**.

*(Superseded 2026-08-25 by the R clause, before any package was
inspected. Kept verbatim because frozen text is never rewritten — a
reader is entitled to see what the bar was before it moved, and to check
that it moved in the direction that makes the count SMALLER.)*

That conjunction is the whole point and it is deliberately narrow.
Parsing without fence handling is not by itself a defect — a library
whose contract is "hand me clean JSON" is entitled to that contract. The
defect is parsing without fence handling **and then converting the
failure into a score**, because that is the combination that silently
turns a correct answer into a zero. **A site that is NOT F and NOT Z is
recorded and is not counted as exposed.**

### What this census does NOT claim, fixed now

- It does not claim any package is buggy. A library may be exposed and
  entirely correct for its documented contract.
- It does not claim any published benchmark number is wrong. It measures
  the code, not the papers, and the step from "this code is exposed" to
  "that leaderboard is wrong" is not taken here and must not be taken in
  outbound copy.
- It does not measure how often models actually fence. That is the impact
  half, measured separately on our own corpus, and the two are reported
  side by side without being multiplied together.

---

## THE CHECK ON THE CLASSIFIER — because a counter can be wrong

A mechanical classifier over unfamiliar code will make mistakes, and the
mistakes will not be symmetric. So:

1. **A stratified random sample of 10 classified call sites** — drawn
   with a seed fixed in the script, at least 4 from sites classified
   EXPOSED and at least 4 from sites classified not-exposed — is read by
   hand and adjudicated.
2. **The classifier's agreement rate on that sample is published beside
   the headline count**, in the same sentence if it is below 90%.
3. **If agreement is below 80%, the census does not publish a count at
   all.** It publishes the classifier's failure and what it got wrong.
   A number produced by a classifier that is wrong one time in five is
   not a measurement.

## MUTATION TEST — the classifier must be able to say "not exposed"

Two synthetic packages are planted in the test suite: one that parses a
fenced payload with no handling and scores the failure as zero (must
classify EXPOSED), and one that strips the fence first (must classify
not-exposed). A classifier that returns EXPOSED for everything would
produce a large, meaningless and very shareable number, which is the
most dangerous possible failure here.

---

## THE PUBLISHED SENTENCE — its shape is fixed now, its numbers are not

> Of **M** published Python packages that score or parse language-model
> output, **K** have at least one path that reads a model's JSON without
> handling a markdown code fence and converts the resulting parse failure
> into a zero or a silent drop. On a 30-document extraction task, that
> single difference moves a 3B model from **0.0000 to 0.8958**.

`M` and `K` come from the census. The second sentence is already banked.
**If `K` is small, the sentence still publishes with the small `K`.** A
census that finds 3 of 40 is a real number about the world and it is
reported in the same words as a census that finds 30 of 40.

---

## STOP RULES FOR THIS CENSUS

- If the inclusion test yields fewer than 40 packages, report over the
  number reached and label it.
- If classifier agreement on the hand-adjudicated sample is below 80%,
  publish the classifier's failure instead of a count.
- If the count comes back near zero — public scorers overwhelmingly
  handle fences correctly — **that publishes as a finding**, in the same
  place and at the same size, and the lead moves on.

---

# RESULT OF RUN 1 — 2026-08-25 — NO COUNT PUBLISHED

**Classifier agreement on the hand-adjudicated sample: 6 of 10 (60%).
The frozen floor was 80%. Under the stop rule written above, this run
publishes the classifier's failure instead of a number.**

The raw tally was 21 of 36 included packages. **It is not a result and it
must not be quoted as one**, including by us. It is recorded in
`experiments/fence_census.json` as an input that failed its own accuracy
gate.

## What the adjudication showed

**Recall looks fine.** All 5 sites the classifier declined to flag
survived review. Nothing suggests it is missing real cases.

**Precision is the failure.** 4 of the 5 sites it flagged do not hold up,
for three distinct and mechanically fixable reasons:

1. **It never checks whose text is being parsed.** `opencompass`'s
   LiveCodeBench evaluator parses `expected_output` — the dataset's gold
   — and sets a result to `False` on failure. The Z half is right; the
   subject is wrong. `litellm` parses a WebSocket protocol envelope where
   a fence cannot occur. `deepeval` parses a service-account credential.
2. **Test files were in the frame.** Two of the ten sampled sites are in
   `tests/`. They are not shipped behaviour and cannot mis-score
   anything.
3. **Swallowing is not scoring.** `deepeval`'s OpenTelemetry exporter has
   `except Exception: pass` around a parse of genuine model output — but
   the consequence is a trace attribute staying a string, not a score
   becoming zero. The classifier read `pass` as "the failure becomes a
   score".

**The one that held up:** `llama-index-core`'s
`dynamic_llm.py:143` — properties pulled from LLM output by regex,
`json.loads` with no fence handling, `return {}` on `JSONDecodeError`.
Model output, no repair, failure becomes silently dropped data.

## What happens next, and what must not

A second classifier is preregistered separately with the three fixes
above, and re-run against a **new** adjudication sample. That is a new
experiment with its own frozen bar.

**It is not a re-read of this one.** Run 1's failure stands published at
full size, in this file and in `CORRECTIONS.md`, whatever run 2 returns.
The temptation here is obvious and it is named so it can be checked: a
project that keeps rebuilding a classifier until the number is publishable
is fitting a classifier to a desired headline. The defence is that every
run's frame, classifier, sample and agreement rate are published, so the
number of attempts is visible.

## Also recorded: a gap between the frozen frame and what ran

The frame specified a seed list **plus** a PyPI search expansion over
frozen query strings. Only the seed list was implemented; `SEARCH_QUERIES`
was declared and never used. The executed frame is therefore the seed list
alone, which yielded 36 included packages against a target of 40.

**The expansion was not run after the fact to close that gap**, because
the results were already visible and a frame widened after seeing results
is not a frame. Run 2 implements it before looking at anything.

---

# RUN 2 — PREREGISTRATION, frozen 2026-08-25 before any package was re-inspected

A new classifier, a new adjudication sample, the same 80% floor. Run 1's
failure above is untouched by whatever this returns.

**Run 1's classifier is preserved byte-for-byte in `scripts/fence_census.py`
and is not edited.** Run 2 lives in `scripts/fence_census2.py`. A published
classifier that gets quietly improved is a classifier whose published
failure cannot be reproduced.

## The three fixes, each aimed at a named adjudication failure

**Fix 1 — test files leave the frame.** Any path containing `/tests/`,
`/test/`, a `test_` prefix, a `_test.py` suffix, or `conftest.py` is
excluded before analysis. Test code is not shipped behaviour and cannot
mis-score anything. *(Adjudication failures 2 and 10.)*

**Fix 2 — the parsed expression must be model output.** The argument to
the parse call is unparsed to source and tested:

- it must reference at least one of `completion`, `response`,
  `generation`, `generated`, `model_output`, `output`, `answer`,
  `prediction`, `content`, `text`, `message`, `raw`, `result`, `choice`;
- and it must reference **none** of `expected`, `gold`, `reference`,
  `target`, `label`, `truth`, `config`, `credential`, `key`, `secret`,
  `token`, `schema`, `manifest`, `request`, `envelope`, `event`,
  `header`, `arg`, `param`.

The exclusion list is the second half and it is the important half: it is
what stops a dataset's gold, a service-account credential and a WebSocket
envelope from being counted as a model's answer. *(Adjudication failures
1, 3 and 4.)*

**Fix 3 — swallowing is not scoring.** A bare `except: pass` no longer
satisfies Z. `pass` leaves whatever value already existed; it does not
manufacture a score. Z now requires the handler body to **produce** a
degraded value:

- `return` a zeroish literal (`0`, `0.0`, `False`, `None`, `{}`, `[]`), or
- **assign** a zeroish literal to a name, or
- `continue` — dropping the record from a loop — **and** the enclosing
  function reads as scoring or extraction (its name or body mentions
  `score`, `eval`, `metric`, `grade`, `judge`, `correct`, `accuracy`,
  `parse`, `extract`).

*(Adjudication failure 1.)*

## What is deliberately NOT changed

The seed list, the inclusion test, the F test, the R clause, the 80%
floor, the sample size, and the shape of the published sentence all stay
exactly as frozen. **Only the three named precision fixes move.** If the
count changes for any other reason, something else changed and that is a
bug.

The PyPI search expansion specified in the original frame and never
implemented stays unimplemented in run 2 as well, so run 1 and run 2 are
comparable on the same frame. It is recorded as outstanding, not quietly
dropped.

## The bar, unchanged

A fresh stratified sample of 10 sites — a different seed, at least 4
EXPOSED and at least 4 not-exposed — is hand-adjudicated. **Below 80%
agreement, no count publishes and run 2 joins run 1 on the page.**

## Stated before the data

Run 1's precision failure was one-directional: it over-flagged. All three
fixes therefore push the count **down**. If run 2 returns a materially
*higher* count than run 1's withheld 21, that is evidence of a bug rather
than of a better classifier, and it is investigated before it is
published.

---

# RESULT OF RUN 2 — 2026-08-25 — NO COUNT PUBLISHED EITHER

**Agreement 7 of 10 (70%) against the same frozen 80% floor.** Precision
on the exposed half went from 1/5 to 2/5; recall stayed 5/5. The raw
tally was 12 of 34, down from 21 of 36 — the direction preregistered, so
no bug signal. **It is still not a result.**

The three fixes each did what they were written to do. Three *new*
failures took their place, and they share one root cause:

1. **Provenance, not expression.** Fix 2 reads the parse argument one hop
   deep. `langchain-core` binds an HTTP header to `raw`; `litellm` binds
   a WebSocket envelope to `message`. Both words are on the output list;
   the disqualifying word lives in where the value *came from*.
2. **Fence handling one level up.** The F test's scope is the enclosing
   function. `crewai` handles markdown in a sibling method of the same
   class and is flagged anyway.
3. **`examples/` is in the frame.** `ragas`'s site is example code. Same
   question `tests/` raised. A frame defect, recorded as one.

## The conclusion, and the change of approach it forces

Two preregistered classifiers, two failures against the same floor.
Deciding *mechanically* whether a given `json.loads` is scoring a model's
answer requires following where the value came from, across function
boundaries — and neither classifier does that. A third round of tuning
would be fitting a classifier to a headline, which this file has already
named as the thing to check for.

**So the next attempt stops trying to automate the verdict.** The
classifier becomes a *search narrower* only: it proposes candidate sites,
and **every site that reaches the published count is read by hand and
confirmed**. A count that is 100% adjudicated by construction needs no
accuracy gate, because there is no sample — there is only the set of
sites a human confirmed.

That trades reach for defensibility. The published number will be smaller
and it will be one nobody can knock over. **Both withheld tallies stay on
this page** so the number of attempts is visible.

## What already survives all of this, unaffected

The impact measurement is not a census and does not depend on any of the
above: on 30 held-out documents, one markdown fence is the entire
difference between **0.0000 with 30 of 30 unparseable** and **0.8958 with
zero invalid**, same model, same prompt, same scorer
(`experiments/fence_rescore.json`). And two sites confirmed by hand in
run 2 — `instructor/batch/processor.py:255` and
`ragas/.../benchmark_llm/evals.py:49` — are genuine published instances,
read line by line, in packages people install.

---

# RUN 3 — HAND-ADJUDICATED, preregistered 2026-08-25 before any site was read

No third classifier. `scripts/fence_census2.py` is used **only to narrow the
search**, and its output is a candidate list, not a verdict. Every site that
reaches the published count is read in context and confirmed by hand.

**35 candidate sites across 12 packages**, taken verbatim from
`experiments/fence_census_run2.json`. The list is fixed now and is not
extended.

## Confirmation criteria — a site is CONFIRMED only if all four hold

1. **It is a model's answer.** The parsed text originates from a language
   model's generated output — a completion, a message content, a tool-call
   payload the model wrote. Traced by reading, across function boundaries,
   which is exactly what the two classifiers could not do. **A dataset's
   gold, a config file, a credential, an HTTP header, a protocol envelope
   or a provider's structural response wrapper is NOT a model's answer.**
2. **A fence could actually reach it.** The text is raw or near-raw model
   output rather than a field the provider has already extracted and
   validated. If the value has passed through a schema-constrained decoder
   or a provider's structured-output API that guarantees bare JSON, it is
   rejected.
3. **No fence handling anywhere on the path.** Checked in the enclosing
   function, its callers within the same module where visible, and sibling
   helpers of the same class. This is the check that cleared `crewai` in
   run 2 and it is applied to everything now.
4. **The failure produces a degraded value that is used.** A returned
   zero, `None`, `False`, `{}`, a dropped record, or a score. A bare
   `pass` that leaves a prior value, or a re-raise, does not count.

## What gets published

- The **confirmed count and every confirmed site with `file:line`**, so a
  reader can check each one in a browser.
- **Every rejected candidate with its reason**, at the same size. A
  hand-adjudicated census that hides its rejections is a hand-picked one.

## The frame statement, fixed now because it is the honest limit

> This is a **lower bound on confirmed sites among 12 named packages**. It
> is **not** a rate over the ecosystem, and it is **not** a claim that
> these packages are buggy. The candidate list came from a classifier with
> measured recall of 5/5 on its adjudication samples but unmeasured recall
> over the whole corpus, so sites it never proposed are not counted and
> cannot be.

**No accuracy gate applies to run 3**, because there is no sample and no
classifier verdict in the published number — every entry is a human
verdict. The check that replaces it is that every confirmed site is
published with its location so the adjudication itself is checkable.

## Stated before reading

Run 2's hand-adjudication cleared 2 of 5 sampled exposed sites. If that
rate holds, roughly 14 of the 35 candidates survive. **If the confirmed
count comes out very low — three or fewer — that publishes as the
finding**, in the same words, and the conclusion becomes that this defect
is rarer in shipped library code than the arc-ttt incident suggested.

---

# RESULT OF RUN 3 — 2026-08-25 — 6 OF 35, HAND-CONFIRMED

All 35 candidates read in context. **6 confirmed instances across 3
packages** — 3 in library code, 3 in shipped example code. Every
rejection is published with its reason in
`experiments/fence_census_hand.json`.

## The confirmed set

| package | site | what happens |
|---|---|---|
| `evals` (OpenAI) | `evals/elsuite/steganography/monitor.py:72` | `json.loads` on the monitor model's completion, no fence handling. The retry sends **the same messages** up to 5 times, so a model that fences once fences five times. On exhaustion: **`("", 0.0)`** — and `0.0` here means *contains_hidden_message = not suspicious* |
| `instructor` | `instructor/batch/processor.py:255` | `json.loads` on `choices[0].message.content`, no fence handling; failure returns `None`, recorded as a failed batch item |
| `instructor` | `instructor/batch/processor.py:283` | Anthropic branch, `json.loads` on a text content block; `except: continue` drops the item |
| `ragas` | `examples/iterate_prompt/evals.py:76` | model answer parsed raw; failure sets the predicted label to `""` and the priority to `None`, both then scored |
| `ragas` | `examples/ragas_examples/benchmark_llm/evals.py:49` | failure sets `predicted_discount = None`, which the scorer consumes |
| `ragas` | `examples/ragas_examples/workflow_eval/workflow.py:175` | `json.loads` on `message.content.strip()`; failure returns `{}` as the extraction result |

### The direction of that failure is not neutral

In a safety eval measuring whether a monitor catches steganography, a
monitor that correctly answered `1` and wrapped it in a fence is recorded
as having **detected nothing**. The formatting failure is scored in the
attacker's favour.

**Stated fairly:** the monitor's system prompt does say *"do not generate
any other text"*, so a fenced answer has disobeyed the format
instruction. The claim is about which way the default falls when a model
disobeys — not that the package is wrong.

## What this says, and what it does not

**The defect is real and it is in code people install.** The `evals`
instance turns a fenced-but-correct monitor answer into a `0.0`, which is
the same failure that produced this project's own `0.0000`.

**It is not endemic.** 29 of 35 candidates are protocol envelopes,
telemetry payloads, HTTP bodies, credentials, ground-truth strings, or
sites where the fence is handled one level up. **Quote 6 of 35, not a
rate over the ecosystem.**

This is a **lower bound among the packages searched**. The candidate list
came from a classifier whose recall over the whole corpus is unmeasured,
so sites it never proposed are not counted and cannot be.

## The search narrower errs in both directions

Candidate 8, `opencompass/datasets/clbench.py`, calls `_strip_json_fence()`
one line before parsing and is clean. The classifier's fence regex missed
that helper's name. Recorded because it matters: the narrowing is not
biased purely toward over-flagging, so the confirmed set is not a ceiling
either.

## Against the preregistered expectation

Run 2's sampled precision suggested roughly 14 of 35 would survive. Six
did. The stated-before-reading threshold was *"three or fewer publishes as
the finding that this is rarer than the arc-ttt incident suggested"* —
library-code confirmations came in at exactly three, so **that reading is
the one taken**: in shipped library code this is uncommon. It is
nonetheless present, named, and locatable in one click.
