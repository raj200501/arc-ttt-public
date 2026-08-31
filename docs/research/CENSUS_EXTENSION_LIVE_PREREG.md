# Census extension: live packages — preregistration

**Frozen 2026-08-31, before any package in the frame was scanned.**

## Why this run exists

A reviewer rejected the application with: *"a live, load-bearing
target — demonstrate the defect changing a number people currently rely
on — not a package frozen since May 2024 and example scripts."* He is
right that the original census frame cannot answer him: it was chosen
for exposure to the defect, not for liveness, and its sharpest confirmed
instance sits in `openai/evals`, whose last PyPI release is 28 months
old.

So this extension asks the question he asked, against packages that
cannot be dismissed the same way. **The result is binding either way:**
if the defect appears in maintained scoring paths, the "dead code"
objection is answered with instances; if it does not, that is published
at full size and the objection stands — a base rate of zero in live
code is a fact about the wedge, not a wording problem.

## The frame, frozen before scanning

Every Python package satisfying ALL of: (a) an LLM-evaluation or
LLM-scoring library or harness, (b) installable from PyPI, (c) **a PyPI
release dated within 6 months of 2026-08-31** (checked from PyPI
metadata and recorded per package). Named now, scanned in this order:

1. `lm_eval` (EleutherAI lm-evaluation-harness)
2. `deepeval`
3. `inspect_ai` (UK AISI)
4. `autoevals` (Braintrust)
5. `openevals` (LangChain)
6. `evaluate` (Hugging Face)
7. `mlflow` (its LLM-evaluate subsystem only)
8. `arize-phoenix-evals`

A package failing the 6-month test is recorded as OUT OF FRAME with its
release date, not silently dropped. Nothing may be added to or removed
from this list after scanning begins.

## The instrument and the rule, frozen

The shipped tool (`tools/fencecheck.py` at the commit carrying this
file) runs `scan` over each package's installed source. **Every finding
is hand-adjudicated under the census's original rule:** CONFIRMED only
if model output demonstrably reaches the parse AND the swallowed failure
becomes a score, a dropped item, or a silent skip in an evaluation
path. Everything else is REJECTED with its reason published. The same
conservatism as the original census applies: an unproven path is a
rejection.

## What may be claimed afterwards

Only the counts this run produces, stated with the frame. No
extrapolation from this frame to "the ecosystem." If the count is zero,
the outbound sentence is "zero confirmed in the eight maintained
packages we scanned," not silence.
