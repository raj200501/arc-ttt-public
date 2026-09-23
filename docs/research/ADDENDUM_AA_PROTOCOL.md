# Addendum AA — Addendum X's decoder isolation, read on the four families whose determinism gates did not fail (preregistration)

**DRAFT, NOT FROZEN (2026-09-23): nothing here has been stamped and no
cell has been read. When frozen it will read: frozen before the reader
ran on any cell — not before the data existed, and not before any of
it was seen (below). Anchored as
`ADDENDUM_AA_PROTOCOL.md.STAMP_PLACEHOLDER.ots`; the anchored bytes are
kept beside it as `snapshots_ADDENDUM_AA_PROTOCOL_STAMP_PLACEHOLDER.md`,
so errata appended to this file later never break the check. The reading
code is covered by the same anchor through its digests:
`scripts/decoder_isolation_read.py` sha256 `READER_SHA_PLACEHOLDER` and
`scripts/cord_decoder_isolation.py` sha256 `XMODULE_SHA_PLACEHOLDER`. The
reader refuses to bank its reading unless the proof commits to the
snapshot, this file begins with the snapshot, both digests match, and the
proof carries a Bitcoin attestation — so the reading cannot precede the
block that dates this protocol. It records its own start and finish
clocks and the git HEAD.**

## Why this exists

Addendum X (2026-09-16) was built to separate three things Addendum V
could not: the JSON **constraint**, the **decoding path** (our loop
against `model.generate`), and the checkpoint's **generation defaults**.
It put a determinism gate in front of its reading. The gate failed on
`microsoft/Phi-3-mini-4k-instruct` in both of its runs, and the frozen
protocol made a failed gate terminal for the whole addendum, so nothing
was read. X's artifact calls the other four families' cells *"raw data
for a successor addendum, not results of this one"*
(`experiments/cord_decoder_isolation_2026-09-16.json`,
`what_this_does_not_say`), and the roadmap has named this successor as
next since 2026-09-17.

**What Addendum Y explained, and what it did not.** Y (2026-09-17)
measured part of the Phi-3 failure. The constrained-arm flake on
`cord-004` in X's first gate run is reduction order: that document
decodes differently at four threads and at one, and the text X's first
run flaked to is byte-identical to Y's one-thread decode (a post-hoc
check, not one of Y's readings). Every pair Y ran reproduces inside one
process at a fixed thread count. What Y did **not** explain is the
failure present in **both** gate runs: the `generate` comparator on
`cord-001` and `cord-002` differs from the cell banked on 2026-09-03, and
Y found Phi-3's `generate` pass differing from that cell on 4 of 10
documents today, two of them (`cord-007`, `cord-008`) outside X's five
gate documents — for reasons the unrecorded 2026-09-03 environment cannot
name. That also shows what a five-document gate is: a sample that can
miss drift. This addendum does not read Phi-3.

## What changes from X, named as a choice with a direction

Everything X froze is used as frozen **except one rule: the gate's
terminality for the whole addendum**, which this addendum sets aside for
four families. By X's own words a failed gate means *"the whole addendum
withholds"*. Reading four families anyway overrides that rule. It is not
a neutral change of scope, and what it buys is named here:

- It publishes readings X's frozen rule said must not be published, and
  some of them can only help us. *Our loop reproduces `generate`
  token-for-token* is a sentence in our favour, and X predicted PATH
  REPRODUCES on SmolLM2.
- Some readings it publishes can go against us: CONSTRAINT HURTS on any
  family, V'S ATTRIBUTION IS TOO STRONG on Qwen2.5-0.5B, or the decoder's
  own accounting being wrong.
- The costlier alternative is not taken: re-running X in full, with no
  reused arm at all, so that no reuse gate is needed and Phi-3 could be
  read too. It needs the box that is running Addendum Z for the next
  40–60 hours, and it is named here so the choice is visible.

The families read are `qwen2.5-0.5b`, `smollm2-1.7b`, `granite-2b` and
`falcon3-1b`. For the three whose arm C is reused from V, X's gate passed
with 0 mismatches over 5 documents × 2 reused arms. Granite's gate is
**NOT APPLICABLE** (n = 0): its arms were all decoded inside X's own run
under a pinned prompt, so nothing was reused, and the artifact records
it as NOT APPLICABLE, never as passed. Y re-decoded two of X's reused
families, ten documents each at four threads. On Qwen2.5-0.5B (float32)
the constrained pass matched V's cell and the `generate` pass matched
S's cell, 10 of 10 each, compared after the shipped fencecheck strip
(Y's comparison rule, weaker than the byte-identity X's gate used); five
of those ten are X's gate documents. On Phi-3 it is the mixed picture
above.

**The combined X1 claim.** X's combine rule licenses *the constraint
removes invalid JSON* only if CONSTRAINT REMOVES fires in at least 4 of 5
families and no family fires HURTS. Phi-3 is counted as a family where
REMOVES did not fire, so the denominator stays five. But leaving Phi-3
unread also takes it **out of the HURTS veto**: REMOVES on the four read
and HURTS on Phi-3 would satisfy X's rule as coded. So the reader never
emits X's unscoped HOLDS sentence. If the rule would read HOLDS, the
reader emits *"X1 HOLDS ON THE FAMILIES READ ONLY: CONSTRAINT REMOVES in
4 of 5 families (4 read; phi3-mini not read, so the condition that no
family fires HURTS is unverified on 1 of 5)"*, and the words *"no family
is an exception"* are never written. Wherever X's wording says *families
tested*, the reader writes *families (4 read; phi3-mini not read)*.

## X's code, moved and corrected

The per-family arithmetic is X's reader function `read_families`, moved
out of X's `read()` on 2026-09-23 so the successor reads with X's code
and not a copy. Against X's banked commit `b4facac` the diff is: the
function boundary; the loop header, which now iterates the families
asked for; the gate-free path no longer importing the scorer before the
gate check (a side-effect order, nothing computed); and two corrections
under the rule that the protocol wins over its code:

1. **The X2 prediction string** in X's reader said Granite was predicted
   to reproduce. X's protocol says Granite is not predicted. Never
   banked (X withheld); corrected to the protocol's text.
2. **The X2 token-prefix test counted `generate`'s own stop token as a
   divergence.** Our loop breaks on EOS without appending it; `generate`
   keeps its stop token in the ids it returns. Where the plain arm
   stopped on EOS, the code as first written therefore counted a
   divergence where X's protocol says the two *"must end in the same
   place"*. The comparison now drops one trailing token from the
   generate arm when it is in that arm's own EOS set
   (`effective_generation_config`). **This correction moves X2 toward
   PATH REPRODUCES, the direction that favours us**, so the reader
   publishes both counts per family, as first coded and as
   protocol-conforming, and names the documents that differ between
   them. It was found from the code alone by the pre-freeze audit,
   whose agents were instructed not to compute any quantity from the
   cells. Decoded-text comparisons carry no stop token and are
   unaffected.

No test in X's suite executes `read_families`; the equivalence of the
move rests on the diff, which a stranger reproduces with
`git diff b4facac -- scripts/cord_decoder_isolation.py`, and on a
synthetic-cell test added with this addendum.

## The disclosure this addendum owes

**Nothing here is read blind.** X's artifact records that during X's run
the operator computed the per-family X1 quantities for these families
by hand, for scheduling, and published none of them. The record names
X1 only and is silent on X2 and X3, so all three are treated as possibly
seen. X's readings and combine rule were committed on 2026-09-16 at
05:52Z (`0e5a11c`), before the first plain cell by the cells' own file
times; X's anchor came after its run, and `ANCHORS.md` records that it
does not prove freeze-before-run. So *by our own record* what the
operator saw could not have shaped X's readings, and a stranger cannot
check that record. **The decision to read these cells, and every choice
in this document, were made after the hand computation.** Each choice is
listed above with its direction for that reason.

## What is published first, before any contrast

**Prompt identity is checked per comparison and has three states:**
verified by sha256 per document, **not verifiable** because the reused
cell banks no prompt digest, or MISMATCHED. Arm C reused from V banks no
digest (Qwen2.5-0.5B, SmolLM2, Falcon3: C vs P not verifiable), and the
`generate` comparator cells G₀ bank text only (SmolLM2 and Falcon3: P vs
G₀ not verifiable; Qwen2.5-0.5B: G vs G₀ not verifiable). Verified pairs:
Granite C vs P, and P vs G on Qwen2.5-0.5B and Granite. For every pair
that is not verifiable, prompt identity rests only on X's five-document
determinism gate, and the artifact lists those pairs by name.

In this order, each one if it fires:

1. **Prompt identity MISMATCHED** on any compared pair of a family: that
   family is **VOID**, its readings are published as VOID, and no
   sentence about its decoder isolation is licensed.
2. **The decoder's own accounting is wrong** (X's frozen prediction,
   checked per document): a document with 0 banked constrained steps
   whose arms C and P differ, or one with steps whose arms are identical.
   It publishes in X's words: *"the decoder's own accounting is wrong,
   and that is the result — a larger and worse one than the contrast
   this addendum was built to make."*
3. **V's published regression count does not reproduce** from its own
   banked cells: X3's attribution is withheld — its explained count and
   residual attribution are banked as null — and the disagreement
   publishes first.

## Predictions

This addendum adds none. X's, verbatim (`ADDENDUM_X_PROTOCOL.md`, "Two
predictions, written before the run"), with what four families can
check:

- **X1:** *"arms C and P must be byte-identical on every document with 0
  constrained steps [...] so X1 reads INERT on Qwen2.5-0.5B, SmolLM2,
  Granite and Phi-3, and on Falcon3 the arms differ on exactly the seven
  documents where the validator fired."* Checkable on three of the four
  families predicted INERT, and on Falcon3; the per-document accounting
  is checked on all four read. Phi-3's INERT prediction and its
  accounting are listed as not checkable.
- **X2:** *"PATH DIVERGES on Phi-3 [...] and PATH REPRODUCES on SmolLM2.
  Qwen, Falcon3 and Granite are not predicted."* SmolLM2 is checkable,
  on the decoded-text basis, and is banked with that basis beside it.
  Phi-3's is not checkable.

## What may be said afterwards

The reader emits the licensed sentences; nothing else may be written
about these families' decoder isolation.

- **X1**, per family, as X's reading string. Where arm C is V's reused
  cell and the arms differ, the sentence carries: *the difference is the
  constraint, or V's cell not reproducing across processes on that
  document, and this reading cannot separate the two.* The combined X1
  sentence is the scoped one above. A HURTS family is named with its
  numbers at full size.
- **X2**, per family: *our loop reproduces `generate` token-for-token on
  these fifty documents* only where X2 reads PATH REPRODUCES on the token
  basis (Qwen2.5-0.5B, Granite); *reproduces `generate`'s decoded text on
  these fifty documents (WEAKER TEST)* on the text basis (SmolLM2,
  Falcon3). PATH DIVERGES publishes with its count, its basis and the
  invalid-output counts of both arms; on the text basis it adds that the
  comparator cell carries no environment record, so path divergence and
  drift in the banked cell are not separated. (That last clause is this
  addendum's own addition to X's wording, made in the direction against
  us.)
- **X3** on Qwen2.5-0.5B, as X's reading string.

**The same day, by obligation** (the reader lists which fire): whatever
X1 reads, `tools/README.md`'s `jsongreedy.py` section — which still
says *"Addendum X is preregistered and running"* beside the paragraph
saying X withheld — carries this addendum's reading. If X1 reads HURTS
anywhere, the paragraph beginning *"What it does and does not do is
measured"* names that family at full size. If X3 reads **V'S ATTRIBUTION
IS TOO STRONG**, V's sentence is narrowed to *"E of the R_V"* by a dated
erratum beside `ADDENDUM_V_PROTOCOL.md`, in `VERDICT.md`'s Addendum V
row, and in `CORRECTIONS.md`. No sentence about Phi-3's decoder
isolation may be written at all.

## What this cannot show

Everything X said it could not, plus: one family fewer, so the combined
X1 claim is harder to reach and X2 has no bfloat16 family. Fifty
receipts per family, one corpus, schema-only prompts, CPU, 0.5B to 2.5B
parameters (Granite-3.1-2B is 2.53B), one box. The reused arms were gated
on five documents per family, a sample, and most compared pairs cannot
have their prompt identity verified at all.

Artifacts read: `experiments/cord_decoder_isolation_cells/`,
`experiments/cord_constrained_families_cells/` and the S/T comparator
cells X names, `experiments/generation_configs_2026-09-16.json`,
`experiments/cord_constrained_families_2026-09-08.json` (V's published
regression counts). Artifact written, once, then anchored:
`experiments/cord_decoder_isolation_read_2026-09-23.json`. Reader:
`scripts/decoder_isolation_read.py` (`--check` reports the preconditions
and reads nothing; `--out` recomputes into another path and is refused
until the banked reading exists). Tests:
`tests/test_decoder_isolation_read.py`, on synthetic cells only.
