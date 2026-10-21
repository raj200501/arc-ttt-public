# Addendum AA — Addendum X's decoder isolation, read on the four families whose determinism gates did not fail (preregistration)

<!-- DRAFT-HEADER-START -->
**DRAFT, NOT FROZEN: nothing here has been stamped and no cell has been
read. `scripts/freeze_protocol.py prepare` removes this paragraph and
fills the placeholders below; nothing else changes at the freeze.**
<!-- DRAFT-HEADER-END -->
**Frozen at the stamp below, before the reader ran on any cell — not
before the data existed, and not before any of it was seen (below).
Anchored as `ADDENDUM_AA_PROTOCOL.md.STAMP_PLACEHOLDER.ots`; the anchored
bytes are kept beside it as
`snapshots_ADDENDUM_AA_PROTOCOL_STAMP_PLACEHOLDER.md`, so errata appended
to this file later never break the check.**

**What the anchor pins.** Every code file the reading imports or runs,
and one manifest that pins its input data, each by sha256:

- `scripts/decoder_isolation_read.py` sha256 `READER_SHA_PLACEHOLDER`: the reader.
- `scripts/cord_decoder_isolation.py` sha256 `XMODULE_SHA_PLACEHOLDER`: X's arithmetic (`read_families`) and the plain loop's stop id per family.
- `scripts/aa_pinned/fencecheck_x.py` sha256 `FENCECHECK_SHA_PLACEHOLDER`: `strip_fence`, which decides every fenced count and feeds every invalid count. Byte-identical to `tools/fencecheck.py` at X's banking commit (below).
- `scripts/cord_fence_tax.py` sha256 `CFT_SHA_PLACEHOLDER`: the gold split's location and reader, and the S/T cell folders.
- `scripts/cord_constrained_families.py` sha256 `VMODULE_SHA_PLACEHOLDER`: each family's model and comparator cell, V's cell folder, and the sign test.
- `src/arcttt/__init__.py` sha256 `ARCTTT_INIT_SHA_PLACEHOLDER`: the package the scorer is imported from.
- `src/arcttt/scoring.py` sha256 `SCORING_SHA_PLACEHOLDER`: `parse_json_object` and `field_micro_f1`.
- `src/arcttt/text_task.py` sha256 `TEXTTASK_SHA_PLACEHOLDER`: the parse error the classifier catches.
- `src/arcttt/tasks.py` sha256 `TASKS_SHA_PLACEHOLDER`: imported by the one above.
- `scripts/aa_pinned/aa_input_manifest.json` sha256 `INPUTS_SHA_PLACEHOLDER`: the sha256 of each of the nineteen input files the reading opens: for each of the four families, arm P, arm C (V's reused cell for three, X's own for Granite) and the G₀ comparator cell; arm G for Qwen2.5-0.5B and Granite; V's artifact; X's artifact (for the gate status); the scope record; and the gold split's two files. The manifest was written by hashing those files' bytes, none of them parsed, and only after checking that each is byte-identical to its blob at X's banking commit `b4facac`.

The reading refuses unless the proof commits to the snapshot, this file
begins with the snapshot, every digest above matches, every pinned file
is committed and unchanged in git, every input matches the manifest, no
repository module outside the list above was loaded, and the proof
carries a Bitcoin attestation. Not pinned: the Python, torch and
transformers versions, which the artifact records. The artifact banks
the proof's sha256 and the block height read from the proof, the git
HEAD, and `git status` of the pinned paths. An upgraded proof contains
the path into its block, so its sha256 cannot be known before the block;
that the reading came after the block rests on that digest and on this
code refusing otherwise. The start and finish clocks it records are ours
to report.

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
`what_this_does_not_say`). The roadmap has named a reading of those
cells as next since 2026-09-23 (`534e7fd`), and as the step after a
direct measurement of the Phi-3 instability (Addendum Y) since
2026-09-16 (`b4facac`).

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
documents on 2026-09-17, two of them (`cord-007`, `cord-008`) outside X's
five gate documents — for reasons the unrecorded 2026-09-03 environment
cannot name. That also shows what a five-document gate is: a sample that
can miss drift. This addendum does not read Phi-3.

## What changes from X, named as a choice with a direction

Everything X froze is used as frozen **except two things, each named
here with its direction.** The first is **the gate's terminality for the
whole addendum**, which this addendum sets aside for four families. The
second is **X1's combine rule**, which can no longer emit X's unscoped
HOLDS sentence and gains a scoped verdict X never defined (below). By
X's own words a failed gate means *"the whole addendum withholds"*.
Reading four families anyway overrides that rule. It is not a neutral
change of scope, and what it buys is named here:

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
with 0 mismatches over 5 documents × 2 reused arms. Granite's gate did
not apply: its arms were all decoded inside X's own run under a pinned
prompt, so nothing was reused. X's artifact banks Granite's gate as
n = 0, mismatches 0, passed = true. This addendum reads that entry as
**NOT APPLICABLE**, never as passed, and its own artifact records it so,
with X's banked values beside it. Y re-decoded two of X's reused
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
emits X's unscoped HOLDS sentence, not even quoted inside a refusal. If
the rule would read HOLDS, the reader emits *"X1 HOLDS ON THE FAMILIES
READ ONLY: CONSTRAINT REMOVES in 4 of 5 families (4 read; phi3-mini not
read, so the condition that no family fires HURTS is unverified on 1 of
5)"* and says that X's unscoped HOLDS sentence is not licensed. Wherever
X's wording says *families tested*, the reader writes *families (4 read;
phi3-mini not read)*. The direction of this change: against X's rule as
written, which cannot be met while Phi-3 is unread, the scoped verdict
is a HOLDS-labelled reading where X's rule licenses none, and that is
the direction that favours us. It is why the scope sits inside the
sentence and the unverified veto is counted in it. Against X's code as
written, which would emit the unscoped HOLDS, it is narrower.

A VOID family (below) is treated as Phi-3 is: it counts as a family
where REMOVES did not fire, so with one VOID family X1 cannot read HOLDS
at all; its HURTS veto is unverified; and its raw reading is never
consulted. Where X's wording says *families tested*, the reader then
writes *families (3 read, 1 VOID (<family>); phi3-mini not read)*.
Counting it against REMOVES does not favour us; leaving its HURTS out of
the veto does, and its per-family entry says VOID in place of a reading.

## X's code, moved and corrected

The per-family arithmetic is X's reader function `read_families`, moved
out of X's `read()` on 2026-09-23 so the successor reads with X's code
and not a copy. Against X's banked commit `b4facac` the diff is: the
function boundary; the loop header, which now iterates the families
asked for; the gate-free path no longer importing the scorer before the
gate check (a side-effect order, nothing computed); two parameters added
on 2026-10-21, the fencecheck module (by default X's live
`tools/fencecheck.py`; this addendum passes the copy X ran with) and the
plain loop's stop ids; the X2 reading string, which on the token basis
now carries both counts of the correction below; X's `--read`, which no
longer writes over an existing file (below); and two corrections under
the rule that the protocol wins over its code:

1. **The X2 prediction string** in X's reader said Granite was predicted
   to reproduce. X's protocol says Granite is not predicted. Never
   banked (X withheld); corrected to X's sentence, word for word.
2. **The X2 token-prefix test counted `generate`'s own stop token as a
   divergence.** Our loop breaks on the tokenizer's `eos_token_id`
   without appending it; `generate` keeps its stop token in the ids it
   returns. Where the plain arm stopped on EOS, the code as first written
   therefore counted a divergence where X's protocol says the two *"must
   end in the same place"*. The comparison now drops one trailing token
   from the generate arm, and only where the plain arm stopped on EOS and
   that token is the one id our loop stops on. Neither X's cells nor its
   scope record bank that id, so it is pinned per family in X's module
   from each checkpoint's `tokenizer_config.json` at the Hugging Face
   revision current before X ran (Qwen2.5-0.5B: 151645; Granite: 0), and
   the reader re-derives it from the tokenizer at read time and refuses
   on any disagreement. It is not `generate`'s own stop set, which for
   Qwen2.5-0.5B lists two ids: a different stop id at the stop position
   is a real disagreement, and it stays a divergence. The first version
   of this correction (2026-09-23) dropped any id in `generate`'s set,
   and so could have read that disagreement as agreement, in our favour;
   it was narrowed on 2026-10-21, before any cell was read. A stop id
   that both arms emitted as an ordinary token is compared as a token
   and never dropped. **This correction moves X2 toward PATH
   REPRODUCES, the direction that favours us**, so each token-basis X2
   reading string and its licensed sentence carry both counts, as first
   coded (X's code at `b4facac`) and as corrected, and the artifact names
   the documents that differ between them and the id dropped from each.
   It was found from the code alone by the pre-freeze audit, whose agents
   were instructed not to compute any quantity from the cells.
   Decoded-text comparisons carry no stop token and are unaffected.

**Where the copy of fencecheck comes from.** `strip_fence` lives in
`tools/fencecheck.py`, a shipped tool that changes for other reasons.
The reading must not move when it does, so the reader loads
`scripts/aa_pinned/fencecheck_x.py` and refuses to load the live tool.
That file is `tools/fencecheck.py` as it stood at `b4facac`, byte for
byte: its git blob is `e138847`, the same blob `b4facac` holds at that
path. X's artifact records no git HEAD, so X's run commit is taken from
git history: `b4facac` is the only commit that touches X's artifact, and
every cell X wrote first appears there. The other code files pinned
above, the two scripts from Addenda S/T and V and the four `src/arcttt`
files, hold the same bytes at `b4facac` as today. The reader and the
manifest are new, and X's module changed as listed above.

**X's own `--read`.** X's artifact names `--read` as the way to
recompute it, and until 2026-10-21 that command wrote over the anchored
artifact with a fresh environment record. It now refuses when its
target exists, whatever the spelling of the path, and recomputes only
into a new path given with `--out`. No banked byte changes.

No test in X's suite executed `read_families` before this addendum. The
equivalence of the move rests on the diff, which a stranger reproduces
with `git diff b4facac -- scripts/cord_decoder_isolation.py` at the
commit that adds this protocol's snapshot, and on the synthetic-cell
tests in `tests/test_decoder_isolation_read.py`. Those run
`read_families` on the token basis (the correction above, a different
stop id at the stop position, and a stop id both arms emitted), on the
decoded-text basis, and on X3 with V's count reproducing and not
reproducing. They run the reader end to end on synthetic cells for all
four families, with a VOID family, and they check the anchor, the
manifest, the once-only write and the `--out` refusals. No test opens a
real cell.

## The disclosure this addendum owes

**Nothing here is read blind.** X's artifact records that during X's run
the operator computed the per-family X1 quantities for these families
by hand, for scheduling, and published none of them. The record names
X1 only and is silent on X2 and X3, so all three are treated as possibly
seen. X's readings and combine rule were committed on 2026-09-16 at
06:05Z (authored 05:52Z; `0e5a11c`), and X's cells were first committed
at 18:27Z the same day (`b4facac`). Nothing banks the cells' own file
times, so that order rests on git's commit order alone; X's anchor came
after its run, and `ANCHORS.md` records that it does not prove
freeze-before-run. So *by our own record* what the operator saw could
not have shaped X's readings, and a stranger cannot check that record.
**The decision to read these cells, and every choice in this document,
were made after the hand computation.** Each choice is listed above with
its direction for that reason.

**One later access, dated 2026-10-21.** An agent designing a different
addendum (AC) read the `per_document_seconds` field of the Addendum S/T
comparator cells to estimate decode cost. It read no prediction text, no
token ids and no other field. That field cannot bear on X1, X2 or X3: no
reading takes decode time as an input. X1 compares arms C and P by their
texts and their banked constrained steps; X2 compares arm P with G by
token ids, or with G₀ by decoded text; X3 classifies the texts of arms
C, G and G₀ as valid or invalid JSON. Decode time tracks roughly how long
an output was, and none of the three readings is a function of output
length alone. It is the only access to those cells since X's run that
the operator knows of.

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
   family is **VOID** in every field the reader banks. Its row, its X1
   and X2 reading strings, its prediction checks, its X3, its console
   lines and its obligations carry VOID and the reason, never the raw
   reading. The one exception is X3's check of V's own published
   regression count, which uses V's two cells and none of the family's
   compared pairs. No sentence about the family's decoder isolation is
   licensed.
2. **The decoder's own accounting is wrong** (X's frozen prediction,
   checked per document): a document with 0 banked constrained steps
   whose arms C and P differ, or one with steps whose arms are identical.
   It publishes in X's words: *"the decoder's own accounting is wrong,
   and that is the result — a larger and worse one than the contrast
   this addendum was built to make."* Where arm C is V's reused cell, it
   adds that this reading cannot separate a wrong accounting from V's
   cell not reproducing across processes on those documents, and that
   either one is a defect in a published result.
3. **V's published regression count does not reproduce** from its own
   banked cells: X3's attribution is withheld. Its explained count, its
   residual documents, their attribution and the per-document list they
   are counted from are banked as null, and the disagreement publishes
   first.

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
- **X2:** *"PATH DIVERGES on Phi-3 (V saw 32 of 50 bodies differ at
  bfloat16) and PATH REPRODUCES on SmolLM2. Qwen, Falcon3 and Granite are
  not predicted — Granite because its arms are being compared under a
  prompt V never used."* SmolLM2 is checkable, on the decoded-text basis,
  and is banked with that basis beside it. Phi-3's is not checkable.

## What may be said afterwards

The reader emits the licensed sentences; nothing else may be written
about these families' decoder isolation.

- **X1**, per family, as X's reading string. Where arm C is V's reused
  cell and the arms differ, the sentence carries: *the difference is the
  constraint, or V's cell not reproducing across processes on that
  document, and this reading cannot separate the two.* The combined X1
  sentence is the scoped one above. When a family it counts as REMOVES
  or HELPS has arm C reused from V and differing arms, it names that
  family and carries the same clause. A HURTS family is named with its
  numbers at full size.
- **X2**, per family: *our loop reproduces `generate` token-for-token on
  these fifty documents* only where X2 reads PATH REPRODUCES on the token
  basis (Qwen2.5-0.5B, Granite); *reproduces `generate`'s decoded text on
  these fifty documents (WEAKER TEST)* on the text basis (SmolLM2,
  Falcon3). On the token basis the sentence also carries both counts of
  the EOS correction and what X's code at `b4facac` reads. PATH DIVERGES
  publishes with its count, its basis and the invalid-output counts of
  both arms; on the text basis it adds that the comparator cell carries
  no environment record, so path divergence and drift in the banked cell
  are not separated. (That last clause is this addendum's own addition
  to X's wording, made in the direction against us.)
- **X3** on Qwen2.5-0.5B, as X's reading string.

**The same day, by obligation** (the reader lists which fire): whatever
X1 reads, `tools/README.md`'s `jsongreedy.py` section carries this
addendum's reading beside the paragraph saying X withheld itself. If X1
reads HURTS anywhere, the paragraph beginning *"What it does and does
not do is measured"* names that family at full size. If X3 reads **V'S
ATTRIBUTION IS TOO STRONG**, V's sentence is narrowed to *"E of the
R_V"* by a dated erratum beside `ADDENDUM_V_PROTOCOL.md`, in
`VERDICT.md`'s Addendum V row, and in `CORRECTIONS.md`. No sentence about
Phi-3's decoder isolation may be written at all, and none about a VOID
family's.

## What this cannot show

Everything X said it could not, plus: one family fewer, so the combined
X1 claim is harder to reach and X2 has no bfloat16 family. Fifty
receipts per family, one corpus, schema-only prompts, CPU, four small
checkpoints (named 0.5B, 1.7B, 2B and 1B; no parameter count is banked
here), one box. The reused arms were gated on five documents per family,
a sample, and most compared pairs cannot have their prompt identity
verified at all.

Artifacts read, each pinned by the input manifest:
`experiments/cord_decoder_isolation_cells/` (the cells named in the
manifest; not the gate cells, whose values the reader takes from X's
artifact), `experiments/cord_constrained_families_cells/` and the S/T
comparator cells X names, `experiments/generation_configs_2026-09-16.json`,
`experiments/cord_decoder_isolation_2026-09-16.json` (X's gate),
`experiments/cord_constrained_families_2026-09-08.json` (V's published
regression counts), and `experiments/ladder_e6_cord_split/gold.jsonl`
with `train.jsonl` (the gold the scores use). Artifact written, once,
then anchored: `experiments/cord_decoder_isolation_read_<date>.json`,
where `<date>` is the UTC date the reading starts, filled in at run time
and recorded in the artifact beside its start clock. Reader:
`scripts/decoder_isolation_read.py`. `--check` reports the preconditions
and opens no cell. `--out` recomputes into a new path outside
`experiments/`, only after the banked reading exists, and under the same
anchor, attestation, clean-tree and manifest checks; no spelling of a
path reaches a banked file, and no write replaces a file. Tests:
`tests/test_decoder_isolation_read.py`, on synthetic cells only.

The freeze runs in this order: `scripts/freeze_protocol.py prepare` with
every pinned file as `--code`, this protocol's rows in `ANCHORS.md` and
`VERDICT.md` quoting the stamp, a commit, `freeze_protocol.py stamp`, a
commit of the proof, and after the block, `ots upgrade` and a commit of
the upgraded proof. The reading runs once after that.
