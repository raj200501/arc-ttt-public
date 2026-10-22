# Addendum AC: does the fence tax hold on a third public corpus? (preregistration)

<!-- DRAFT-HEADER-START -->
**DRAFT, NOT FROZEN. Drafted 2026-10-21 by an agent for the owner. No
SROIE output exists from any model. Before it can freeze: the licence gate
below must be passed and recorded; the runner changes and the fetcher
named under "Code" must exist, so their digests can be filled; the file
moves to `docs/research/ADDENDUM_AC_PROTOCOL.md`; then
`scripts/freeze_protocol.py prepare` removes this paragraph and fills the
stamp and the two digests. Nothing else changes at the freeze. The
corpus survey behind every choice here is `ac_corpus_survey.md`, written
the same day; before the freeze it moves in-tree (suggested:
`docs/research/ADDENDUM_AC_CORPUS_SURVEY.md`) with its three
tokenizer-only measurement scripts and their outputs, and the references
below are updated to that path.**
<!-- DRAFT-HEADER-END -->
**Frozen at the stamp below, before any arm ran. Anchored as
`ADDENDUM_AC_PROTOCOL.md.STAMP_PLACEHOLDER.ots`; the anchored bytes are
kept beside it as `snapshots_ADDENDUM_AC_PROTOCOL_STAMP_PLACEHOLDER.md`.
The code is covered by the same anchor through its digests:
`scripts/cord_fence_tax.py` sha256 `RUNNER_SHA_PLACEHOLDER` and
`scripts/fetch_sroie.py` sha256 `FETCHER_SHA_PLACEHOLDER`. Per
`ANCHORS.md`, the anchor proves these bytes existed by the attested block.
That every cell started after it rests on the runner's own clocks
(`started_utc` per cell) and is ours to report.**

## Why this exists

Every fence-rate number this project has comes from two corpora: 30
agent-authored waybills, and 100 CORD receipts. Two simulated reviewers
in one review round wrote the objection down: *"all extraction corpora
synthetic or agent-authored; the one real dataset failed at every scale"*
and *"the only real public dataset (CORD) failed at every scale"*
(`experiments/review_simulation.json`, round index 7, reviewers 3 and 0).
The "failed at every scale" half is about the adaptation gates on CORD
(`VERDICT.md`: "FAILED gates at all three scales"). This addendum cannot
change that and does not try. The other half is fair as stated: one
public corpus is one point. `ROADMAP.md` lists "Other corpora for the
fence tax beyond waybills and CORD."

What S and T established, from their banked artifacts
(`experiments/cord_fence_tax_2026-08-25.json`,
`experiments/cord_fence_tax_families_2026-09-03.json`):

| family | CORD schema-only fenced | CORD k=20 fenced | CORD reading |
|---|---|---|---|
| Qwen2.5-0.5B (S) | 100/100 | 0/80 | (a) |
| Falcon3-1B (T) | 92/100 | 0/80 | (a) |
| SmolLM2-1.7B (T) | 0/100 | 0/80 | (c) |
| Granite-3.1-2B (T) | 0/100 | 0/80 | (c) |
| Phi-3-mini (T) | 0/100 | 0/80 | (c) |

This addendum asks the same question of the same five families on a
receipt corpus somebody else built: same instrument, one forced change to
k, thresholds copied.

## Predictions, written before any arm

Written from the table above and from Addendum R, before any SROIE prompt
was sent to any model. The drafter had already rendered the SROIE prompts
to count their tokens (disclosed below); no output exists. The owner
adopts or replaces these before the freeze, and a replacement is dated.

1. **Schema-only.** Qwen2.5-0.5B and Falcon3-1B fence at least 50 of 100
   SROIE schema-only outputs. SmolLM2-1.7B, Granite-3.1-2B and Phi-3-mini
   each fence fewer than 50.
2. **k-shot at k=4.** Every family fences at most 8 of 80. The basis is
   thin and is stated: Addendum R measured Qwen2.5-0.5B on waybills at
   "k=0 schema 30/30 · k=1 8/30 · k=3 2/30 · k=20 0/30"; no other family
   has been measured at k below 20. Falcon3-1B at k=4 is the prediction
   most likely to fail.
3. **So the predicted readings are** (a) for Qwen2.5-0.5B and Falcon3-1B,
   (c) for the other three, and the predicted combined reading is SAME
   PATTERN ON A THIRD CORPUS (defined below).

If these hold, AC adds no new family. It moves an existing sentence onto
a second public corpus. If any fails, the failure is the result and
publishes at full size.

## The corpus

**SROIE, ICDAR 2019 Task 3**: scanned receipts with OCR transcripts and
a key-information JSON per receipt, built for a competition run by
others. The paper (arXiv 2103.10213) says "a new dataset with 1000 whole
scanned receipt images and annotations is created for the competition",
shows the Task 3 gold as one JSON object with `company`, `date`,
`address`, `total`, and says "some sensitive fields (such as name, address
and contact number etc) of the receipts are blurred". Why SROIE and not
the others surveyed (FUNSD, Kleister-NDA, Kleister-Charity, VRDU, DocILE,
WildReceipt, XFUND) is in `ac_corpus_survey.md`. In one line: it is the
only candidate whose gold is a flat key set and on which a k-shot arm
fits for all five families.

**Source, pinned.** The organisers' portal (rrc.cvc.uab.es) sends its
server certificate without the intermediate that signed it ("GEANT TLS
RSA 1"). With that intermediate added from repo.harica.gr, and verified
against the system roots, its public pages were read on 2026-10-21. The
download sits behind registration. AC uses
the Hugging Face mirror `rth/sroie-2019-v2` at revision
`ede4bd3cd7e687bd5580d0e3f52cda1b4e9bac1c`, file
`data/test-00000-of-00001.parquet`, 182,810,878 bytes, sha256
`695057e834bc72fdd84c2f09d0a7e93ecdd439cbda16318bc17eb1d0a5e9a893` (as
the Hub reports it; the fetcher recomputes it and refuses on mismatch).
The mirror is a derivative. Its card says duplicates "were removed by
https://github.com/Losyash/SROIE-datasetv2 as far as I can tell", and its
split counts (train 626, test 347) differ from the paper's ("trainval"
600, "test" 400). They match the folder names in the Google Drive folder
that the third-party zzzDavid repository links as the original dataset
("0325updated.task1train(626p)", "task3-test 347p)"). Who owns that folder
is not verified. Its Task 3 test folder holds 347 images and no gold, so
where the mirror's test gold comes from is not verified. Its fidelity to
the organisers' files is not verified and is listed below as a limit.

**Documents.** The first 100 rows of the mirror's test split, row
indices 0 to 99, in file order. No row is dropped for length or content.

**Rendering, fixed now.** Document text: the row's `objects.text`
transcripts in the order the file gives them, joined with `"\n"`. No
coordinates. This mirrors the CORD adapter (`arcttt.text_task.from_cord_gt`:
OCR text from annotations, lines joined with `"\n"`, no layout). Gold:
`objects.entities`, keys with an empty value dropped, serialised by
`arcttt.scoring.json_canonical` (sorted keys, compact separators, unicode
kept), the same serialiser as the CORD demonstrations.

**Split, by the E6 rule.** Indices 0 to 99, `random.Random(1).shuffle`,
first 20 are the demonstration pool, next 80 are the k-shot evaluation set
(`experiments/ladder_e6_cord_split/manifest.json`, "rule"). The
demonstrations are the first k of the pool in shuffled order. The 16 pool
receipts not used as demonstrations stay out of the k-shot evaluation, so
its n is 80 as in S and T. Schema-only covers all 100. The split
manifest (`experiments/ac_sroie_split/manifest.json`) banks the row
indices, the rule, the pool order, and the sha256 of every rendered text
and canonical gold. It does **not** bank the texts or the gold: SROIE's
redistribution terms are not established, so the fetched rows live under
`data/sroie/` (gitignored) and a stranger re-fetches them from the pinned
revision.

## The licence gate, before the freeze

No first-party licence text for SROIE could be read on 2026-10-21. The
public RRC pages carry none. The downloads page says "You will need to
register to get access to the download section." The registration terms
say nothing about how the data may be used. The paper states no licence;
it says "The new datasets used in this competition will be made available
after the event." Two third parties say CC BY 4.0: OpenMMLab's mmocr
dataset index ("License: Type: CC BY 4.0") and the jsdnrs/ICDAR2019-SROIE
card ("licensed under the CC-BY-4.0 as noted on" the RRC downloads page).
The chosen mirror's README says "license: cc-by-2.0" (unparsed by the
Hub). Of 101 Hugging Face datasets matching "sroie", 77 carry no licence
tag and the rest disagree (other, mit, cc-by-4.0, unknown, openrail). None
of this is the organisers' text. The evidence and URLs are in
`ac_licence.md`. This repository's own spec already called SROIE's
licence "the weak point" (`ENTERPRISE_EVAL_SPEC.md` 1.3).

So: **AC does not freeze until the owner has read the organisers' terms
on the RRC downloads page (registration required) and recorded them,
quoted and dated, in this document's freeze commit.** Registration is
"only meant as an expression of interest" (RRC downloads page). If the
terms do not permit research use, exclude use by a company, or permit
it but not publishing model outputs that
quote receipt content (the cells bank raw outputs, which contain company
names and addresses from the receipts), AC is withdrawn. It is **not**
moved to another corpus under the same name: the survey shows no other
candidate holds a k-shot arm for all five families, so a move would be a
different addendum with its own predictions.

**Recorded terms:** not yet recorded. (The owner replaces this line with
the quoted terms, the URL and the date read. The runner refuses while it
reads as it does now.)

## What changes from S and T, each named with its direction

Everything S and T froze is used as frozen except the following.

1. **The corpus.** SROIE instead of CORD. The point of the addendum.

2. **k = 4, not 20.** Forced by length, fixed by a rule before data. At
   k=20 every one of the 80 SROIE evaluation prompts is over the runner's
   8,192-token guard for every family (longest: 9,656 tokens for
   Qwen2.5-0.5B, 11,192 for Phi-3-mini; tokenizer-only, `ac_corpus_survey.md`).
   **The rule:** k is the largest value up to 20 such that, for every
   family and every evaluation prompt, prompt tokens + 512
   (`max_new_tokens`) is at most min(8,192, the checkpoint's
   `max_position_embeddings`). The second term is new. It exists because
   Phi-3-mini-4k has 4,096 positions, and a prompt the model was not
   trained to read is a confound this addendum should not carry. By the
   pre-freeze measurement the rule gives **k = 4**, set by Phi-3-mini
   (largest k for the other four: 13 to 15). **Direction:** fewer
   demonstrations is a weaker suppressor in R's dose curve, so k=4 makes
   reading (a) harder to reach, not easier. It also means a (b) on SROIE
   cannot be separated into "the corpus" and "the smaller k"; that
   sentence is fixed below. The costlier alternative, a k=13 arm on the
   four families whose context allows it, is not taken: it adds four
   cells and a second k-shot reading with no frozen place to land.
   **Found while measuring, about T:** by the same rendering, T's longest
   Phi-3 k=20 prompt on CORD is 3,628 tokens, so an output longer than
   468 tokens would pass 4,096 positions. Whether any banked T output did
   is not checked here (written while the T cells were held back for
   Addendum AA, which was read on 2026-10-22; it can be checked before
   AC freezes). It is named so it is not discovered later.

3. **The schema instruction's vocabulary.** Derived from SROIE gold by
   the S rule (keys and their types read from the gold, asserted at run
   time against a fresh derivation; drift refuses to run). All four gold
   values are strings in all 100 rows. Verbatim, all families:

   ```
   Extract the receipt fields from the document and return them as a
   single JSON object. Use only these keys:
   - address: a string
   - company: a string
   - date: a string
   - total: a string
   Omit any key the document does not support.
   ```

   The first two sentences and the last are the S instruction's words.
   Only the key lines differ, because the gold differs.

4. **Reproducibility records, none of which changes what is computed.**
   Each checkpoint is loaded at a pinned revision (table below). Every
   cell banks an environment record (Addendum X's `_environment()` shape),
   its own start and finish clocks (S/T banked a constant run date; T's
   row discloses it), and the sha256 of every rendered prompt (Addendum X's
   convention). Threads stay at 4, as in the S/T runner; Addendum Y showed
   Phi-3's text can change with reduction order, so the thread count is
   recorded, not assumed.

5. **Granite's clock.** Granite's chat template writes the run date into
   its system message (T's erratum of 2026-09-16). AC renders it from
   Addendum X's frozen string (`RENDER_DATE = "September 16, 2026"` in
   `scripts/cord_decoder_isolation.py`), checked at run time to reproduce
   the template's own output except for the date; the run refuses if it
   does not. **Direction:** neutral for the fence rate; it makes AC's
   Granite cells reproducible, which T's are not.

**Unchanged, stated so nobody has to check:** greedy through
`model.generate(do_sample=False)`, which still applies each checkpoint's
own generation defaults (Addendum X measured: "`Qwen/Qwen2.5-0.5B-Instruct`
carries `repetition_penalty` 1.1 [...]; SmolLM2, Granite, Phi-3 and
Falcon3 carry none"); `max_new_tokens=512`; the 8,192-token guard; each
family's own chat template via `apply_chat_template`; schema-only as one
user turn; k-shot through `run_challenge.build_task` +
`text_task_to_messages`; the same dtypes as T; per-document checkpoints;
raw text banked; classifier the SHIPPED `tools/fencecheck.py`
`strip_fence`, leading-fence scope, with the undercount property R's
erratum documents; the carried quantity is the fence RATE. Scores
(`field_micro_f1` against SROIE gold) are banked as context only, and no
score is compared across corpora.

## Families

The five families of Addendum V's table, one checkpoint each. Qwen2.5-1.5B
from S is not run (cost); it is named so its absence is visible.

| family | checkpoint | revision (Hub API, 2026-10-21) | dtype | CORD reading |
|---|---|---|---|---|
| Qwen2.5 | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | float32 | (a) |
| HuggingFace | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `31b70e2e869a7173562077fd711b654946d38674` | float32 | (c) |
| IBM | `ibm-granite/granite-3.1-2b-instruct` | `bbc2aed595bd38bd770263dc3ab831db9794441d` | float32 | (c) |
| Microsoft | `microsoft/Phi-3-mini-4k-instruct` | `f39ac1d28e925b323eae81227eaba4464caced4e` | bfloat16 | (c) |
| TII | `tiiuae/Falcon3-1B-Instruct` | `28ba2251970a01dd1edc7ba7dad2eb71216ccfdf` | float32 | (a) |

The Qwen revision is the one Addendum Z pins. If any revision has moved
on the Hub by the freeze, this table is updated in the freeze commit,
before any cell, and says so. S and T did not pin revisions, so a
difference between a family's CORD and SROIE behaviour carries a
possible checkpoint difference as well as the corpus; this cannot be
separated after the fact and is listed below.

**Cells:** five families × {schema-only n=100, k-shot k=4 n=80} = ten
cells, `experiments/sroie_fence_tax_cells/<family>_<regime>.json`.

## Frozen readings, applied by arithmetic in the reader

Per family, with `f_schema` and `f_kshot` the SROIE fence rates. Copied
from S and T, thresholds and inclusive boundaries unchanged
(`family_reading` in `scripts/cord_fence_tax.py`):

- **(a) replicates:** `f_schema ≥ 0.50` and `f_kshot ≤ 0.10`.
- **(b) partial:** `f_schema ≥ 0.50` and `f_kshot > 0.10`.
- **(c) does not replicate:** `f_schema < 0.50`.

The CORD readings are read from the S and T artifacts'
`reading_per_model` fields, never retyped (S's `0.5b` for Qwen2.5-0.5B).
Three combined statements, each computed and each published:

**R1, the replication question, on the two families where CORD showed
the tax (Qwen2.5-0.5B, Falcon3-1B).** Each family's SROIE reading is
printed beside its CORD reading, then:
- **REPLICATES ON SROIE** if both read (a).
- **THE FENCE TAX DID NOT REPLICATE ON A THIRD CORPUS** if both read (c).
- **MIXED ON SROIE** otherwise, with each family named at full size:
  (a) as REPLICATES, (b) as PARTIAL, (c) as DOES NOT REPLICATE.

**R2, the stability of T's exceptions (SmolLM2-1.7B, Granite-3.1-2B,
Phi-3-mini):**
- **STABLE** if all three read (c) on SROIE.
- **CORPUS-DEPENDENT: <family>** for each that reads (a) or (b), named at
  full size with both corpora's rates.

**R3, the pooling ban.** No count adds SROIE and CORD together. Counts
are per corpus, always written with the corpus named.

**Headline.** **SAME PATTERN ON A THIRD CORPUS** may be printed only if
every one of the five families reads the same on SROIE as on CORD (R1
REPLICATES and R2 STABLE). Anything else prints the R1 and R2 statements
and no headline.

Readings are applied by arithmetic in the runner, never re-derived from
the shape of the data. The reader withholds every number until all ten
cells exist.

## What may be said afterwards, in the non-flattering direction

The reader emits the licensed sentences. Nothing else about AC is
written.

- **SAME PATTERN ON A THIRD CORPUS:** *"On SROIE, a public receipt corpus
  nobody here built, schema-only prompts were fenced and k=4 prompts were
  not on Qwen2.5-0.5B and Falcon3-1B, and SmolLM2, Granite and Phi-3
  fenced fewer than half, as on CORD at k=20. The fence tax has been
  measured on two public receipt corpora, on the same 2 of the 5 families
  tested on each."* Never *across corpora*, never *on public data*
  without *receipt*, never *across model families*.
- **REPLICATES ON SROIE with R2 not STABLE:** the R1 sentence above for
  the two families, and the R2 sentence below. No headline.
- **PARTIAL for a family on R1:** *"Schema-only prompts were fenced on
  SROIE on <family> (<rate>); at k=4, demonstrations did not bring it
  under the 0.10 bar (<rate>). Whether that is the corpus or the smaller
  k cannot be separated by this addendum."* The mechanism sentence that
  demonstrations suppress the fence keeps its scope: measured at k=20 on
  CORD, and on waybills by R's dose curve.
- **DOES NOT REPLICATE for a family on R1:** that family is named
  with both corpora's rates, and the tax on it is written as *measured on
  CORD* (Qwen2.5-0.5B: *and on the waybills*), never as a property of
  the family.
- **THE FENCE TAX DID NOT REPLICATE ON A THIRD CORPUS** (both (c)): this
  is S's re-scoping commitment, applied one level up. In the same commit
  that publishes the reading, every sentence in `README.md`, `VERDICT.md`,
  `EVIDENCE.md` and the outbound documents under `docs/strategy/` that
  says the tax holds *on public data* or *on a corpus somebody else
  built* gains *on CORD*, and the AC sentence goes beside it. The parser
  findings of Addendum U stand on their own terms: where a fence occurs, a
  fail-open parse loses the output. Only the prevalence claim narrows.
- **CORPUS-DEPENDENT: <family>:** *"<family> fenced <n>/100 schema-only
  outputs on SROIE and 0/100 on CORD. Whether a family wraps its JSON
  depends on the corpus as well as the family."* In the same commit, T's
  sentence that three of its four families "emit no fence at all under a
  bare field list" gains *on CORD*. This outcome raises the SROIE count
  and is still written against us: it says T's family-specific reading
  was incomplete, and R3 forbids adding it to CORD's count to say
  *3 of 5* or more.
- A (c) or a CORPUS-DEPENDENT reading is published, not explained. What
  about the corpus or the family produces it is not tested here.

## What runs first, and what stops the run

The runner refuses to start a cell, and says why, if any of these fails:

1. **Licence recorded.** The "Recorded terms" line in the licence gate
   section no longer reads "not yet recorded".
2. **Corpus identity.** The parquet's sha256 matches the value above, and
   the 100 rendered texts and golds match the split manifest's digests.
3. **Vocabulary.** The gold's keys and value types re-derive to exactly
   the four string keys in the instruction.
4. **Length.** k=4 is recomputed with each family's own
   `apply_chat_template` on every evaluation prompt, under the bound
   above. If the recomputed k differs from 4 for any family, the run
   refuses and a dated erratum is written before any cell runs. (The
   pre-freeze value was computed outside transformers with jinja2 and
   `tokenizers`; small differences are possible.)
5. **Granite's frozen string** reproduces the template's own system
   message except for the date.
6. **Revisions** match the table.

No determinism gate is set. The carried quantity is a count of fenced
outputs. T's errata kept its fence rates through the Granite clock and
Phi-3 reproducibility findings on the argument that those differences do
not turn a fenced output into an unfenced one. AC reuses that argument;
it does not test it. That is a choice, and it is named.

## Disclosures, so nobody has to find them

- **What the drafter saw before writing the predictions:** the S and T
  aggregate artifacts (rates and readings), the R row, the S/T cells'
  `per_document_seconds` field for the cost estimate below (the
  `predictions` field was not accessed), and the first 100 SROIE test
  rows' texts and gold, from which only prompt token counts and gold
  lengths were computed. No model has been run on any SROIE text. No
  quantity was computed from any S, T, V or X prediction text.
- **The predictions are foreseeable from T.** That is their basis, and
  it is stated rather than presented as foresight.
- **k was chosen after the corpus was seen**, by the rule in item 2,
  from token counts alone. The rule and its value are frozen here before
  any output exists.

## Cost, from the S/T banked timings

Summed `per_document_seconds` of the ten CORD cells for these five
checkpoints (decode time only; model loading not banked):

| family | schema-only (n=100) | k-shot k=20 (n=80) |
|---|---|---|
| Qwen2.5-0.5B (`cord_fence_tax_cells/0.5b_*`) | 2,360.1 s | 1,517.2 s |
| SmolLM2-1.7B | 5,222.7 s | 4,995.3 s |
| Granite-3.1-2B | 7,288.4 s | 5,751.9 s |
| Phi-3-mini (schema cell resumed from 15/100) | 13,843.9 s | 5,829.7 s |
| Falcon3-1B | 5,722.5 s | 2,603.0 s |
| **total** | **9.57 h** | **5.75 h** |

Sum: 55,134.7 s, **15.32 hours** on four cores. What differs on SROIE
(tokenizer-only, `ac_corpus_survey.md`): schema-only prompts are longer
(median 428 to 492.5 tokens against 223.5 to 276 on CORD); k=4 prompts
are shorter than CORD's k=20 (median 2,335.5 to 2,676.5 against 3,241.5
to 3,571.5); the target JSON is shorter (median canonical gold 159
characters against 216.5 over the 20 demonstration receipts). CPU decode
time follows the number of tokens the model writes, which is not known
until it runs. So the estimate is the S/T figure, **about 15 hours of
decode, not discounted**, plus model loads, one 182,810,878-byte fetch,
and any restarts. Phi-3's schema-only cell alone was 3.85 hours on CORD.

## When it runs

**Only after Addendum Z has finished on the shared box.** Z is running as
of 2026-10-21 (restart 3 of seed 1 since 20:25Z, no arm banked, per its
VERDICT row), priced at 40 to 60 wall-hours on four cores, and measured
at 11.9 GB peak RSS (`ADDENDUM_Z_PROTOCOL.md`). The box has 4 cores and
MemTotal 16,479,424 kB (read 2026-10-21). A second model process would
compete with Z for both, and would change Z's timings. AC
starts after Z's last arm banks, or after the owner stops Z in writing
in `experiments/addendum_z_operator_log.jsonl`. One AC cell at a time.
The driver is idempotent per cell, as S's and T's are.

## What this cannot show

- **One more corpus, and it is receipts again.** Two receipt corpora say
  nothing about forms, contracts or invoices. The form and contract
  candidates surveyed either state no licence (Kleister, VRDU) or leave
  no k-shot arm for Phi-3 under the bound above (FUNSD).
- **k=4, not 20.** A k-shot difference between CORD and SROIE cannot be
  split into corpus and dose.
- **Mirror fidelity and licence.** The documents come from a
  redistributor's copy whose relation to the organisers' release is not
  verified, under terms that are a gate, not a fact, as of this draft.
- **Checkpoint drift.** S and T loaded unpinned checkpoints; AC pins
  them. A family that reads differently could, in principle, have
  changed on the Hub in between.
- **Five families at 0.5B to 3.8B, CPU, greedy.** Nothing about hosted
  APIs or larger checkpoints.
- **Leading-fence scope.** The shipped classifier undercounts a fence
  after prose (R's erratum). The same classifier runs on both corpora, so
  the undercount applies to both; its size on either is not measured.
- **Not the cheapest signal.** One of the same reviewers named it: file
  the two upstream issues (`cheapest_signal`, same file). AC is not that.

## Code and artifacts

To be written before the freeze; their digests are filled at `prepare`:

- `scripts/cord_fence_tax.py`, `--addendum AC`: the corpus loader, the
  instruction and its vocabulary assertion, k and its recomputed bound,
  pinned revisions, Granite's frozen string, the environment record, the
  per-prompt digests, the cells directory, and a pure function for R1, R2
  and the headline beside `family_reading`, reading CORD's readings from
  the S and T artifacts. The decode loop is the S/T loop, not a copy.
- `scripts/fetch_sroie.py`: the pinned parquet, its sha256 check, the
  `objects` column only (never the images), rows 0 to 99, written to
  `data/sroie/` (gitignored), and the split manifest.
- `tests/test_cord_fence_tax_readings.py`: the AC combination at its
  boundaries on synthetic readings: SAME PATTERN; R1 REPLICATES, MIXED
  (one (b); one (c)) and DID NOT REPLICATE; R2 STABLE and
  CORPUS-DEPENDENT; and that no emitted sentence says *across corpora* or
  *across model families*.

Artifacts: `experiments/ac_sroie_split/manifest.json` (before any cell),
`experiments/sroie_fence_tax_cells/` (ten cells),
`experiments/sroie_fence_tax_ac.json` (the reading). Driver:
`scripts/run_addendum_ac.sh`; the reader withholds until all ten cells
exist.
