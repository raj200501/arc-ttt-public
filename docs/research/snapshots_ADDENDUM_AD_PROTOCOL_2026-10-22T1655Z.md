# Addendum AD: does the fence move a published number? LiveBench's stored 2024 answers, re-scored with one fence strip (preregistration)

**Frozen at the stamp below, before the runner fetched or scored any
stored answer and before any published `tablejoin` value was read.
Anchored as `ADDENDUM_AD_PROTOCOL.md.2026-10-22T1655Z.ots`; the anchored
bytes are kept beside it as
`snapshots_ADDENDUM_AD_PROTOCOL_2026-10-22T1655Z.md`, so errata appended
to this file later never break the check. Every code and data file the
reading runs is covered by the same anchor through its digest, listed
below. Each stage of the runner refuses to start unless the proof is
committed and parses as an OpenTimestamps proof of the committed
snapshot, this file begins with the snapshot, every digest below
matches, and the parsed proof carries a Bitcoin attestation with a block
height. The runner walks the proof's structure: eight attestation bytes
found anywhere else in the file do not count. It does not recompute the
merkle path or check the block header. `ots verify` does that against a
node: the operator runs it and commits its output before `fetch` (How
it runs), and the runner does not check that it was done. So no
download, score or reading can precede the block that dates this
protocol, as far as that verification goes. Each stage banks its own
start and finish clocks, the block heights and the git HEAD.**

Pinned, one file per line:

- `scripts/published_number_audit.py` sha256 `0631ce7b6bfa95863d1c298fe1d3e0f10201c430f018ee6d360f129603231af3`
- `tools/fencecheck.py` sha256 `3bc8efff435273a471797e89f9bdb4f67cec5f59ee2f8f99af59a7e54fc05fe7`
- `experiments/ad_manifest_2026-10-21.json` sha256 `ed5a027e24f8868c952f373d16cdbb33592738d92b279dea77ecaac317ab03f0`
- `experiments/ad_synthetic_scorer_check_2026-10-21.json` sha256 `80d21f435f745c81dbd7963c17e6b6437583869721396b51426a486eba3ad79b`
- `scripts/ad_pinned/tablejoin_utils_905b71f.py` sha256 `ee638884f08ea802f59d14583f77c7e38b0c045e8d6bf309af0cca7d7bf12f3a`
- `scripts/ad_pinned/tablejoin_utils_f774d19.py` sha256 `8d7fa703061d4536ee9f07c1a3d103690a6a20c01d8ab2c284fcb43ce45724b9`
- `scripts/ad_pinned/tablejoin_utils_19c7757.py` sha256 `b04aee527f1e38d8a2dad456d2bcf2b2af622d64cd01221a3f31a480bb54c0e8`
- `scripts/ad_pinned/tablejoin_utils_c5c293c.py` sha256 `47a6fe5b2277b5760a186e9c556023e90a291bf915715dc5b87d0dba9e8bc0eb`
- `scripts/ad_pinned/tablejoin_utils_e647d72.py` sha256 `1d8e56395b6ad033391503fe5914eb242f080eed5f07695c655e28d779dd213c`
- `scripts/ad_pinned/util_19c7757.py` sha256 `e8f97597589c640a89e1358093f9aa5d90f0673fedbc3998fcc267ce0854c0f1`
- `scripts/ad_pinned/util_f3501c7.py` sha256 `f4f8769b5b0d1bf696151212962823009753b822e3310c2087818f7a36b1ac0a`
- `scripts/ad_pinned/gen_ground_truth_judgment_905b71f.py` sha256 `b35746a0003db176914a5e3fa33496276c66ebc666156a5a57e2284c93dfa220`
- `scripts/ad_pinned/gen_ground_truth_judgment_f774d19.py` sha256 `81226f3dfc4c06b3c4a168000dd55b1e868a992565ed365d3a6c37bd81d1e3ba`
- `scripts/ad_pinned/gen_ground_truth_judgment_19c7757.py` sha256 `c414031603e32ae3868299149b90241bc5841501adf38b38414ffaecbd08b6db`
- `scripts/ad_pinned/gen_ground_truth_judgment_fa6a3c6.py` sha256 `ed7973d19360ccf3a4a58d3112918b4f561009cdbdd2c4df55a2c85525a3d137`
- `scripts/ad_pinned/gen_ground_truth_judgment_f3501c7.py` sha256 `5d0b7a28d3b8f042d2eae7ee192e92d8c56c949c635a181441fb8f01ab22ccac`
- `scripts/ad_pinned/gen_ground_truth_judgment_e647d72.py` sha256 `a6882e2f3bbc7679601aa776b4dec8318d233935d4b3b7e4d8e3b408f2a4b746`

The manifest pins every remote file by URL, size and digest, so the
anchor covers the data too. It does not pin `tools/fencecheck.py` a
second time: the line above is that file's one pin. The vendored files
are LiveBench's code, copied byte for byte (`scripts/ad_pinned/README.md`,
Apache-2.0 licence beside them).

## Why this exists

`tools/README.md` says it plainly: *"Whether that bias shows up in
published work is a separate question that this tool does not answer."*
Every fence number this project has is on its own outputs or on code.
The census found the defect in code (7 confirmed instances across 34
published packages; 1 confirmed instance in 7 actively maintained ones;
`README.md`). It never asked whether the defect ever reached a number
someone published.

The closest precedent is Hugging Face's Math-Verify re-evaluation
(2025-02-14): an answer extractor scored correct answers as wrong, and
fixing it re-scored "all 3,751 models ever submitted to the Open LLM
Leaderboard" (as quoted in `docs/strategy/FDM1_EVIDENCE_MEMO_2026-10-21.md`,
section 3E). That was the leaderboard's owner fixing its own extractor.
AD is an outsider doing the same arithmetic on stored text, with the
protocol frozen first.

## Why this target, and only this one

A scoping survey on 2026-10-21 (`docs/research/AD_SURVEY_2026-10-21.md`,
copied unchanged from the scoping notes; its script paths are the
scoping session's scratch directory) checked eleven public result sets
for three things together: stored raw per-sample outputs that a stranger
can download, a scorer that fails closed on a leading fence, and
published per-model numbers computed by that scorer. One set has all
three:

- **LiveBench `tablejoin`, 2024.** The answers are public and ungated
  (`livebench/model_answer`, 93,715 answers, 4,350 of them `tablejoin`).
  The scorer, `livebench/process_results/data_analysis/tablejoin/utils.py`,
  extracted only a ```` ```python ```` fence until `f774d19`
  (2024-10-22T01:43:34Z; 2024-10-21 in the committer's time zone) and
  did not extract a ```` ```json ```` fence until commit `e647d72` on
  2025-04-18. Executed on synthetic input during scoping: every version
  from `905b71f` to `e2a8e99` scores a correct ```` ```json ````-fenced
  answer 0.0, `905b71f` also scores a correct bare-fenced answer 0.0,
  and the shipped `fencecheck.strip_fence` restores 1.0 on each. The
  published per-model numbers are versioned in
  `LiveBench/livebench.github.io` and served at
  `https://livebench.ai/v1/table_<date>.csv`. The four 2024 tables have
  not changed since 2025-03-31.

The others were set aside, with reasons in the survey: the Open LLM
Leaderboard and HELM IFEval checks strip a leading fence; HELM's judge
and annotator parses, ZeroEval and the ARC Prize harness extract by
braces; BFCL and TaskBench publish no per-sample outputs; WildBench and
AlpacaEval parse a judge's output, and WildBench calls its judge in JSON
mode; tau-bench's parse fails inside the agent loop, which a re-score
cannot undo. OpenCompass's T-Eval predictions fail closed on a narrower
fence class but are gated and were not linked to a published number;
they are named as a possible successor, not part of AD.

**The owner already fixed this.** AD does not find a live bug. It asks
whether a bug the owner fixed on 2025-04-18 had already moved numbers
that are still published under the 2024 releases.

## What was seen before this was written

Disclosed so it is not discovered later. Each item was read, fetched or
left on disk during scoping, while building the runner, or during the
pre-freeze audit, before the freeze. Nothing in this list is a score of
a stored answer or a value of a published table that anyone recorded.
Where held-out bytes sit on disk, that is said, with where.

- **One file of stored answers, fence tags only.** The 50 `tablejoin`
  answers of `gpt-4o-2024-05-13` (datasets-server `/filter` page, 28,125
  bytes, sha256 `2dc81677aa0f76acb1f20a1780866ad8bc7ca657a74131a224358c0498968015`).
  Counted with a script that does not parse, score or read ground truth:
  **50 of 50 open with a fence, and all 50 fences are tagged `python`**.
  That script lower-cased the tag, so a ```` ```Python ```` fence would
  also have counted as `python`; LiveBench's regex is case-sensitive.
  The counts are banked as `experiments/ad_feasibility_counts_2026-10-21.json`.
  That model is one of the frame and is not excluded. The page is still
  on disk in the scoping scratch directory, next to all 150 question
  rows with their ground truth (below), so that model's O and F could
  have been computed from local files before the freeze. No such score
  is recorded anywhere; that nobody computed one cannot be verified.
- **The published tables: names, metadata, and bytes on disk.** The
  header line and the model-name column of the five tables below, their
  commit metadata (dates, titles, counts), their sizes and git blob ids
  from the site's tree, and their sha256 (computed by the scoping
  session). No value in any table was printed, and no diff of any table
  was opened. During scoping, two `GET`s at the site's root path
  returned 404 (9,379 bytes each, body discarded), and two `GET`s under
  `/v1/` returned the `2024_06_24` and `2025_04_02` tables in full
  (8,879 and 8,441 bytes); both bodies were discarded unread (survey,
  request log). While building the runner, and again in the pre-freeze
  audit, `HEAD` requests on the five live tables returned the same
  sizes as the pinned files (8,879; 9,261; 7,694; 8,508; 8,441 bytes)
  and no content. **The table bytes are on disk.** The scoping session's
  blobless clone of the site repository holds the blobs of all five
  pinned tables, and the build session's clone holds the same five plus
  the `2025_04_02` table before the 2025-04-18 edit (blob `b78dce2`,
  8,299 bytes), which git fetched into that clone at 2026-10-21T21:22:52Z
  for a command that needed its content; which command is not recorded.
  So every published `tablejoin` value AD reads was one offline command
  away before the freeze. No value from them is quoted anywhere in this
  repository; that none was displayed cannot be verified. They were not
  deleted.
- **A judgment file on disk.** A full `livebench/model_judgment` parquet
  (revision `9704e5da`, 737,444 bytes) was downloaded into the same
  scratch directory at 2026-10-21 20:50 UTC by a different task. That
  task's notes say it holds only the language, coding and
  instruction_following categories, so no `tablejoin` judgment; that is
  not checked here, because checking it would read it. The scoping
  session separately downloaded the judgment dataset's column
  statistics with its `score` column; the printing script skipped that
  column.
- **The questions and their ground truth, all of them.** All 150 rows of
  `livebench/data_analysis` were downloaded during scoping (datasets-server
  `/rows`, offsets 0 and 100: 204,417 and 140,217 bytes) and are on
  disk. The scoping notes counted them by task and release date and
  printed one `tablejoin` question and the length of its ground truth
  string (74 characters). The pinned questions parquet is also on disk
  twice, one copy from another task and one from the pre-freeze audit.
  Ground truth is not held out; it is listed because it is half of what
  a score needs.
- Column statistics of the answer and question datasets (counts by
  task, model and category; timestamps). The Hugging Face tree listings
  of the two parquets (sizes and LFS sha256) were read while building
  the manifest.
- The scorer code at every version, its output on five synthetic
  answers, and the site's averaging code and category maps.
- **The pre-freeze audit (2026-10-22).** It read the Hugging Face commit
  listing of `livebench/data_analysis`, downloaded the questions parquet
  at four revisions, and compared their `tablejoin` rows by equality
  only (below, "The ground truth"); it re-ran the scorer and site code
  through `git show`, and made the `HEAD` requests above. It read no
  answer, judgment or table value. The session that then fixed what the
  audit found made no remote request; it re-ran that equality check,
  listed the object ids and sizes in the two site clones, and hashed
  only the files in the listing below that hold no held-out bytes.
- What the pinned `fencecheck scan` reports on this scorer (below, "What
  the scanner says about this scorer"). That is a fact about the tool;
  AD does not depend on it.

Every file above that holds held-out bytes or ground truth, with its
place in the scratch directory, size, git object ids and recorded
digests (none computed from a held-out file), is listed in
`experiments/ad_preexposure_2026-10-22.json` sha256 `ed9c4f931add63dc68e00ef4f8764327a2531b94983bf64eabc1b2185ec3aa2e`.

## Corrections made before the freeze

The scoping draft of this protocol was turned into code on 2026-10-21,
and the code and this text were audited on 2026-10-22. Each fix below
was made before any data. Where a fix moves a reading or a prediction
in one direction, the direction is stated; where it would make a
prediction easier to hold, the prediction is also checked under the rule
it was written under (below, "Predictions").

1. **The `<think>` removal starts at `fa6a3c6` (2025-01-22), not at
   `f3501c7`.** `f3501c7` added a `<think>` removal for the
   instruction-following path only. The line that removes `<think>` from
   every answer before scoring, in `play_a_match_gt`, first appears at
   `fa6a3c6`. Read from the vendored files.
2. **States of `main`, not commits.** The draft counted 21 commits that
   touch the three files. Three of them reached `main` only through two
   merge commits. The runner uses the first-parent states of `main`: 20,
   two of them merges. De-duplicated by the digest tuple below, they give
   **7 versions**, not 6: `905b71f`, `f774d19`, `19c7757`, `c5c293c`,
   `fa6a3c6`, `f3501c7`, `e647d72`.
3. **The in-force rule no longer needs `git blame`.** The draft bounded
   each cell by the table's first commit and the blame time of its row.
   The first commit is not a lower bound: a model's `tablejoin` cell can
   be carried into a later table from an earlier scoring, since the
   questions did not change. The rule below uses the newest stored answer
   as the lower bound and the table's last commit as the upper bound.
   The upper bound is looser than the draft's. The lower bound is not:
   it is a different quantity, and for a model whose newest answer is
   after a table's last commit it leaves no version in force, where the
   draft's rule kept one (for `2024_06_24` and `2024_07_26`, stored
   answers run to 2024-10-17 and the tables' last commit is 2024-09-07).
   Such a cell is ANSWERS POSTDATE TABLE. That cuts both ways: it counts
   against the G3 prediction (not reproduced), and it drops out of AD2
   and AD3, which can only make NULL and NO RANK CHANGED easier. So the
   reading banks, per table, how many cells are POSTDATE under this rule
   and had a version in force under the draft's lower bound.
4. **The reproduction tolerance follows the printed precision.** The
   draft fixed it at 0.0005 points. A CSV writer drops trailing zeros,
   and a 50-answer mean of scores rounded to two decimals is a multiple
   of 0.02, so a cell may print with two decimals. The tolerance is half
   a unit of the d-th decimal, where d is the most decimals any cell of
   the column carries, and d is at most 3. So it is never below 0.0005
   points: 0.0005 for a three-decimal column (the draft's value), 0.005
   for two, 0.05 for one, 0.5 for whole numbers. A coarser column makes
   REPRODUCED easier; each table's d is banked.
5. **Only an upward move is the fence failure's.** The draft counted a
   cell as moved when F − O was 2.0 or more in either direction. A
   downward move means the strip scored the answer lower than
   LiveBench's own code did: that is the strip's error, not the fence
   failure. It happens: `strip_fence` keeps everything up to the last
   ```` ``` ````, so an answer that opens a second fence (a correction,
   or code then the dict) can parse under LiveBench's regex and fail
   after the strip (executed on synthetic input: O = 1.0, F = 0.0 under
   `19c7757`). Counted as a move, it would have licensed "the fence
   failure moved a published number" on fencecheck's own error. Now
   only F − O ≥ 2.0 moves a cell; a cell with F − O ≤ −2.0 is counted
   and published as the strip's error. This makes AD2 = NULL easier to
   hold, so that prediction is also checked under the drafted rule.
6. **Which reproducing version a claim uses, and which a prediction
   uses.** The draft used the smallest F − O, calling it the reading
   against a positive. With moves in both directions counting, that was
   wrong: the most negative F − O made a downward move more likely. With
   only upward moves counting (5), the smallest F − O is the reading
   against a positive: a cell moves up only if it moves up under every
   version that reproduces it. The predictions are now checked under
   the largest F − O, the reading that makes them hardest to hold, and
   the licensed NULL sentence says how many cells would move under it.
7. **AD1 counts every in-frame answer.** The draft's code skipped the
   answers of a model none of whose cells had a version in force, and
   took a model's reading version from the earliest table that had one,
   not from the earliest table that shows the model. Both now follow the
   text (below, AD1). This follows the prediction as written; it can
   move the AD1 ratios either way.
8. **The licensed NULL sentence claims only what was tested.** The
   draft's sentence ended "on these answers it did not reach the
   leaderboard" whatever moved by less than a question, whatever AD3
   said, and however few cells reproduced. The sentence below states
   each of those.
9. **Landing is committer time, on both sides.** A version is in force
   from its commit's committer time; the table's upper bound was the
   author time of its last commit and is now its committer time. The
   manifest's two times are equal for every version and every table, so
   nothing moves.
10. **The scanner changed under this protocol.** `tools/fencecheck.py`
    was reworked on `main` after the draft (`7f9068a` and `452527b`,
    2026-10-22). What `scan` reports on the vendored scorers changed with
    it (below). The draft manifest quoted a digest of the old file; it
    now quotes none.

The predictions were sharpened, never loosened: "upward flip" is any
answer with F > O, not only 0 to 1; the reproduction prediction holds
only if each of the two tables reaches 90%; the AD1 flip counts take an
answer as moved if it moves under any version in force for its model;
and AD2 = NULL and AD3 = NO RANK CHANGED hold only if they hold under
the largest F − O and under the rule they were written under.

## Frozen target set

**The task.** `tablejoin` only: 50 questions, all released 2024-06-24,
none removed. The prompt asks for "a Python dictionary". `tablereformat`
is out: every version of its scorer in the window strips a fence that
opens at the first character of the text it receives
(`re.sub(r'^```.*\n', "", s)`) and then removes every remaining
```` ``` ````, so a leading-fence strip can change its input only when
the fence follows leading whitespace (then the strip also removes the
info line, which LiveBench's code leaves in). AD does not measure that
case. Every other task is out.

**The answers.** `livebench/model_answer` at revision
`bc43a286b08243d6dcfa89ba2e906283e9adac9c`, read from its parquet, not
from a datasets-server page. Rows with `task == "tablejoin"`. LiveBench's
loader keeps the last answer per (model, question); so does the runner,
in parquet order, and it banks the duplicate count. Model ids containing
`/` (six local paths) are out. That is AD's choice: they are local
paths, not published names, and LiveBench's `download_leaderboard.py`
skips them from `3953c6c` (2024-10-22T03:16:49Z; seven minutes after
this answer snapshot was uploaded), not at `905b71f`.

**The ground truth.** `livebench/data_analysis` at revision
`31b9661ff678df9958e2f7fa228427f4c858c1a1`, rows with
`task == "tablejoin"`, passed to the scorer as the string the dataset
stores, as LiveBench's judgment does. That revision was uploaded on
2025-04-07, after the five tables. Earlier revisions exist (`9b73ab8`,
2024-10-22; `47fc858`, 2024-07-27; `4362c1a`, 2024-06-26). The
pre-freeze audit compared their `tablejoin` rows with the pinned one by
equality only, printing no text: the same 50 question ids, and no
difference in `turns` or `ground_truth` at any of the three; the release
date column is absent at `4362c1a`. So the pin does not change what any
2024 run scored against.

**The published numbers.** The `tablejoin` cell of five tables,
`<date>` in `2024_06_24`, `2024_07_26`, `2024_08_31`, `2024_11_25`,
`2025_04_02`, each table's categories file and the site's averaging
code, pinned at site commit `19e766a5de4de07d672ed5bf9f0a69ceed1d39bf`.
The runner also fetches each live table from
`https://livebench.ai/v1/table_<date>.csv`. **If a live table differs
from the pinned one, the live file is the published number**, its upper
time bound becomes its fetch time, and the difference publishes first.

A cell is **in frame** when its row's model name equals an answer-set
`model_id` exactly, or case-insensitively when exactly one id matches
(LiveBench's results script lower-cases model names). Aliases are not
guessed; unmatched and ambiguous rows are listed by name, out of frame. A
matched row with no number in its `tablejoin` cell is listed and is not
a cell. Counted from names alone during scoping: 67, 67, 44, 14 and 9
cells, **201 cells, 75 distinct models**, out of 73, 73, 59, 65 and 63
table rows (333): 132 rows are not in frame and are never read. If the
run counts differently, the difference publishes first. Later tables
use refreshed `tablejoin` questions and post-fix code and are out.

**The scorer versions.** Every first-parent state of LiveBench's `main`
that changed any of three files, from `905b71f` (2024-06-14) to
`e647d72` (2025-04-18) inclusive:
`livebench/process_results/data_analysis/tablejoin/utils.py`,
`livebench/process_results/util.py`, and the answer extraction in
`livebench/gen_ground_truth_judgment.py`. The extraction is defined as
the run of consecutive assignments to `llm_answer` in `play_a_match_gt`,
lifted from each version's file by syntax tree and run as written: the
line that reads `choices[0].turns[-1]`, and from `fa6a3c6` the
`<think>` removal after it. A version is a state whose tuple of digests
(tablejoin utils, util, extraction) differs from the previous state's.
The manifest banks all 20 states and the 7 versions; the runner checks
that the version list is the de-duplication of the states and that each
version's vendored files are that state's files.

The window starts at `905b71f`, not at the first public state,
`686be1e` (2024-06-12T15:06:47Z). That state already scored a correct
```` ```json ````-fenced answer 0.0, but on the string ground truth the
dataset stores it raises `AttributeError` on every synthetic answer it
can parse (`experiments/ad_synthetic_scorer_check_2026-10-21.json`), and
LiveBench's run stops on such an exception. So it could have produced
only a cell in which no answer parsed, and `905b71f`, which parses the
same way (survey, section 2.3), gives that cell the same O.

A version is **in force** from the committer time of its commit on
`main` (push time is not public; committer time stands in for landing)
until the next version's. The last version, `e647d72`, is in force
until the window closes: `84c4667` (2025-04-29T22:13:09Z), the next
change on `main` to any of the three files. Nothing later is pinned.

A cell's **in-force set** is every version in force at some time between
its lower bound and its upper bound. The lower bound is the newest
`tstamp` among that model's stored `tablejoin` answers: a cell cannot be
scored before its answers exist. The upper bound is the committer time
of the last commit that changed the table file (manifest, equal to its
author time for all five: 2024-09-07T04:35:50Z for `2024_06_24` and
`2024_07_26`, 2024-11-21T22:10:05Z for `2024_08_31`, 2025-03-31T21:17:10Z
for `2024_11_25`, 2025-04-18T20:08:41Z for `2025_04_02`). For
`2024_06_24` and `2024_07_26` that set is the `905b71f` state only. If
the lower bound is after the upper bound, the stored answers are newer
than the table, no version could have produced the published number
from them, and the cell is **ANSWERS POSTDATE TABLE**: listed, counted as
not reproduced, and counted per table against the draft's rule
(Correction 3).

**The manifest.** `experiments/ad_manifest_2026-10-21.json` names every
remote file the reading downloads, 73 in all: the two parquets (Hugging
Face LFS sha256, the host's own digest), the five pinned tables (the git
blob id the host publishes, and the sha256 computed at scoping), the five
live tables (size only; the host publishes no digest, so each is compared
with its pinned twin byte for byte), the five category maps and
`Averaging.js`, the `2025_04_02` table at `ad51514`'s parent (blob id
only), the scoping feasibility page, and the three scorer files at each
of the 20 states (53 files: `util.py` does not exist in the first seven
states). It also carries the version list, the vendored files' digests,
the frame counts from scoping, the feasibility disclosure and the
environment pins (Python 3.11, numpy 2.4.6, pyarrow 25.0.1; a different
version at run time is recorded, not refused).

| Item | Pin | Digest at scoping |
|---|---|---|
| answers parquet `data/leaderboard-00000-of-00001.parquet` (56,430,153 bytes) | HF `livebench/model_answer` @ `bc43a286b08243d6dcfa89ba2e906283e9adac9c` | `1d4b5547c2ae2ea40280ae452885a39c7d395a7a0abac643868231cc5187a4a5` |
| questions parquet `data/test-00000-of-00001.parquet` (144,796 bytes) | HF `livebench/data_analysis` @ `31b9661ff678df9958e2f7fa228427f4c858c1a1` | `fb86a7a02fa9eabf785d9e8af85955990cf6d228cc9d3f805a54f863bc8c4c52` |
| `public/table_2024_06_24.csv` (8,879 bytes) | site @ `19e766a` (blob `b22e753`) | `df181166a5a670a5b7678ee02f96c6896d8a88a503aec20243234b698d25a8c0` |
| `public/table_2024_07_26.csv` (9,261 bytes) | same (blob `d042001`) | `670dc4af2497d2daca7b5179f9558854163d954a95939bc4a7d71d8fce298a85` |
| `public/table_2024_08_31.csv` (7,694 bytes) | same (blob `6ad8a18`) | `8c66d248110076378f7eb58aaeef524c65bba6dff0b55f190fd9ac8e92d73b2a` |
| `public/table_2024_11_25.csv` (8,508 bytes) | same (blob `c2ab984`) | `17ca6b3d122921d24b591d0da52566c5552ed4f4001ae9d83a11b2712325aae4` |
| `public/table_2025_04_02.csv` (8,441 bytes) | same (blob `c99d92f`) | `d7fd6a3e461cff4f96569093553789115e87adeb9cd95dca043bafa51ff2acaa` |
| `src/Table/Averaging.js` | same | `aba4395c7a32d3bf0fe40d2fb05abafa8d08a7d4190794ff8441b03b4e0f8b02` |

## Design: two scorers, exactly one change

The unit is one **answer**: one model's text for one question. Both
arms run LiveBench's own code, vendored byte for byte and digest-pinned,
never retyped.

- **O (the original).** For a version: that version's extraction, then
  that version's `joinmap_process_results(question_text, ground_truth,
  answer)`, as shipped.
- **F (one change).** The same, except the extracted text is passed to
  the shipped `fencecheck.strip_fence` just before the scorer. **If it
  reports `was_fenced`, F scores the stripped body. If not, F is O**: the
  original text is scored byte for byte, so the `.strip()` inside
  `strip_fence` is not a second change. F can differ from O only on an
  answer that opens with a fence, in either direction (Correction 5).

If the extraction or the scorer raises on an answer, the exception's
type is recorded. LiveBench's run stops on such an exception (from
`19c7757` it re-raises), so a version under which any of a model's
answers raises under O cannot have produced that model's published
cell. If only F raises, that answer keeps its O, and the count is banked
beside the cell.

Units follow LiveBench: an answer scores in [0, 1] (an F1 rounded to two
decimals by `np.round`); a cell is the mean over the model's answers of
score times 100 (`show_livebench_result.py` at `905b71f`, lines 22 to
31, which rounds to three decimals). One answer moving from 0 to 1 moves
a 50-answer cell by exactly 2.0 points; a partial F1 moves it by less.
The Data Analysis average is the mean of its task columns and the Global
Average is the mean of the category averages, as `Averaging.js`
computes them (every category checked, the site's default; `toFixed(2)`
with JavaScript's rounding). Rank is position by displayed Global
Average, descending, over every row of the CSV; tied models share the
higher position; a row without a Global Average is unranked. The 2026
site's row filters are not applied.

## Gates, in order

Each publishes whichever way it reads. A failed gate stops what it says
it stops, and nothing else.

- **G0, manifest.** `fetch` downloads exactly the manifest's files. A
  file not received in full stops `fetch` with nothing banked: any
  transfer that fails, a body shorter than the length the host
  announced, or a pinned file received at a size other than the
  manifest's (every pinned file is immutable at its commit or revision,
  so a size that differs is a transfer fault). `fetch` may then run
  again; every attempt and its outcome are logged. If a file cannot be
  received at all, only a dated erratum moves the run on. Of the files
  received in full: every `refuse` entry must match every digest the
  manifest gives; every vendored file must equal its manifest digest and
  the bytes downloaded from its commit; every state's digests must
  re-derive from the downloaded files; the version list must be the
  de-duplicated states. If not, nothing is scored, and the run waits for
  a dated erratum to this protocol. A live table that differs from its
  pinned twin is recorded and publishes first: the served file is the
  published number. The feasibility page that differs from the scoping
  digest is recorded and publishes first in its own words (it is not a
  published number); G2 then compares texts. Neither stops anything.
- **G1, the synthetic check repeats.** Each version scores the five
  synthetic answers of the survey (section 4) and must give the same O
  and F as the scoping run, against
  `experiments/ad_synthetic_scorer_check_2026-10-21.json`. The scoping
  script stripped every answer for F; the runner strips only an answer
  that opens with a fence (Design). On these five answers the results
  are the same, because the two unfenced ones have no surrounding
  whitespace for the strip to remove. `fa6a3c6` is checked against the
  scoping key `c5c293c` and `f3501c7` against `e2a8e99`, whose scorer
  files are byte-identical. If not, nothing is scored, and the mismatch
  is the result.
- **G2, the feasibility page is the parquet.** The runner fetches the
  scoping page again. If its sha256 is the scoping digest, the 50
  `gpt-4o-2024-05-13` texts from the parquet must equal the page's 50
  texts byte for byte, and their tags, counted by the scoping script's
  rule, must be 50 `python`. If not, the difference publishes first; the
  run continues on the parquet.
- **G3, reproduction, per cell.** A cell is **REPRODUCED** when some
  version in its in-force set gives an O cell within the column's
  tolerance of the published cell (Correction 4: half a unit of the
  d-th decimal, d the most decimals any cell of the column carries, at
  most 3; never below 0.0005 points). That the site's CSVs are the
  results script's output is not verified, and this gate is what checks
  it. A cell reproduced by no version in its set is **NOT REPRODUCED**:
  its O and F under every in-force version are banked and listed, and it
  never counts toward a bar, because then we do not know what code
  produced the published number. If several in-force versions reproduce
  a cell, F − O is banked under each. The claim (AD2's words, AD3's
  words and the licensed sentences) uses the smallest: only an upward
  move counts (Correction 5), so a cell moves up only if it moves up
  under every version that reproduces it, and a downward strip error is
  counted wherever one reproducing version shows it. The predictions use
  the largest (Correction 6), and both readings publish. **If fewer than
  half of the in-frame cells reproduce, AD2 and AD3 are WITHHELD (NOT
  REPRODUCED)** and only AD1 publishes, with the sentence: *we could not
  reproduce LiveBench's published `tablejoin` numbers from its stored
  answers, so nothing is said about them.* Cells that are ANSWERS
  POSTDATE TABLE, or whose model has no stored answers, count in the
  denominator as not reproduced.

## Readings, applied by arithmetic in `published_number_audit.py read`

- **AD1, answers whose score moves.** Over every in-frame answer: every
  stored answer of every model with at least one in-frame cell, each
  (model, question) once. Each is read under the reading version of the
  earliest table that shows the model: the version that reproduced that
  cell (the smallest F − O) if it reproduced; else the earliest version
  in force for it; else (no version is in force for it: its answers
  postdate the table) the version on `main` when the model's newest
  stored answer was made. Models read by that last rule are listed. The
  count with F > O (up), F < O (down), F = O; among the ups, the count
  with O = 0 and F = 1 (full flips); every count split by the opening
  fence's info string, exact case (`python`, `json`, empty, other, a
  one-line fence) and by no fence. Also the fence prevalence itself: how
  many in-frame answers open with a fence. Beside these, the harder
  count the flip predictions use: an answer counts as up (down) if
  F > O (F < O) under any version in force for one of its model's
  cells, or under its reading version when none is. No bar. It says
  whether the defect was ever reached by a real stored answer.
- **AD2, published numbers that move.** For each REPRODUCED cell:
  published value, O, F and F − O in points, under each reproducing
  version. A cell **MOVED UP** when F − O ≥ 2.0 (one question's worth).
  A cell with F − O ≤ −2.0 is the strip's own error (Correction 5): it
  is counted, listed and published beside the words, and never counts
  toward them. Count distinct models with at least one cell moved up (a
  model appears in up to five tables; the same answers are not counted
  twice), with the cell count beside it. Also counted: the cells that
  moved up by less than 2.0, with the largest such move, and the cells
  the strip lowered. Each cell moved either way also publishes its Data
  Analysis and Global Average before and after.
- **AD3, ranks that move.** For each table, add to every REPRODUCED cell
  its upward F − O (0 when F − O is not upward), keep every other
  published number as it is, recompute the Global Average and the rank.
  Count models whose rank changes, per table, with the largest move and
  whether any top-10 position changed.

AD2 and AD3 are each computed twice: under the smallest F − O among the
versions that reproduce each cell (the claim) and under the largest (the
predictions). Both publish.

The words of AD2 are the whole vocabulary, read under the smallest
F − O:

- **NULL**: no model has a cell moved up.
- **MOVES A NUMBER**: 1 or 2 models do, named.
- **MOVES NUMBERS**: 3 or more models do.

And AD3 adds **RANK CHANGED** or **NO RANK CHANGED**.

**The bar for the sentence.** The reader emits these sentences, and
nothing else may be written about whether the fence moved a published
number. Each one carries its reach: *"K of the I in-frame cells
reproduced, and only those are read; R of the T table rows were not in
frame."* Each one adds, when the strip lowered any reproduced cell:
*"The strip itself lowered L reproduced cell(s), L2 of them by a
question's worth or more (largest X points): that is fencecheck's own
error, not LiveBench's."*

- At **MOVES NUMBERS**: *"The fence failure moved published evaluation
  numbers: on LiveBench's 2024 `tablejoin` task, K models, cells listed
  (C cell(s) up by a question's worth or more under every scorer version
  that reproduces them)."*
- At **MOVES A NUMBER**: the same in the singular, naming the models.
- At **NULL**: *"Re-scoring LiveBench's stored 2024 `tablejoin` answers
  with one fence strip moved no published number up by a question's
  worth (2.0 points), in N reproduced cells over M models."* Then the
  reach; then either *"S reproduced cell(s) moved up by less than that
  (largest X points)."* or *"No reproduced cell moved up at all."*; then,
  when the largest F − O moves any cell up by 2.0 or more, *"Under
  another scorer version that also reproduces them, U cell(s) over V
  model(s) would move up by a question's worth or more."*; then the
  ranks: *"Ranks still changed: ..."* (AD3 RANK CHANGED), or *"No rank
  changed, but under another scorer version that also reproduces the
  cells one would: ..."*, or *"No rank changed."*; then the strip's
  error, if any; then *"The fence failure was real in the scorer."* Only
  when every reproduced cell has F − O = 0 under every version that
  reproduces it does that end *"; in these N cells it changed no
  published number and no rank."*

The reader also emits the AD3 sentence and a directions line (cells
moved up, cells the strip lowered), and the exact text that replaces the
open question in `tools/README.md`: the AD2 sentence, followed by the
AD3 sentence whenever AD3 reads RANK CHANGED and AD2 is not NULL (the
NULL sentence already carries the ranks).

**Secondary, banked beside the readings, never used for a bar:**

- **AD-S1, the owner's fix.** For each REPRODUCED cell, O under
  LiveBench's own `e647d72` scorer minus the cell's O. It differs from O
  in more than one way, so it is context, not the experiment. It says
  what LiveBench's fix would have changed had it been applied to these
  tables.
- **AD-S2, the wider scope.** A third arm, A: `fencecheck.extract_json`
  with its opt-in wider scope, run where F runs the leading strip. It
  takes the leading strip first; else the first fenced block anywhere (a
  fence after prose); else a bare object in prose, which must parse as
  strict JSON with no other brace or `"key":` around it, so a
  single-quoted Python dictionary in prose is never found; else the text
  unchanged (A = O). Counts only.
- **AD-S3, the 2025-04-18 table edit.** Whether commit `ad51514`, made
  5 hours 25 minutes after the fix, changed any in-frame `tablejoin`
  cell of `table_2025_04_02.csv`, read from its parent's file and its
  own at run time, not before.

## Predictions, written before the run

Written after the one feasibility file above was seen (50 of 50 tagged
`python` by a lower-casing count), and stated with that in mind.

- **G3 = REPRODUCED** on at least 90% of the in-frame cells of
  `2024_06_24`, and on at least 90% of those of `2024_07_26`. The other
  three tables are not predicted.
- **AD1:** more than half of the in-frame answers open with a fence, and
  more than half of those fences are tagged `python` in exact case.
  Upward flips (any answer with F > O under any version in force for
  its model) are fewer than 2% of in-frame answers. Downward flips
  (F < O, counted the same way) are at most 5.
- **AD2 = NULL**, under the largest F − O among the versions that
  reproduce each cell, and under the rule this prediction was written
  under (the smallest F − O, a move of 2.0 either way counting). It
  holds only if both read NULL. The prompt names Python, every scorer
  version extracts the ```` ```python ```` tag, and the census prior is
  that the defect is rare where people look.
- **AD3 = NO RANK CHANGED**, under the largest F − O, and under the
  drafted rule (every reproduced cell plus its whole F − O). It holds
  only if both read NO RANK CHANGED.

Each of these can fail in public. A failure publishes as written. The
reader checks each one and banks HELD, FAILED, or not checkable when
its reading is withheld.

## What the scanner says about this scorer

`python3 -I -B tools/fencecheck.py scan`, as pinned above, flags four of
the five vendored `tablejoin` scorer versions and passes the fifth, run
2026-10-22 on the vendored files. On `905b71f`, `f774d19`, `19c7757` and
`c5c293c` it reports one site each, the `literal_eval` of the extracted
answer (lines 12, 14, 23 and 27), and exits 1. On `e647d72`, the owner's
fix, it prints "1 file(s) scanned, nothing exposed." and exits 0. G1
executes the first four as scoring a correct ```` ```json ````-fenced
answer 0.0 and the fifth as scoring it 1.0, so on these five files the
pinned scanner agrees with the executed behaviour.

The scanner as it stood when this protocol was drafted (`c35abfa`,
sha256 `55cb41de7bbd26311065fc903029eac5c430a7317a5d9e420239f8d0990b0ac3`)
reported nothing exposed on all five, exit 0 (seen during scoping and
the build on 2026-10-21 and 2026-10-22, and re-run on that file by the
pre-freeze audit): a false negative on four real leaderboard scorers.
The rework on `main` (`7f9068a`, 2026-10-22, "a finding now needs
evidence the parsed text is model output") removed it. Both facts stand.
AD does not use `scan`. A test in `tests/test_published_number_audit.py`
fails if the pinned scanner's verdicts on these five files change, so a
later change to the scanner cannot leave this section false at the
freeze. Whether and how to record the earlier miss in `tools/README.md`
is the founder's call; this protocol records it so it cannot be
forgotten.

## What the run obliges

**Binding whichever way it reads.** On the day the reading lands,
`tools/README.md`'s sentence *"Whether that bias shows up in published
work is a separate question that this tool does not answer."* is
replaced, word for word, by the reading's `readme_replacement` text, at
the same place and size, with its scope (one task of one benchmark,
2024 answers) as the sentence states it. A NULL takes that place exactly
as a positive would; it is not moved to a footnote and is not softened.
`VERDICT.md` gets an AD row with the words under both readings and
every prediction as checked. If G3 withholds, the row says so and the
README sentence stays, with a pointer to the row.

Nothing is sent to LiveBench or anyone else by the runner. Whether to
tell the maintainers, and when, is the founder's call. If AD2 reads
MOVES A NUMBER or MOVES NUMBERS, no outbound document uses it until that
call is made and recorded.

## How it runs

```
ots verify docs/research/ADDENDUM_AD_PROTOCOL.md.<stamp>.ots   # against a node; commit its output
python3 scripts/published_number_audit.py check   # every stage's preconditions; touches nothing
nice -n 19 python3 scripts/published_number_audit.py fetch
nice -n 19 python3 scripts/published_number_audit.py score
python3 scripts/published_number_audit.py read
```

Each stage checks every precondition before it does any work, and
writes its artifact once, exclusively and atomically (a synced temp file
with a unique name, hard-linked into place; an existing artifact is
never overwritten). Before any work it appends its start clock to
`experiments/ad_attempts_2026-10-21.jsonl`, and an attempt that banks
nothing appends its outcome, so every attempt stays on record and the
next banked artifact lists it. Without `--out`, `score` and `read`
make the banked artifact only from the banked inputs: `--fetch` and
`--score` are refused without `--out`. `read` withholds and writes
nothing while a stage is missing or incomplete, or while the score
artifact was not made from the fetch artifact it is given. Data paths
are recorded relative to the repository, and `score` checks that every
fetched file on disk is still the bytes `fetch` received; if one is
missing (`data/` is not in git) or changed, `score` refuses before any
work, and `restore` downloads it again, accepting only the exact bytes
the banked `fetch` received. A file the host now serves differently
cannot be restored; only a dated erratum moves the run on.

After the banked artifacts exist, anyone can recompute into new files
from a clone that has no `data/`:

```
python3 scripts/published_number_audit.py fetch --out F --data-dir D
python3 scripts/published_number_audit.py score --out S --fetch F
python3 scripts/published_number_audit.py read --out R --fetch F --score S
```

A recompute never writes over a banked artifact or into the banked data
directory, and its attempts are logged with their output paths.

## Cost

No model calls, no GPU, $0. Download about 57 MB (the answers parquet
is 56,430,153 bytes; the questions parquet 144,796; the other 71 files
are small text) into `data/ad/`, which git ignores. Scoring: 4,350
answers × 7 versions × up to 3 arms, each a regex and one
`literal_eval`: seconds to minutes of CPU. Memory: the answers parquet
is 159,113,809 bytes uncompressed; the runner reads it in 2,048-row
batches of five columns and banks its peak RSS. It needs numpy
(LiveBench's scorer calls `np.round`) and pyarrow. It does not run while
Addendum Z holds the box, unless its measured peak is under 1 GB at
`nice -n 19`.

## What this cannot show

- **One task of one benchmark.** A null here does not bound any other
  leaderboard, and a positive here is not a rate. The survey checked
  eleven result sets, chosen by hand, not a census.
- **A prompt that asks for Python.** The answers were asked for "a
  Python dictionary". A model that follows the prompt tags its fence
  `python`, which every version extracts. A task that asks for JSON
  would load the same defect very differently.
- **2024 answers only.** The public answer snapshot ends 2024-10-17. The
  commit that fixed the scorer is titled "gemini 2.5 flash preview
  support", which suggests (by inference, not checked) that a 2025
  model's answers prompted it; those answers are not public and are not
  in AD.
- **Only the reproduced cells.** A NOT REPRODUCED or POSTDATE cell says
  nothing, and about 132 of the 333 table rows are never in frame. Every
  sentence states how many cells it covers.
- **What code produced a published number.** "In force" means landed on
  LiveBench's `main`, by committer time. The maintainers may have scored
  with code that was never committed, or with a different Python whose
  `literal_eval` differs. G3 is the only defence, and a NOT REPRODUCED
  cell says nothing.
- **The strip is not a perfect fence handler.** F is fencecheck's
  `strip_fence`, which can score an answer lower than LiveBench's own
  code did (Correction 5). A cell's F − O is net: an answer the strip
  lowers offsets one it raises in the same cell, and both counts are
  banked per cell.
- **Correctness.** F uses LiveBench's own F1 against LiveBench's own
  ground truth. AD says whether the fence changed a score, not whether
  the ground truth is right.
- **Reliance.** It cannot show that anyone chose a model on these
  numbers.
- **Not blind.** One model's fence tags, every table's model names, and
  the scorer's behaviour were seen before the predictions were written,
  and the held-out bytes listed above sat on disk before the freeze
  (above).

Artifacts: `experiments/ad_fetch_2026-10-21.json` (G0, every file as
received), `experiments/ad_score_2026-10-21.json` (G1, G2, the versions
with their extraction source, per-answer O, F and A under every version,
per-cell published value, in-force set, O, F and A under each version,
status, and F − O under each reproducing version),
`experiments/ad_reading_2026-10-21.json` (G3, AD1 to AD3 under both
readings, the secondary readings, the licensed sentences and the README
replacement, the predictions checked, the obligations),
`experiments/ad_attempts_2026-10-21.jsonl`. The date in each name is
this protocol's, not the run's; each artifact banks its own clocks.
Runner: `scripts/published_number_audit.py`. Tests:
`tests/test_published_number_audit.py`, on synthetic answers and
synthetic tables only. Survey: `docs/research/AD_SURVEY_2026-10-21.md`.
Pre-freeze exposure listing: `experiments/ad_preexposure_2026-10-22.json`.
