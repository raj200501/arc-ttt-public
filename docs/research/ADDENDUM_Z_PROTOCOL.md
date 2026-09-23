# Addendum Z — the +46.5 gate, run again with every prediction stored (preregistration)

**Frozen 2026-09-23, before any arm ran. Anchored before the first arm as
`ADDENDUM_Z_PROTOCOL.md.2026-09-23T0219Z.ots` — the first protocol
under the ANCHORS.md policy. What that proves, exactly: these bytes
existed by the stamp. That every arm started after the stamp rests on
the runner's own clock (`started_utc` in each arm's record) and is ours
to report, not the anchor's to prove.**

The most-cited number in this repository is the one a stranger can check
least. `scripts/verification_coverage.py` classifies the Addendum B gate
— `novel_schema_summary_2026-08-12.json`, mean paired delta **+46.5** at
k=30 over seeds {1,2,3} — as AGGREGATE, and its six arms as ARITHMETIC:
per-receipt scores were banked and the model outputs were discarded at
generation time, so the step from output to score cannot be re-checked
by anyone, including us. The coverage artifact as it stood at source
commit `0f65fa1` named the fix and priced it: *"Re-running those arms
with predictions stored is the fix, it costs about nine CPU-hours, and
it is not done."* The price was wrong (corrected in `CORRECTIONS.md` the
day this protocol freezes; the measured figure is below), and the fix is
what this addendum does.

## Design — the Addendum B recipe, unchanged, with nothing thrown away

Exactly the frozen Addendum B gate (`ENTERPRISE_EVAL_SPEC.md` B.2–B.5,
execution revisions B.7-r2/r4/r6), as the banked kernel
`kaggle/entry_novel_schema_s1a.py` ran it on Kaggle's CPU sessions:

- corpus: `arcttt.novel_schema.make_task(seed, n_train=30, n_test=60,
  task_id="novel-0.5b-k30-seed{seed}")`, seeds **1, 2, 3** — three
  invented tenants, generation deterministic in the seed;
- model `Qwen/Qwen2.5-0.5B-Instruct`, CPU, float32, SDPA attention, at
  snapshot `7ae5576` (the banked kernels ran with internet on and no
  revision, so which snapshot they resolved is not recorded; this one is
  pinned and its commit hash, chat-template sha256 and generation
  defaults are written into every artifact);
- adapted arm: fresh LoRA rank 16 / alpha 32 (rsLoRA scaling), AdamW
  5e-5, **one epoch** over the leave-one-out corpus with demonstration
  order shuffled by the arm seed, `max_sequence_tokens` 8192, chunked
  cross-entropy at 512 tokens, gradient checkpointing; k-shot arm: the
  same call with zero epochs (an untrained adapter is an identity);
- decode, both arms: the full 30-demonstration prompt (spec B.9.1 — the
  delta is adaptation *on top of* in-context prompting), vote/rescore,
  1 greedy + 4 sampled at T=0.7, `max_new_tokens` 512, canonical-JSON
  pooling, count + likelihood top-1 (`predict_text_voted`, step for
  step). Stated because the banked run never did: every pass goes
  through `model.generate`, which applies the checkpoint's own
  generation defaults — repetition penalty 1.1 on all five passes, and
  top-k 20 / top-p 0.8 on the four sampled ones — so "greedy" is the
  argmax under that penalty, then as now;
- scorer `score_text_output` (field micro-F1, exact match, valid JSON);
  a prompt over the 8192-token cap returns no prediction and the
  document is excluded from the mean, never scored 0 (B.9.2);
- validity windows on the k-shot arm, floor 0.15 / ceiling 0.95 (B.5);
- the verdict computed by `scripts/novel_schema_summary.py`
  **unchanged**, under `--date 2026-09-23`: the two-statistics rule,
  +5.0 bar, receipt-level interval and sign test.

**What is kept this time, per document:** every completion in the pool
in decode order, each distinct completion's mean supervised-token
log-probability, the pooled candidates with their counts, the selected
text (`prediction`), the parsed score unrounded, the prompt's token
count and sha256 (for an over-cap document too), and the wall seconds.
Per arm: the adapter's sha256 and its weights on disk under `work/z/`
(not committed, nor are the journals), the training example count
within the cap, the environment (torch, transformers, Python, CPU,
thread count, model snapshot), adaptation and decode time, and the
number of process restarts. The arm mean is canonical as: raw micro-F1
summed in document-index order over the scored documents, divided by
their count, rounded to four places. **No arm-level number from a
2026-09-23 artifact is quoted anywhere before the reading lands**; the
arms are banked as they finish and are not cited.

**Execution.** One process, the six arms in order (seed 1 adapted,
seed 1 k-shot, seed 2, seed 3), `torch.set_num_threads(4)` pinned and
recorded (Addendum Y: a thread count is a reduction order). The run is
expected to outlast this container, so it is resumable exactly as the
banked kernels were (B.7-r6): the adapter is saved once after training,
its digest recorded, and it is restored and re-digested on restart
instead of retrained (an unusable file is discarded and the arm
retrains); every scored document is journaled and never re-decoded, and
a line torn by a kill mid-write is dropped and its document re-decoded.
Every process start after the first is a restart, whether the arm died
during adaptation, between the adapter and its record, or mid-journal;
the count publishes per arm.

**Cost, from one document decoded on this box before the freeze and
the banked adaptation timings:** an 8,068-token k=30 prompt takes ~52 s
per generation and ~58 s per distinct-completion rescore at 4 threads
and 11.9 GB peak RSS, so a document costs 5×52 s + (1…5)×58 s ≈ 5–9
minutes and a 60-document arm 5–9 hours (seed 2's 38 scored documents,
3–6 hours). Adaptation is not measured here; the banked Kaggle arms took
9,651 / 11,498 / 10,496 s (2.7–3.2 hours). Sum: 4 × (5–9 h) + 2 × (3–6
h) + 3 × (~3 h) ≈ **35–57 wall-hours on four cores for the six arms**,
stated as 40–60, not nine. That one smoke document (seed 1, index 0,
k-shot, greedy + one sample) scored 0.75 and was seen before the
predictions below were written; it is one of 60 and is not excluded.

## What differs from the banked run, stated so it is not discovered later

- **The box.** These arms run on this container's CPU; the banked arms
  ran on Kaggle CPU sessions whose library versions were not recorded
  (the arms carry no environment record — the reason every artifact
  since Addendum X carries one). A difference between the two runs is
  therefore run-to-run plus environment, and this addendum cannot
  separate the two. It measures the sum.
- **The two unseeded draws.** The four sampled completions per document
  are unseeded, then as now (B.9.6). So is the LoRA-A initialisation —
  `kaiming_uniform_` from the global RNG, with no `manual_seed` anywhere
  in the kernel or this runner — so a trained adapter is itself a draw:
  two runs of the same arm train different adapters before a single
  document is decoded. The banked run had both sources; B.9.6 named only
  the first. The greedy completion is deterministic given the weights
  and the box, and the weights are not. This is a fresh measurement of
  the same quantity, not a byte re-derivation of the old one, and a
  restart during adaptation retrains a different adapter (counted).
- **The weights.** The banked kernels loaded `Qwen/Qwen2.5-0.5B-Instruct`
  with internet on and no revision, so which snapshot they resolved is
  not recorded; this run pins `7ae5576` and records it. The kernel also
  asked for SDPA attention with a silent fallback and did not record
  which it got; this run records what it got. One process ran one arm
  there; one process runs six here.
- **Nothing else that is recorded.** Same generator, same seeds, same
  model name, same config, same decode, same scorer, same reader.

## Readings — applied by arithmetic in `scripts/novel_schema_rerun.py --read`

The reader withholds every quantity, and writes nothing, until the six
exact arm files exist, each carrying this protocol's name and its
predictions, and the unchanged reader's verdict on them is decidable.

- **Z1 — the gate.** The verdict word of the unchanged reader on the
  2026-09-23 arms: GO / PIVOT / UNINFORMATIVE. UNDECIDABLE is WITHHELD.
- **Z2 — agreement with the banked run.** Let `m` be the new gate mean
  and `m₀ = +0.4648` the banked one, both as the reader rounds them
  (four places). **AGREES** — same verdict word and `|m − m₀| ≤ 0.05`
  (the bar itself, as the tolerance, in either direction). **GO BUT
  MOVED** — both GO and `|m − m₀| > 0.05`, in either direction.
  **DISAGREES** — a different verdict word. These three words are the
  whole vocabulary; the banked word is GO by construction.
- **Z3 — the cross-environment spread.** Per arm, `|new mean − banked
  mean|` in micro-F1; the largest of the six publishes, by dated erratum
  beside spec B.9.3, as an **upper bound on run-to-run spread** for these
  arms — run-to-run plus environment, which this design cannot separate.
  If it is larger than B.9.3's single-pair 0.0098 it is the figure quoted
  from then on; if it is not, B.9.3's figure stands and this one sits
  beside it. Frozen text keeps its bytes either way. No bar. The
  prediction below publishes as **WITHIN PREDICTED SPREAD** or **SPREAD
  LARGER THAN PREDICTED**.
- **Z4 — attrition.** The set of over-cap documents per arm must equal
  the banked set (seed 2: the same 22 in both arms; seeds 1 and 3: none).
  **SAME EXCLUSIONS** / **DIFFERENT EXCLUSIONS** — the cap is a property
  of the corpus and the tokenizer, not of the box, so a difference is a
  generator or tokenizer change and is read before anything else. The
  banked arms journaled no reason for an exclusion; every banked
  exclusion is taken as over-cap per B.9.2. A new document whose five
  completions were all empty is sampling attrition, published under its
  own reason and not counted here. **If Z4 reads DIFFERENT EXCLUSIONS,
  the two runs scored different document sets: Z2 and Z3 publish as NOT
  COMPARABLE, the citation rule below is suspended, nothing moves on any
  page, and the discrepancy goes to `CORRECTIONS.md` first.**

**Citation rule, binding whichever way Z2 reads.** From the day the
reading lands, every page that quotes +46.5 quotes the primary-verifiable
number instead — README, VERDICT, EVIDENCE, `scripts/verify_verdict.py`
by a second date — with the banked number kept beside it as history.
Frozen text keeps its quotes: every frozen site that carries +46.5 (the
spec's B.7-r6 board note, the Addendum C preamble, B.9.7) gets a dated
erratum, and only unfrozen pages are edited. If Z1 is not GO, the
headline is **withdrawn**: the two runs disagree, the disagreement is
the result, `CORRECTIONS.md` carries the row, and no page keeps the
banked GO on the strength of its being first. If Z1 is GO and Z2 reads
GO BUT MOVED, the movement publishes beside the number at full size. A
larger number is not promoted on the strength of being larger; a
smaller one is not softened on the strength of being second. On the
same day: the coverage map (`experiments/verification_coverage.json`) is
regenerated so the six PRIMARY arms and two AGGREGATE files count, and
its sentence that the fix "is not done" is retired together with the
test that pins that phrase.

## Predictions, written before the run

- **Z1 = GO.** The banked effect is ~9× the bar with a 156W/0L sign
  test; a fresh sample stream on the same recipe should not lose it.
- **Z2 = AGREES.** `|m − m₀| ≤ 0.05`.
- **Z3 ≤ 0.03** per arm. The one honest same-environment duplicate was
  0.0098; a different box should cost more, not three times more.
- **Z4 = SAME EXCLUSIONS**, and seed 2 excludes exactly 22 documents in
  both arms.

Each of these can fail in public; a failure publishes as written.

## What this cannot show

It cannot make the +46.5 an independent replication — the same operator,
generator, model and recipe on a second CPU box is a replication by us,
which makes one by a stranger *possible* (every prediction is stored)
rather than *done*. It does not touch the B.9.1 scoping (the delta is
adaptation on top of the same model's 30-shot prompt, at 0.5B), the
B.9.5 corpus limitation (one schema geometry, vocabulary re-rolls), the
CORD negative (B.6: the positive is always stated beside it), or the
scale question (Addendum C). Six arms, one rung, one box.

Artifacts: `experiments/novel_schema_0.5b_k30_seed{1,2,3}_{adapted,kshot}_2026-09-23.json`
(the arms, PRIMARY), `experiments/novel_schema_summary_2026-09-23.json`
(the unchanged reader's verdict), `experiments/novel_schema_rerun_2026-09-23.json`
(Z1–Z4 and the predictions). Runner and reader:
`scripts/novel_schema_rerun.py`; readings pinned at their boundaries by
`tests/test_novel_schema_rerun.py`.
