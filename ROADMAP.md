# Roadmap (updated 2026-09-23)

The company is the eval-integrity work: the fence tax, the shipped
check, and the discipline that found them. The adaptation program that
produced this repository is closed by its own measurements (see
`VERDICT.md`; the nine results against it, and the five since against the fence
lane's own generality claims, are enumerated in
`experiments/results_against_thesis_2026-09-17.json`).

## Now

- **Addendum Z (preregistered 2026-09-23; launches after its anchor):**
  the +46.5 gate run again on this box with every prediction stored,
  under the Addendum B recipe unchanged and a protocol anchored before
  the first arm — the first one here anchored before its run (the anchor
  proves the bytes; the start time is the runner's clock, ours to
  report). The unchanged reader
  computes the verdict; four readings (gate word, agreement with the
  banked +46.5 within 0.05, the largest per-arm spread across the two
  environments, the over-cap exclusions) are applied by arithmetic and
  withheld until all six arms exist. Whichever way it moves, the
  primary-verifiable number replaces +46.5 on every page; if it is not
  GO, the headline is withdrawn. ~40–60 wall-hours on four cores, not
  the nine this repository had estimated (CORRECTIONS).
- **Upstream filings — owner's action.** Final issue text exists for
  `openai/evals` (two sites) and Braintrust `autoevals` (`JSONDiff`,
  executed repro); nothing is filed as of this date. This is the single
  item every simulated review round ranked first.
- **Addendum U (banked 2026-09-04):** the shipped parsers on every raw
  output this repository has banked — 1,950 outputs, four families.
  Strict parsing loses 100% of schema-only outputs on Qwen2.5 and
  Falcon3 and 0% on Granite and SmolLM2: the fence tax is
  family-dependent, stated at full size. U-ext (same day): instructor,
  smolagents and llama-index helpers — instructor's last-span rule hands
  back a piece of the receipt as the answer (post-hoc substance check,
  disclosed); smolagents harmless. Next: the Phi-3 cells into
  the corpus; the fourteen census packages still not attempted.
- **Addendum V (banked 2026-09-09):** the JSON-constrained decoder on
  five families' schema-only prompts: NO EFFECT in all five by the
  frozen thresholds (Falcon3 14 → 8, one short; its six recoveries are
  the one clean decoder effect), five regressions on Qwen 0.5B from a
  comparator confound (`model.generate`'s repetition-penalty default)
  that also touches Ladder II's SYSTEM rows, and every remaining invalid
  output a truncation at the 512-token cap. The drop-in claim is
  withdrawn.
- **`fencecheck template` (shipped 2026-09-16):** a third check, from a
  defect found in this repository's own banked work. A chat template
  that builds its own system message can put the current date in it,
  so outputs banked on different days were never produced from the
  same prompt and none of them can be reproduced later. Reads
  `tokenizer_config.json` or a bare `.jinja`, standard library only,
  exit 1 on a finding. One of the seven checkpoints this project has
  run is affected.
- **Anchors (2026-09-17):** every frozen protocol since Addendum S and
  the X/Y readings now carry an OpenTimestamps anchor
  (`docs/research/ANCHORS.md`). Stated plainly there: an anchor made
  after a run proves the bytes existed by that day, not that the
  protocol preceded its data; from Addendum Z on, a protocol is anchored
  before its first arm runs and the stamp is quoted in its freeze line.
- **Addendum Y (banked 2026-09-17):** the Phi-3 instability measured
  directly. In one unbroken process every pair reproduces 0-of-10 on
  immediate repeat, state dependence and the `generate` path — Phi-3 in
  bfloat16 included, and its constrained pass matches V's banked cell
  10/10 — so the prediction that Phi-3 would read state-dependent failed.
  What fired is the mechanism: the document that flaked in X's gate
  decodes differently at 4 threads and at 1 (reduction order at eight
  mantissa bits), while Qwen reproduces in bfloat16 too, so the dtype
  alone is not sufficient. Phi-3's 2026-09-03 `generate` cell still
  differs today on 4 of 10 for reasons the unrecorded environment cannot
  name. The 09-16 sentence "every Phi-3 number is not byte-reproducible"
  was too strong and is narrowed (CORRECTIONS). X stays withheld.
- **Addendum X (2026-09-16): WITHHELD by its own determinism gate, and
  the gate failure is the result.** X was built to separate the
  constraint from the decoding path and the generation defaults — the
  experiment V named and did not run. Before reading anything it
  re-decodes the arms it reuses and requires byte-identity. Three
  families pass 0-of-10 (Granite's gate is not applicable: its arm was
  re-run, not reused); `Phi-3-mini`, the one bfloat16 family, does
  not, so nothing is read about the decoder at all. Two failures: the
  `model.generate` comparator no longer reproduces the text banked on
  2026-09-03 (same two documents in both of today's runs), and the
  constrained arm differed on one document in one gate run and matched
  in the next. **Every Phi-3 number in Addenda S, T, U and V is
  therefore not byte-reproducible** — errata beside T and V, row in
  `CORRECTIONS.md`. It does settle one thing V left open: V attributed
  Phi-3's 32-of-50 between-arm divergence to bfloat16 numerics and
  called it untested; it is tested now, and the family does not
  reproduce against itself. Next: a successor addendum that measures
  the Phi-3 instability directly, under its own preregistration, before
  anything reads the four banked-and-unread families.
- **Addendum T (banked 2026-09-05):** the fence tax on four other
  families. Falcon3-1B replicates (92/100 schema-only fenced, 0/80
  k-shot); SmolLM2, Granite-3.1 and Phi-3 fence 0/100. Reading (c) in
  three of four: the tax is family-specific, not a property of small
  models as a class — 1 of 4 families in T, 2 of 5 counting Qwen2.5
  from Addendum S — and every document says so.
- **`tools/fencecheck.py`:** stdlib, one file; `scan` for fail-open
  parse sites, `score` for what a fence costs your saved outputs, and
  since Addendum W (2026-09-08) `score --scope any` for the
  prose-prefixed fence and the bare object in prose — measured at 3 of
  the 143 outputs the default scope rejects on this repository's corpus.
  Next: a `--baseline` mode that diffs two scored files.

## Next

- One external team running `fencecheck scan` in CI, with a published
  before/after — the falsifiable test the application names.
- ~~The JSON-constrained decoder as a drop-in for any HF causal LM~~ —
  withdrawn 2026-09-09 by Addendum V: on five families' schema-only
  cells it cleared no family's bar (Falcon3 14 → 8 missed REDUCES by
  one; every remaining invalid output is a truncation at the token
  cap) and its Qwen comparison carried a decoding-defaults confound.
  `tools/jsongreedy.py` stays published with that row as its measured
  limit.
- Other corpora for the fence tax beyond waybills and CORD.

## What is explicitly not on the roadmap

- Reopening the adaptation product claim without a preregistered rung
  that its own decompositions license.
- Any number in outbound copy that does not trace to an artifact.

<details>
<summary><b>History — the ARC Prize and adaptation roadmap as it stood on 2026-08-19</b> (click to expand)</summary>

# Iteration roadmap (updated 2026-08-19, post-v10)

## Where things stand
- kaggle-v7 scored 0.00 (submitted 00:13 UTC 08-09; scored 08-09): 40/240
  real predictions — first non-degenerate file, but of 167 attempted
  tasks, 98 were lost to a transformers cache-API bug in the pinned image
  (root-caused the same day = paper incident 6.8) and 52 hit the OOM
  ladder. 0/40 correct is consistent with the measured per-attempt hit
  rate; no new pipeline bug indicated
  (kaggle_v7_scored_2026-08-09.json).
- kaggle-v8 scored **1.67 public** on 08-10 — the first nonzero (row
  55392326; kaggle_v8_scored_2026-08-10.json): the cache-API fix held,
  150 real predictions across 137/240 tasks. At ~2.7% per attempt, the
  pipeline is proven end-to-end and solver quality is now the binding
  constraint: reaching the ~10% bar set for Sept 1 needs roughly a 4x
  hit-rate improvement — a solver-quality program, not a throughput
  program.
- kaggle-v9 scored 1.67 on 08-12 — exactly flat vs v8: recall bound
  widened 0.1->0.02 (+ candidate cap doubled), score unchanged;
  preregistered FLAT branch taken: the bound was not binding, further
  widening ruled out (kaggle_v9_scored_2026-08-12.json).
- kaggle-v10 scored 1.67 on 08-15 — second exactly-flat single-variable
  null: DFS time budget 60 -> 90 s/task, score unchanged. Budget levers
  are exhausted; candidate-generation quality is the binding constraint
  (a multi-week solver program); leaderboard climbing is formally
  deprioritized and GPU quota goes to the enterprise gates
  (kaggle_v10_scored_2026-08-15.json).
- Micro-tier own-weights run prestaged in kaggle/micro/ (~4h T4 on the
  free interactive quota — now the primary compute vehicle).
- Test suite: 459 green, pinned by tests/test_doc_counts_agree.py
  (83/83 was the 08-11 count; the intermediate figures in this line were
  stale five times before the count was pinned — see CORRECTIONS.md).

## Landed since the first draft
- DFS decoding with probability cutoff (v4d validated the code path).
- Full 8-element dihedral sweep + color-permutation TTT sets (expanded_sweep)
  decoupled from prediction frames (SolveConfig.ttt_augmentations).
- Example-shuffle TTT augmentation (deterministic per augmentation index).
- Per-GPU task sharding in the kernel (2x per-task time budget on the
  T4x2 environment — the 2026 track offers T4/P100 only; the earlier 4x
  figure assumed the 2025-vintage L4x4 note retracted in
  docs/research/KAGGLE_MECHANICS.md).

## Next algorithmic increments (ordered by expected score-per-effort)
Note (2026-08-19): v9/v10 closed the budget levers — the DFS recall bound
and time budget are preregistered nulls and further widening is ruled
out. What remains is candidate-generation quality; with leaderboard
climbing formally deprioritized (see above), the list below is the
ordering for that solver program, not an active submission plan.
1. **Act on the diagnostic**: if lp(true) is healthy, scale TTT (rank, augs,
   epochs) into the enlarged 2-GPU (T4x2) budget; if not, fix serialization first.
2. **LoRA rank 64-256 (rslora)** — champion ran r=256; measure T4/L4 step cost
   at rank 64/128/256 before committing the kernel budget.
3. **Batched DFS expansion** — the per-beam KV-cache forward is the current
   inference bottleneck; batching frontier expansions cuts DFS wall-clock.
4. **Unsloth + FlashAttention-2** for TTT speed, if it installs offline.
5. **TRM ensemble** — DEMOTED per docs/research/TRM_PLAN.md: NVARC's own
   ablations show ~+1 pt for their 2B and ~zero for their 4B (their
   in-competition ensemble scored below LLM-only). Worth +0.5-2 pts at
   best for ~21h work + 2h/run of scoring budget; revisit only after
   items 1-4 are exhausted. De-risk smoke test (~$1) is cheap if wanted.

## Submission cadence
One competition submission per day. Identity verification complete
(2026-08-08); first nonzero on the board (v8, 08-10). The scored file
comes from the kernel's OWN interactive run — keep the accelerator
pinned (machine_shape). Never submit blind: validate each config on the
public-eval slice first, submit only improvements. Log every config +
score in experiments/.

