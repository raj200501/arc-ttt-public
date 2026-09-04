# Roadmap (updated 2026-09-03)

The company is the eval-integrity work: the fence tax, the shipped
check, and the discipline that found them. The adaptation program that
produced this repository is closed by its own measurements (see
`VERDICT.md`; the nine results against it are enumerated in
`experiments/results_against_thesis_2026-09-03.json`).

## Now

- **Upstream filings — owner's action.** Final issue text exists for
  `openai/evals` (two sites) and Braintrust `autoevals` (`JSONDiff`,
  executed repro); nothing is filed as of this date. This is the single
  item every simulated review round ranked first.
- **Addendum U (banked 2026-09-04):** the shipped parsers on every raw
  output this repository has banked — 1,950 outputs, four families.
  Strict parsing loses 100% of schema-only outputs on Qwen2.5 and
  Falcon3 and 0% on Granite and SmolLM2: the fence tax is
  family-dependent, stated at full size. Next: the Phi-3 cells into
  the corpus; more parsers behind the one-function interface.
- **Addendum T (in flight):** the fence tax on four other model
  families — SmolLM2, Granite, Phi-3, Falcon3 — same cells as
  Addendum S, thresholds frozen in
  `docs/research/ADDENDUM_T_PROTOCOL.md` before any arm ran.
- **`tools/fencecheck.py`:** stdlib, one file; `scan` for fail-open
  parse sites, `score` for what a fence costs your saved outputs. Next:
  a `--baseline` mode that diffs two scored files, and the
  prose-prefixed-fence scope documented in Addendum R's erratum.

## Next

- One external team running `fencecheck scan` in CI, with a published
  before/after — the falsifiable test the application names.
- The JSON-constrained decoder (`src/arcttt/constrained_json.py`) as a
  drop-in for any HF causal LM: it removed every invalid output on both
  arms in Ladder II and is schema-blind by design.
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
- Test suite: 396 green, pinned by tests/test_doc_counts_agree.py
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

