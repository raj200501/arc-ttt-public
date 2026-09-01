# A Clean-Room Reproduction Harness for Test-Time Training on ARC-AGI-2: What Transfers, What Breaks, and What It Costs

**Draft — sections 1–3 and 5–7 in prose; the section 4 ablation grid has
5 of 8 rows filled from the experiment registry. Every number in this
document either comes from a
machine-readable artifact in `experiments/` or is explicitly attributed to the
NVARC team's published results. TODO markers flag claims we cannot yet
support with an artifact.**

## 1. Introduction

The ARC Prize 2025 competition on ARC-AGI-2 was won by NVARC (Sorokin and
Puget), whose published solution combined large-scale synthetic puzzle
generation, supervised fine-tuning of a 4B-parameter language model,
per-task test-time training (TTT) with LoRA adapters, a constrained
depth-first search over output tokens, and augmentation-based candidate
voting. They report 27.64% on the public leaderboard and 24.03% on the final
private leaderboard — their numbers, from their paper and the ARC Prize
results announcement, not ours. The recipe is public: an 11-page paper, a
repository of training configurations and data-generation prompts, and
Kaggle notebooks.

Public, however, is not the same as reproducible. The NVARC repository
carries no license, so its code cannot legally be reused; the paper, like
most competition writeups, describes the recipe at the level of
hyperparameters and leaves the load-bearing engineering — loss masking
conventions, cache management inside the tree search, tokenizer edge cases,
the exact serialization down to the newline — to the reader. Anyone who
wants to build on the winning recipe for ARC Prize 2026 must therefore
re-derive those details, and will re-hit the same class of silent bugs we
document in section 3, most of which produce *plausible wrong behavior*
rather than crashes.

This paper's thesis is that a fully tested clean-room reproduction, with
per-component ablations and honest cost accounting, is more useful to the
field than another recipe writeup. Recipe worship — copying a winner's
hyperparameters without knowing which components carry the accuracy — leaves
the community unable to say what actually matters: does the raw serialization
format matter, or would the standard chat template do? How much does a
*correct* depth-first search buy over the near-greedy search that a subtle
cache bug silently produces? How much of the score is TTT versus search
versus voting? These are ablation questions, and they require an
implementation in which each component can be switched independently and in
which correctness is established by tests rather than by leaderboard
proximity.

We contribute:

- **A clean-room reimplementation** of the NVARC inference recipe
  (serialization, per-task LoRA TTT, constrained DFS decoding, augmentation
  voting), written from the team's public descriptions and never from their
  code (section 2), released under MIT.
- **A test harness** of 344 offline tests that runs without a GPU, network
  access, or model downloads, including an oracle test that verifies the
  tree search token-exactly against a cache-free reference implementation —
  a test that caught a KV-cache aliasing bug which would otherwise have
  silently degraded the search (section 3).
- **A decomposed ablation grid** in which every cell is a machine-readable
  artifact in the public repository, including negative and in-progress
  results (section 4), plus cost accounting on commodity GPUs (section 5).
- **Bug postmortems** for every failure mode we hit that a reader
  reproducing this recipe would otherwise re-hit (sections 3 and 6).

On claims: this work is an active entry in the ARC Prize 2026 ARC-AGI-2
Kaggle track. Scored history as of 2026-08-17: v6 0.00 (wrong-architecture
GPU; the scored file was the safety fallback — incident 6.7), v7 0.00 at
40/240-task coverage (library API drift between dev and scoring images —
incident 6.8), then 1.67 public three times running (v8 on 08-10 at
~150-prediction coverage, v9 on 08-12 and v10 on 08-15 — both
preregistered nulls, on the recall-bound and time-budget levers
respectively). The stable 1.67 across three coverage/lever configurations
cleanly isolates candidate quality (~2.7% per-attempt hit rate) as the
binding constraint.
Local results so far are small-scale pipeline validations (a
handful of evaluation tasks on a single T4), reported in full in section 4.
We state this plainly because the paper's value does not depend on
leaderboard rank, and because a reproduction paper that inflates its own
numbers has defeated its purpose.

## 2. The recipe, decomposed

The NVARC pipeline has two halves. The offline half — synthetic data
generation with LLM-written grid programs and supervised fine-tuning of
Qwen3-4B on roughly 3.3M serialized puzzles (the team reports 4 nodes of
8×H100 for 27 hours for the SFT stage alone) — is out of scope for this
reproduction: we could not justify that compute for a reproduction study,
and the team published the resulting fine-tuned 4B checkpoint as a public
Kaggle artifact, which competition rules permit all entrants to use. We
consume that public checkpoint and reproduce the *inference-time* half,
which is where test-time training lives and which fits in the competition's
fixed budget of 12 hours per submission for 240 hidden tasks — on this
track's T4/P100-class notebook accelerators, which we pin to T4 via kernel
metadata (`machine_shape`; section 6.7 — the L4x4 figure we had carried
from 2025-vintage notes turned out to be stale for this track). This
section describes each component as we implemented it, and flags what the
public writeup under-specifies.

**Clean-room methods statement.** The NVARC repository contains no LICENSE
file, so its code is legally unusable regardless of intent. Our
implementation was written from the team's paper and from prose descriptions
of their notebooks; hyperparameter values (such as the LoRA configuration in
section 2.2) were read from their published configs — no code was. Nothing
was copied, ported, or paraphrased from their repository. Where our implementation choices had to fill gaps in the
writeup, we say so explicitly below, because those gaps are exactly the
places where a "faithful" reproduction can silently diverge.

### 2.1 Serialization format

ARC grids are rectangular arrays of colors 0–9, at most 30×30. The recipe
serializes a grid as one line of digit characters per row, and a task as a
chat transcript: each demonstration pair becomes a user turn (input grid)
followed by an assistant turn (output grid), and the test input is a final
user turn that the model completes. The NVARC team additionally cut the
tokenizer down to 16 tokens, so that digits, the newline, and the chat
control tokens are the entire vocabulary.

What the writeup under-specifies is the exact string. Our golden-format
tests pin the champion framing as: raw `<|im_start|>role\ncontent<|im_end|>`
blocks, *no system turn*, and *no newlines between turns* — a format that
differs from what `apply_chat_template` produces for the same messages
(which inserts a system turn and inter-turn newlines, and for thinking
variants may inject reasoning scaffolding). This distinction is invisible in
a paper-level description and material in practice: a checkpoint fine-tuned
on one framing is evaluated off-distribution under the other. Our harness
implements both (`raw_qwen_format` on/off) precisely so the difference is an
ablation rather than an assumption, and the raw format is verified against a
golden string in the test suite.

### 2.2 Per-task LoRA test-time training

For each task independently, the recipe fine-tunes a fresh low-rank adapter
on the task's own demonstration pairs before predicting the test output. The
training corpus is built leave-one-out: for each of the k demonstrations,
the other k−1 serve as chat context and the held-out pair supplies the
supervised completion, with the loss masked to the final assistant turn.
This corpus is then multiplied by augmentations (section 2.4), and the
demonstration order within the context can be shuffled per augmented copy so
the adapter cannot overfit a fixed presentation order. The NVARC
configuration, per their repository's configs and notebook, is LoRA r=256,
alpha=32 with rank-stabilized scaling (rslora), learning rate 5e-5, one
epoch, batch size 1, and 16 augmented copies of the demonstrations.

The under-specified parts are the corpus conventions: whether the loss
covers every assistant turn or only the held-out completion, how
demonstration order is handled, and what happens for tasks with very few
demonstrations. We chose final-turn-only supervision and deterministic
seeded shuffling; these choices are encoded in `arcttt/serialize.py` and
unit-tested, so a reader who believes a different convention was intended
can flip it and measure the difference rather than argue about it.

### 2.3 Constrained depth-first decoding

Because a serialized output grid can only contain digit tokens, a row
separator, and a stop token, the model's next-token distribution can be
restricted to that ~13-token subset, and *every* completion whose cumulative
negative log-probability stays under a cutoff can be enumerated exactly by
depth-first search. NVARC report using a probability cutoff of 0.2 (a
cumulative-NLL bound of −log 0.2) in their ARC-AGI-1 harness. The search
turns one adapted model into a scored set of candidate grids per prompt,
rather than a single sampled guess, and the score attached to each candidate
is its exact sequence NLL — which the selection stage can reuse.

This is the component where the writeup's silence is most dangerous, because
the natural implementation over a HuggingFace-style KV cache is wrong in a
way that still produces grids: caches mutate in place on every forward pass,
so per-branch cache references alias one another and corrupt sibling
branches, quietly collapsing the enumeration toward a near-greedy search.
Section 3 describes the bug and the oracle test that is, in our experience,
the only reliable way to catch it. Our implementation keeps one shared cache
with a crop-on-backtrack invariant, bounds the search by candidate count and
wall-clock deadline, and optionally guarantees the greedy completion is in
the candidate pool (the NLL cutoff can exclude even the argmax path — a
corner case the paper does not mention).

### 2.4 Augmentation voting and selection

ARC tasks are invariant under the dihedral group D4 (rotations and
reflections) and under permutations of the color palette. The recipe
exploits this twice. First, for TTT data: the adaptation corpus is expanded
across augmented copies of the task. Second, for prediction and selection:
the model predicts in several augmented frames, each prediction is mapped
back through the *exact inverse* of its augmentation, and the pooled
candidates are ranked. NVARC's selection score, per their paper, combines
how many times a candidate was found by the search with the geometric mean
of its probability re-scored under a fixed set of 8 augmentations — with the
explicitly stated requirement that every candidate is re-scored under the
*identical* augmentation set, so scores are comparable. Our implementation
ranks by found-count plus exp(mean log-probability); since the probability
term lies in (0, 1], it breaks ties between equal counts without ever
outweighing one additional find. Whether that tie-breaking shape matches the
original exactly is under-specified in the writeup; it is one more knob our
harness exposes rather than hides. The top two candidates become the two
scored attempts the competition allows.

Every augmentation in the harness carries an exact inverse, and round-trip
identity (augment, then invert, recovers the input) is unit-tested for the
full dihedral-times-palette family — because a single wrong inverse anywhere
in this stage poisons the vote silently.

## 3. Harness

The harness (`src/arcttt/`, MIT-licensed) is small — roughly 1,650 lines
across eight core modules — and is designed around one principle: every
property the pipeline depends on is either enforced by construction or
pinned by an offline test. The test suite (344 tests) runs on CPU with tiny in-test
models — a few-layer Qwen2-architecture model and a hand-built 13-token
tokenizer constructed inside the test — so correctness never depends on
network access, model downloads, or GPU availability. Scale comes from
swapping the base model and device, not from changing code paths, which is
what makes the local tests evidence about the Kaggle run.

### 3.1 Invariants

- **Grids are immutable tuples** of tuples of ints, hashable by
  construction, so vote counting and candidate deduplication are dictionary
  operations that cannot be corrupted by aliasing or in-place mutation.
- **Every augmentation carries an exact inverse**, and round-trip identity
  is unit-tested across the dihedral sweep and seeded palette permutations.
  Palette generation is seed-deterministic, so an augmentation set is
  reproducible from its seed.
- **The champion serialization is golden-string tested**: the raw-format
  encoder's output for a fixed task is compared against a literal expected
  string, so a drive-by "cleanup" that adds a newline between turns fails CI
  instead of silently changing the evaluation distribution.
- **Loaders fail closed.** Task JSON is validated against the ARC schema —
  ragged rows, out-of-range colors, oversized grids, and missing keys all
  raise instead of degrading into a half-parsed task. The same applies to
  decoding: model output that does not parse as a rectangular digit grid is
  dropped, never coerced.
- **The kernel emits a valid fallback submission first**, before any model
  work, and overwrites entries per task as real predictions arrive, so a
  mid-run failure on Kaggle still yields a scoreable file.

### 3.2 Pure-torch LoRA, or: the scoring image is not your dev image

Our third Kaggle submission (kernel v3) produced a valid 240-task submission
in which every task had fallen back: `peft`, the standard adapter library,
is not installed on the offline competition scoring image, and every
per-task adaptation raised ImportError (registry: `kaggle-v3`). The fix was
to remove the dependency: `arcttt/lora.py` is a ~115-line pure-torch LoRA —
inject rank-decomposed trainable deltas into a frozen model's linear layers,
with rslora scaling, and remove them cleanly between tasks. Injection,
identity-at-initialization, learning, removal, and re-injection are each
unit-tested. The general lesson: an offline scoring rerun is a different
machine from the interactive session you developed in, and every import is a
claim about that machine which should be tested, not assumed.

### 3.3 The KV-cache aliasing bug, and the oracle test that caught it

The constrained DFS is the harness component most worth a postmortem. Our
first GPU-validated version passed its unit tests: it returned valid grids,
scores were finite and sorted, cutoffs and candidate caps were respected. It
was still wrong. HuggingFace KV caches mutate in place on every forward
pass; the search had stored per-branch references to the cache, and since
those references alias the same underlying tensors, expanding one branch
corrupted the recorded state of its siblings. The failure mode is insidious
precisely because the search still produces plausible, high-probability
grids — it just stops being the exact enumeration it claims to be and decays
toward greedy decoding with noise. No property test on the *outputs alone*
flagged it.

What caught it was an oracle test: a brute-force reference enumerator that
uses full forward passes and no KV cache at all, compared token-exactly
against the fast implementation — same candidate set, same scores to 1e-4 —
on a tiny model with a cutoff loose enough to force genuine branching. The
fixed implementation keeps a single shared cache and restores state on
backtrack by cropping it to the parent's length, maintaining the invariant
that the cache always holds exactly prompt-plus-current-path. The bug was
found roughly 25 minutes into our fifth Kaggle run, which we superseded
(registry: `kaggle-v5`; the fix is exercised by the cache-free oracle
test in `tests/test_decode.py` — the full commit history lives in the
private development repo). Two ablation consequences
follow: first, the corrupted search is itself a meaningful baseline —
"how much does exact search matter over near-greedy?" is a measurable
question (section 4.3); second, we now regard a cache-free oracle test as
the minimum bar for trusting any tree search built over mutating caches.

### 3.4 Build the bundle, never sync it

Kaggle script kernels are a single file. Our fourth submission used a
hand-synchronized bundle: package code pasted into the kernel file, with
intra-package imports supposedly removed. Two indented `from arcttt...`
imports inside method bodies survived the sync. Locally the package is
installed, so the imports resolve and every test stays green; on Kaggle
there is no `arcttt` package, so all 240 tasks died with
ModuleNotFoundError and the run produced an all-fallback submission
(registry: `kaggle-v4`; held, not submitted). The root cause was the
process, not the typo: hand-syncing creates a second copy of the code whose
correctness is tested nowhere. The replacement is a bundle builder
(`kaggle/build_bundle.py`) that concatenates the package modules in
dependency order, strips intra-package imports including parenthesized
multi-line spans, then parses, compiles, and greps the result for leftovers,
refusing to emit a bundle that could fail on the scoring VM the way v4 did.
Drift now fails at build time, on our machine.

### 3.5 Gradient checkpointing at the edges of the TTT loop

Gradient checkpointing interacts badly with both ends of the per-task loop.
With the base model frozen and only LoRA parameters trainable, the
checkpointed segments see no input requiring grad, and backward crashes;
the fix is `enable_input_require_grads()` before enabling checkpointing.
Symmetrically, checkpointing must be disabled
again before generation, which uses the KV cache; the adapt/eval transition
in the predictor handles both edges. Neither interaction appears in any
recipe writeup we know of, and both cost us GPU-hours.

### 3.6 Operational hardening for the scoring VM

The remaining harness layer exists because the Kaggle scoring environment is
remote, offline, and different from the development environment in
under-documented ways. Competition data and attached models mount under
paths that differ between interactive and scoring sessions, so the kernel
discovers both recursively (any directory with `config.json` plus
safetensors is a model candidate; challenge files are found by glob) and
prints what it resolved — our first two kernel versions died on a hardcoded
path (registry: `kaggle-v1/v2`). The kernel shards tasks across one worker
process per visible GPU, smallest tasks first; each worker atomically
checkpoints its partial submission after every task, and the parent merges
whatever exists at the deadline, bounding worker joins so a hung straggler
cannot forfeit the submission file. Crucially — and contrary to the
assumption we carried from 2025-vintage notes — the submission is scored
from the kernel version's *own* run output; there is no separate scoring
rerun on different hardware (section 6.7). The run environment must
therefore be pinned explicitly (`machine_shape` in the kernel metadata):
an unpinned session can land on a pre-Ampere GPU without usable bf16
kernels, which is exactly what our v6 run hit, with per-task accelerator
errors in about a millisecond each (registry: `kaggle-v6`). Workers still
detect bf16 support and degrade to fp16 inference with TTT disabled (fp16
training without a loss scaler is unsafe) rather than dying, but
degradation is a floor, not a strategy — the pin is the fix. Finally, cut tokenizers without an UNK token *raise* on
out-of-vocabulary probe strings rather than returning ids, so the
grid-vocabulary discovery treats encoding failures as "not representable"
and fails loudly only when a required token is genuinely missing.

## 4. Ablations (grid, partially filled)

Design: each ablation below toggles exactly one component of the section 2
recipe, holding the rest fixed; every result cell links a machine-readable
artifact in `experiments/`. Rows without an artifact are planned or running,
and are marked TODO(artifact) — no number appears here that does not exist in an
artifact. Current artifacts are small-n pipeline validations (1–15 public
evaluation tasks on a single T4, lite configurations); the table will be
refreshed from the same registry as larger runs land.

| # | Ablation | Arm A | Arm B | Tasks / pairs | Result | Artifact(s) |
|---|---|---|---|---|---|---|
| 4.1 | Serialization: chat template vs raw champion format | chat template | raw `<\|im_start\|>` format | TODO(artifact) | TODO(artifact): same-checkpoint paired run pending | TODO(artifact) (planned: `t4_format_ablation_*.json`) |
| 4.2 | Decoding: sampling vs constrained DFS | 1 sample/aug | DFS, cutoff 0.1 | 8 tasks / 10 pairs | 1/10 vs 0/10 (champion 4B, lite config; caution: the DFS arm predates the KV-cache crop-on-backtrack fix — see 4.3 — so this compares sampling against the *corrupted* search; see registry) | `t4_champ_eval_2026-08-08.json`, `t4_champ_dfs_eval_2026-08-08.json` |
| 4.3 | Search correctness: corrupted (near-greedy) vs exact DFS | pre-fix DFS (corrupted, near-greedy; the fix is the crop-on-backtrack change verified by the oracle test in `tests/test_decode.py`) | fixed DFS, same config | 8 tasks / 10 pairs (fixed arm scored 9 — faa9f03d OOMed, unscored) | 0/10 vs 1/9 scored pairs — +1 task from search correctness alone (greedy-include off in both arms) | `t4_champ_dfs_eval_2026-08-08.json` vs `t4_dfs_fixed_repeat_2026-08-08.json` |
| 4.4 | TTT on/off, teacher-forced lp(true) | no TTT | TTT, 8 augs | 8 tasks; 5 comparable / 7 test pairs (3 tasks OOMed in the TTT arm) | lp(true) sharpened on 3/5 comparable tasks (5/7 test pairs; best −0.120→−0.029), degraded on two (e8686506 −0.112→−0.156, dbff022c −0.073→−0.083); solve count unchanged (1) — recall-limited | `t4_champ_diag_2026-08-08.json` |
| 4.5 | Augmentation count sweep | dihedral only | + color perms / + shuffle | TODO(artifact) | TODO(artifact) | TODO(artifact) (planned: `aug_sweep_*.json`) |
| 4.6 | Base model scale | 0.5B (+400-task SFT) | champion 4B | 15 tasks / 22 pairs vs 8 tasks / 10 pairs | 0/22 (0.5B, honest zero) vs 1/10 (4B); non-paired task sets — indicative only | `t4_sft_eval_2026-08-08.json`, `t4_smoke_2026-08-08.json`, `t4_champ_eval_2026-08-08.json` |
| 4.7 | LoRA rank feasibility sweep (16-aug TTT, single task) | r=16 / 64 / 128 | r=256 (champion rank) | 1 task (7b5033c1) | 630.0 / 484.8 / 577.2 / 585.0 s; peak 12.78 / 10.88 / 12.23 / 12.69 GB — all four ranks fit a 16 GB T4; r=64 is both fastest and lightest; basis for running r=256 in kernel v7 | `t4_rank_sweep_2026-08-08.json` |
| 4.8 | Kernel coverage at fixed 12 h budget: generic cache iteration + level-0 ladder start (v7) vs probed cache API + use_cache=False rescoring + 4-frame search + pacing guard (v8) | v7: 167/240 attempted, 40 real (98 ValueError; 52/19/1 OOM ladder) | v8: 137/240 tasks with a real attempt (150 real predictions) | 240 hidden tasks, 2xT4, 12 h | v8: 137/240 attempted (150 real predictions, 522 min; below the 200-240 pre-run projection); v7's 98-task _gather_cache ValueError eliminated | kaggle_v7_postmortem_2026-08-08.json; kaggle_v8_run_2026-08-09.json; kaggle_v8_interactive_run.log; planned: kaggle_v8_postmortem json |

Context rows from the registry (`experiments/README.md`): the initial
pipeline smoke test (`t4_smoke_2026-08-08.json`) measured 77 s/task mean at
the lite 0.5B configuration, projecting 5.14 h for 240 tasks on a single T4;
the champion-4B DFS validation ran 72–282 s/task on a T4 with no crashes
(the fixed-decoder repeat, `t4_dfs_fixed_repeat_2026-08-08.json`, ran
116–365 s/task with one OOM, faa9f03d).
The teacher-forced diagnostic (`t4_champ_diag_2026-08-08.json`) measured
format fit directly: with no TTT at all, mean per-token log-probability of
the true solutions ranged −0.063 to −0.120 across the eight-task slice,
and a greedy generation emitted a syntactically perfect grid — the
checkpoint is squarely on-distribution under our serialization. One epoch
of 8-augmentation TTT sharpened lp(true) on three of five comparable tasks
— five of seven test pairs — (best: −0.120 → −0.029), degraded it on two
(e8686506 −0.112 → −0.156, dbff022c −0.073 → −0.083), and left the truth
ranked above every candidate the search actually found on two tasks — the
recall gap that motivates the greedy-include guarantee. Three of eight
tasks OOMed a 16 GB T4 in the TTT phase (458–965 s/task otherwise), which
set the kernel's memory ladder and its 8-augmentation budget.

## 5. Cost curves

Cost claims in this section follow two rules: every dollars-per-task figure
is stated with its configuration and its artifact, and no comparison against
frontier-API pricing is made without the narrow-repeated-task scope caveat.
All timings come from `experiments/*.json`. Dollar figures use the measured
T4 credit burn — 0.5–1.0 credit-hour per wall-clock hour, measured from
provider balance deltas across runs on Lightning, our validation
environment (the widest observed delta was 1.98 credits over a 2.0 h
session) — and the table shows the full measured range; quote the 1.0 end
when a single number is needed. The burn rate is measured but not yet
captured in a registry artifact (TODO(artifact): credit-burn balance
deltas).

### 5.1 Measured per-task adaptation cost

On a single T4, on small slices of the ARC-AGI-2 public evaluation set:

| Config | s/task (T4) | ≈$/task adapted | Artifact |
|---|---|---|---|
| 0.5B, 4 dihedral augs, sampled decoding | 77 (mean) | $0.011–0.021 | `t4_smoke_2026-08-08.json` |
| Champion 4B, 4 augs, r=16 TTT + DFS | 72–282 | $0.010–0.078 | `t4_champ_dfs_eval_2026-08-08.json` |
| Champion 4B, 4 augs, r=16 TTT + fixed DFS (crop-on-backtrack) | 116–365 | $0.016–0.101 | `t4_dfs_fixed_repeat_2026-08-08.json` |
| Champion 4B, 8-aug TTT, DFS cutoff 0.05 | 458–965 | $0.064–0.27 | `t4_champ_diag_2026-08-08.json` |
| Champion 4B, r=256 TTT, batched DFS (60 s budget), v8 kernel (2xT4, Kaggle) | 522 min x 2 GPUs / 137 attempted ≈ 457 GPU-s/task (v7 comparator: 649 min x 2 GPUs / 167 attempted ≈ 466 GPU-s/task; both above the <= 330 per-task target) | n/a — competition-provided compute | kaggle_v8_run_2026-08-09.json; kaggle_v8_interactive_run.log |

All ≈$/task figures additionally assume $1.00/credit (the provider's list
price for on-demand credits).

Three structural observations follow. First, cost scales with the knobs the
ablation grid varies — augmentation count, LoRA rank, and search cutoff —
so section 4's accuracy axes and this section's cost axes are the same
axes, which is what makes a $/point curve meaningful once scores exist.
Second, the single-GPU T4 projection from the smoke run (5.14 h for 240
tasks at the lite 0.5B configuration, `t4_smoke_2026-08-08.json`) shows why
the competition's 12-hour budget forces either a small configuration or
parallelism: our kernel shards tasks across every visible GPU, dividing
wall-clock by the device count (section 3.6). Third, for calibration, the
NVARC team's published figure for their winning configuration is roughly
$0.20/task — their claim, from NVIDIA's coverage of their solution, on
different hardware and a much heavier configuration; we do not treat it as
comparable to our lite-config numbers, only as evidence that per-task
adaptation cost sits in the cents-to-dimes range across implementations.

### 5.2 Dollars per accuracy point: not yet computable, by our own rules

The quantity this section exists for — dollars per accuracy point, per
component — cannot yet be honestly reported: our scored submissions to date
are 0.00, 0.00, then 1.67 public three times running (v8–v10; sections 1,
6.7, 7.1) — a single flat aggregate that does not decompose into
per-component rates — and local scored slices are too small to support a
rate (1/9 scored pairs at best, section 4.3). Rather than back into a
$/point number from a projection, we publish the cost side of the curve
now, wired to the same artifacts as section 4, and will populate the
$/point column from scored runs as they land. TODO(artifact): fill in
$/point per ablation arm once scored runs differ across arms (v8–v10 are
flat at 1.67, so per-arm attribution is not yet identifiable).

### 5.3 Why per-task cost is the number that matters

Adaptation is a one-time per-task cost. Once the adapter is trained,
inference is a single greedy pass on a 4B model — a few seconds of
commodity GPU time (well under a cent per instance) — while a frontier-API
call with a few-shot prompt for a comparable structured task costs cents
per instance, every instance. For a task family seen repeatedly, the
adapted-small-model curve is flat after the first instance and the API
curve is linear. The scope caveat, stated as prominently as the claim:
this arithmetic says nothing about arbitrary tasks. It applies to narrow,
repeated tasks with demonstrations, where per-task adaptation can close
the quality gap — and whether it does close the gap is precisely the
empirical question the competition leaderboard scores. We are not yet
entitled to the quality half of that claim (section 6.7); the cost half is
what this section documents.

## 6. Negative results and incidents

This section is the incident log a reader reproducing this recipe should
tape to their monitor. Sections 3.2–3.6 give the deep engineering
treatments; entries here are compressed to symptom, root cause, and rule,
except for the newest incidents (6.7, 6.8), which are reported in full.
Entry 6.9 is not an incident at all but a preregistered negative result
about the method, and is reported in full for the same reason.

**6.1 Kaggle mount paths (kernels v1/v2).** Competition data and attached
models mount under paths that differ between environments. Both runs died
on a hardcoded glob before doing any model work. Rule: discover inputs
recursively, print every resolved path, and treat path assumptions as
untested code.

**6.2 peft absent on the scoring image (kernel v3).** A valid 240-task
submission in which every task fell back: ImportError per task, because the
offline competition image does not ship `peft`. Rule: every import is a
claim about the target machine; our fix was a pure-torch LoRA with no
dependency to miss (section 3.2).

**6.3 UNK-less cut tokenizers raise on out-of-vocabulary probes.**
Vocabulary discovery for the constrained search probes the tokenizer with
candidate strings; a cut-down tokenizer without an UNK token raises on
unrepresentable input instead of returning ids. Rule: treat encoding
failure as "not representable" and fail loudly only when a genuinely
required token is missing (section 3.6).

**6.4 Gradient checkpointing × frozen base, × cached generation.** With
only LoRA parameters trainable, checkpointed segments see no input
requiring grad and backward crashes — `enable_input_require_grads()` is
mandatory; and checkpointing must be disabled again before cache-using
generation. Both edges are handled in the adapt/eval transition (section 3.5).

**6.5 Hand-synced bundles drift (kernel v4).** Two indented intra-package
imports survived a manual sync into the single-file kernel; local runs
stayed green (the package exists locally), and all 240 tasks on Kaggle died
with ModuleNotFoundError. The run was held, not submitted. Rule: build
artifacts from source with a builder that fails on drift; never maintain a
second copy of code by hand (section 3.4).

**6.6 KV caches mutate in place; tree search over them aliases (kernel
v5).** Per-branch cache references alias the same tensors, so expanding one
branch corrupts its siblings and the "exact" enumeration silently decays
toward greedy-with-noise while still emitting plausible grids. No test on
outputs alone caught it; a token-exact oracle comparison against cache-free
full forwards did (fix: crop-on-backtrack, verified by the oracle test in
`tests/test_decode.py`; section 3.3).
Rule: a tree search over mutating caches is untrusted until it matches a
cache-free reference.

**6.7 The scoring environment is the interactive environment; pin your
accelerator (kernel v6).** Our v6 submission scored 0.00 (leaderboard
submission 55338854; registry: `kaggle-v6`). The kernel's interactive run
had landed on a default-resolved non-bf16 GPU (P100-class), where bf16
compute hard-fails: every task raised an accelerator error in about a
millisecond (observed in the run log; measured but not yet artifacted —
TODO(artifact)), and the run produced its valid all-fallback submission file
exactly as designed. We believed this was survivable, because we believed
scoring re-runs the kernel on the competition's upgraded hardware. It does
not. The score was resolved from the submission.json produced by the
kernel version's *own* interactive run — resolution took ~30 minutes
(measured but not yet artifacted — TODO(artifact)), far
too fast for any rerun — and the hardware notes we had carried from the
2025 competition (an L4x4 scoring VM) turned out to be stale for this
track, which offers T4/P100 notebook accelerators. Two lessons, both now
enforced: first, the accelerator is pinned in kernel metadata
(`machine_shape`) to the same GPU class as every validation run, so the
environment that produces the scored file is the environment we tested;
second, graceful degradation can be a trap — the fallback path worked
perfectly and thereby converted a hardware mismatch into a silent zero,
where a crash would have been caught before submission. A fallback
submission should be a safety net for partial failure, paired with a loud
preflight check that refuses to run the full pipeline on hardware that
cannot execute it. (Section 3.6 has been reconciled with this incident.)

**6.8 Library API drift between dev and scoring images (kernel v7).** Our
first non-degenerate submission file carried 40 real predictions out of 240
tasks — but not, as the run-day report first recorded, because of
throughput alone. Log analysis (registry: `v7-postmortem`) shows 98 of the
167 attempted tasks died with `ValueError: too many values to unpack`
inside the batched search's cache compaction. The helper iterated the
HuggingFace KV cache generically (`for key, value in cache`), a pattern
that had worked under our dev pin; the scoring
image's newer transformers removed the legacy-format bridge, generic
iteration yielded raw 4-D tensors, and tuple-unpacking "iterated" a batch
dimension instead of (key, value) pairs. The failure was invisible to
local validation because the dev pin (transformers 4.57.6) retains the
legacy bridge, so local validation never exercised the new
semantics — the same *shape* as 6.2's missing import, one
level deeper: not "is the library there," but "does the library still
honor the contract your code was written against." The remaining 29
attempted-but-fallback tasks are accounted for by memory pressure: the
OOM ladder fired on 52 tasks (19 escalated to level 1, and 1 was lost
outright at level 2), per `kaggle_v7_postmortem_2026-08-08.json` (18 of
the 52 OOM tasks also hit the ValueError; five OOM tasks recovered to
real predictions at a higher ladder level). Fix: the cache helpers
now probe for each known API surface explicitly (layered, list-based,
legacy tuples) and never iterate the cache object, with a regression test
against a mock of the newer API that reproduces the exact failure. Rule:
version-sensitive third-party contracts get an explicit capability probe
and a mock-based regression test for every API generation you cannot
install.

Common shape across these incidents: none were exotic bugs. Each was an
untested assumption about an environment or an API contract, and each
produced valid-looking output. The harness's response in every case was
the same — convert the assumption into either a construction-time
guarantee or a loud failure.

**6.9 The preregistered adaptation gate, and its failure at two scales.**
Sections 6.1–6.8 are engineering incidents: assumptions about
environments that produced valid-looking output. This one is different in
kind. It is a negative result about the method itself, obtained the only
way a negative result is worth anything — by writing the decision rule
down before running the experiment.

We preregistered a gate (ENTERPRISE_EVAL_SPEC Addendum A, frozen
2026-08-08): on CORD-v2 receipt parsing, per-request LoRA adaptation must
beat an in-context k-shot baseline by at least +5 micro-F1, as a mean
paired delta at k=10 over seeds {1,2,3}, with the adapted and prompted
arms sharing a split, a scorer, and an environment. All three rungs
have now completed it. All fail:

| rung | seed 1 | seed 2 | seed 3 | mean | gate |
|---|---|---|---|---|---|
| 0.5B | −1.2 | −15.5 | −5.1 | **−7.3** | fail |
| 1.5B | +1.1 | −19.8 | −16.0 | **−11.5** | fail |
| 4B | +3.3 | −8.5 | −8.2 | **−4.5** | fail |

Model scale was the lever we preregistered to rescue the 0.5B result, on
the strength of two indirect signals: a teacher-forced log-probability
probe that showed 1-epoch TTT sharpening lp(true) on 3 of 5 ARC tasks at
4B, and a published 2B→4B ablation of +5–7 points on ARC under the same
recipe. Both pointed up. The direct end-task measurement points down.
When an indirect proxy and a direct test of the same claim disagree, the
direct test is the result; the proxies become an object lesson in why the
direct test was preregistered.

Three qualifications, all of which cut against overstating the negative.
First, neither failure is significant on a distribution-free basis: the
sign test over pooled receipts gives 27W/32L (p=0.60) at 0.5B and
17W/29L (p=0.10) at 4B. Second, the gate is underpowered against its own
threshold — the minimum detectable effect at 80% power is 13.1 F1 at
0.5B and 11.3 F1 at 4B, against a 5 F1 bar. Third, that underpowering is
structural rather than incidental: CORD's validation split is 100
receipts, per-seed evaluation slices overlap, and the observed
per-receipt spread implies roughly 115 distinct paired receipts are
needed to resolve a 5-point effect — more than the split can supply at
any eval_n. The honest statement is therefore that adaptation has *not
been shown to help* at the decision point at either completed scale, not
that it has been shown to hurt.

We report one further discipline point, because it is the part most
easily lost. The same 4B run produced a k=5 arm with a mean paired delta
of +6.9 F1 whose 95% interval excludes zero — a publishable-looking
number, and not the preregistered decision point. It does not survive
inspection: 27 wins against 20 losses is p=0.38 on a sign test, and
removing the two largest winners from sixty receipts drops the mean to
+3.9 F1, below the bar. A parametric interval on a mean is precisely the
statistic that two extreme observations can move, which is why we now
report a sign test and a drop-the-top-winners jackknife beside every
interval, and treat a row where the two disagree as an artifact rather
than a finding. Had we quoted that k=5 number, we would have published
noise with a confidence interval around it.

Rule: preregister the decision point, report the gate you named rather
than the best number you got, and pair every interval with a statistic
that outliers cannot move.

**6.10 Gradient checkpointing that reports enabled and does nothing.**
An infrastructure incident with a measurement lesson. The Addendum B
novel-schema gate (spec frozen 2026-08-12) required training over
~5.4k-token sequences at k=30; every adapted arm OOMed on a T4 while the
k=10 arms (~2.3k tokens) trained cleanly. Five instrumented runs
eliminated suspects in order, each with a positive signature rather than
a plausible story: a wrong-architecture GPU draw (P100, no sm_60 kernel
images — fixed by pinning the accelerator and adding a fail-fast
capability probe); the full-sequence logits tensor inside the HF
labels-path loss (fixed by a chunked cross-entropy whose gradients we
pinned equal to the labels path in unit tests before shipping — the OOM
moved from the loss to the trunk backward, confirming the fix and
narrowing the search); the attention implementation (refuted:
byte-identical OOMs under eager and SDPA); numeric dtype (refuted:
byte-identical OOMs under bf16 and fp16); allocator fragmentation
(refuted: expandable segments cut reserved-unallocated from 830 MB to
93 MB with no change in outcome). What remained was an accounting
identity: ~430 MB of resident activations per layer at 5.4k tokens is
full activation storage, not checkpointed storage — despite
``gradient_checkpointing_enable()`` reporting success and the trunk flag
reading ``True``. In this scoring image's transformers build, the
checkpointing machinery is a silent no-op at layer level. The next
instrumented attempt abandoned the library mechanism — wrap each decoder
layer in ``torch.utils.checkpoint`` directly, print the wrapped-layer
count so the log proves engagement, and restore original forwards before
generation — and the scoring image defeated that too, as it did
``torch.autograd.graph.save_on_cpu``: GPU memory held byte-stable at
~11.3 GB throughout, though both mechanisms verified working locally on
transformers 5.15. The actual repair abandoned the GPU path for the k=30
pairs entirely: CPU/fp32, made feasible by the gradient-pinned chunked
cross-entropy above (ENTERPRISE_EVAL_SPEC.md B.7-r4).

Rule, and it is the same rule as 6.7 and 6.8 wearing new clothes: a
framework flag is a claim, not a measurement. Flags said checkpointing
was on; the memory ledger said otherwise; the ledger wins. Instrument
the failure site before hypothesizing twice about it — the one run we
instrumented taught more than the three we reasoned about.

Epilogue to the incident, for completeness (2026-08-17): the gate this
infrastructure ultimately served — on that CPU/fp32 path, with a
checkpoint/resume layer added when an external canceller repeatedly
killed 13-hour runs — decided GO on its
preregistered terms: mean paired delta +46.5 micro-F1 over seeds
{1,2,3} against a +5 bar frozen before any data (receipt-level sign
test 156W/0L/2T; two unplanned same-environment replications agreeing to
0.002-0.010). The full novel-schema study is its own artifact
(ENTERPRISE_EVAL_SPEC.md Addendum B and the novel_schema_* records in
experiments/), reported here only because this incident's repair — the
chunked loss plus the CPU/fp32 migration — is what made it measurable. Per that spec's claim rule, the positive
travels with its negative: the same adaptation recipe FAILED its
preregistered gates on CORD at all three scales tested (Addendum A).

## 7. Limitations and what we'd scale next

### 7.1 Limitations

**Competition result is small and flat.** Scored submissions to date:
0.00 (v6, incident 6.7), 0.00 (v7, incident 6.8 coverage cap), then
1.67 public three times running (v8 2026-08-10, v9 08-12, v10 08-15) —
the v9/v10 flats are preregistered nulls on the recall-bound and
time-budget levers respectively, leaving candidate quality as the
measured binding constraint (~2.7% per-attempt hit rate). Every accuracy number in this paper is a
small-slice local validation (at most 22 scored test pairs per run), and we
report them as pipeline evidence, not capability evidence.

**The offline half of the recipe is consumed, not reproduced.** We use the
NVARC team's public fine-tuned 4B checkpoint and reproduce only the
inference-time half. The synthetic-data and SFT stages (they report ~27
hours on 32 H100s for SFT alone) remain unreproduced by us, so this work
cannot yet say what fraction of the winning score the offline half carries.

**Single-seed, small-n comparisons.** The NVARC team reports 1–2 points of
run-to-run variance in their own results — their number, and a caution that
applies with more force to our much smaller evaluation slices. Ablation
cells in section 4 will need paired task sets and, where budget allows,
repeated seeds before differences smaller than a few points can be read as
signal.

**The search cutoff interacts with TTT in a way we did not anticipate.**
Teacher-forced diagnostics on the champion checkpoint
(`t4_champ_diag_2026-08-08.json`) measure per-token
log-probabilities of the *true* solutions at −0.063 to −0.120 per token
before TTT (−0.03 to −0.16 across both arms) under our serialization.
That sounds strong, but a few hundred output
tokens at that rate accumulate tens of nats — far past the DFS cutoff of
−log 0.2 ≈ 1.6 nats — so the search can exclude the true completion even
when the model would rank it above every candidate it did find. This
reframes the pipeline: the cutoff-bounded DFS is not a neutral candidate
generator, and TTT's job is not only to improve the model but to *sharpen*
per-token probabilities enough that true completions survive the cutoff.
Two mitigations are already in the harness — the guaranteed greedy
completion in every candidate pool (`dfs_include_greedy`, section 2.3), and
TTT sharpening as an explicit ablation axis — and the section 4.4 row
quantifies the effect (sharpening on 3/5 comparable tasks, not uniform).

### 7.2 What we'd scale next

**Spend the search budget the multi-frame DFS freed up.** Since section 3
was drafted, the harness gained a lockstep-batched multi-frame DFS
(`constrained_dfs_multi`): the augmentation frames of a task are searched
together, one batched forward per step over the live frames instead of one
forward per frame, with logical (attention-masked) cache cropping and
occasional compaction instead of per-fork cache copies. It is oracle-tested
to return exactly what the single-frame search returns, and its merge
validation measured a 7.55× reduction in forward calls on the batched path
(commit `44f39ff`; measured in that validation run but not yet captured in
a registry artifact — TODO(artifact)); wall-clock gains on competition
hardware are still to be measured (TODO(artifact)). That headroom converts directly into more
augmentation frames or higher LoRA rank within the same 12-hour
envelope — candidate-generation levers, which is where the v9/v10
preregistered nulls and the diagnostic above say the accuracy is (the
cutoff itself is a banked null: v9 widened it 1.6 nats for exactly zero
score change, and the preregistration rules out further widening).

**TTT sharpening toward the champion configuration.** Our earlier
validated runs used lite adapters (r=16–64); the rank sweep (row 4.7,
`t4_rank_sweep_2026-08-08.json`) then validated the champion rank r=256 on
a 16 GB T4, and kernel v7 ran r=256/alpha=32 with rslora end-to-end.
Closing the remaining configuration gap — augmentation frames (the
search cutoff is a banked v9 null) — guided by the teacher-forced diagnostic rather
than by end-to-end score alone, is the highest-signal-per-GPU-hour
iteration available to us.

**External validity: an honest off-ARC test, and a consistent
negative.** As a first check that per-task TTT is not an ARC-specific
artifact, we ran the recipe's task-agnostic core (leave-one-out corpus
construction, per-task LoRA, the same training loop — grid serialization
and DFS excluded) on a text task: structured field extraction from CORD
receipts (CC BY 4.0), with a 0.5B Apache-licensed instruct model. The
first pair (k=10, seed 0) showed a large positive delta: k-shot
prompting 0.661 mean field-level micro-F1 on 20 held-out receipts vs
0.788 after minutes of CPU-only adaptation (+12.7 points; CPU-only per
the registry row — the artifact records adapt_seconds=271 without a
device field). A same-day
variance sweep across 4 paired seed/k arms (k=10 seeds 0/2/3; k=5
seed 1) showed that draw to be the favorable one: per-arm deltas +12.7,
−5.2, −6.5, −6.1 — mean −1.3 F1, net-neutral at this scale — while
output formatting stayed robust in every arm (19–20/20 parseable JSON;
a content-quality result, not a harness failure)
(`cord_variance_summary_2026-08-08.json` and the per-arm
`cord_k*_seed*_2026-08-08.json` files). We report this as a strength of
the methodology, twice over. First, it is consistent with the ARC-side
teacher-forced diagnostic (section 7.1): TTT degraded lp(true) on 2 of 5
comparable tasks there — so across two very different datasets, 1-epoch
TTT is high-variance in content quality while leaving
format intact, which is a reproducible characterization rather than a
cherry-picked win. Second, the outcome exercised the preregistered loss
branch of our enterprise eval spec exactly as written: the claimable
statement is the negative result itself, and the named next lever is
model scale. A k=30 arm (larger example budget) was attempted but its
adapted side never completed — the run was interrupted three times by
container restarts, and we report that as-is; its k-shot baseline did
land at 0.672, the strongest prompted arm
(`cord_k30_seed1_kshot_2026-08-08.json`).

Read together, the two diagnostics leave a scale hypothesis on the table.
At 0.5B, one epoch of per-task adaptation is net-neutral on CORD (mean
−1.3 F1 across four paired arms, one favorable draw;
cord_variance_summary_2026-08-08.json) and scored an honest zero on ARC
(0/22; t4_sft_eval_2026-08-08.json), while at 4B — on the champion's
heavily domain-SFT'd checkpoint — the identical one-epoch recipe sharpened
lp(true) on 3 of 5 comparable tasks and 5 of 7 test pairs
(t4_champ_diag_2026-08-08.json). We state this as a hypothesis rather than
a finding: model scale is confounded with domain-specific SFT and with
task family (grid puzzles vs JSON extraction), and both slices are small.
The preregistered scaled run (ENTERPRISE_EVAL_SPEC.md, loss branch: "next
lever is model scale") therefore holds the CORD harness, seeds, and
paired-arm protocol fixed and swaps only the base model (0.5B → 1.5B and
up): if adaptation gain is scale-dependent, the paired deltas should move
from net-neutral toward uniformly positive; if they stay net-neutral, the
negative result attaches to the recipe, not to the 0.5B model.

**Resolution (updated 2026-08-14): all three rungs complete, all fail.**
−7.3 F1 at 0.5B, −11.5 F1 at 1.5B, −4.5 F1 at 4B (§6.9). The 0.5B→1.5B
pair shares device and dtype (CPU/fp32), so that comparison carries no
environment confound — and tripling the model made the paired delta
worse, not better. The deltas did not move toward uniformly
positive, which is the direction the preceding paragraph named as
evidence against the scale hypothesis.

The protocol did not, however, swap *only* the base model. The 0.5B rung
ran on CPU in fp32 and the 4B rung on a T4 in bf16, so base-model scale
is confounded with device and numeric precision in this comparison — the
same class of confound the paragraph above flags between scale and
domain-specific SFT. We state the reading accordingly: the deltas moved
in the direction that argues against scale rescuing the recipe, across
two rungs that differ in more than scale. The 1.5B rung — CPU/fp32, directly comparable to 0.5B — has
now landed and settles the cleaner test: scale does not rescue the
recipe on this task. (The 4B CPU replication intended to retire the
device/dtype half proved infeasible — 4B fp32 exceeds the CPU session's
memory — so the 4B rung's own number retains that caveat.)

We hold to that reading with two bounds. No rung's failure is
individually significant (p=0.60 at 0.5B, p=0.42 at 1.5B, p=0.10 at 4B by
sign test) against a gate whose minimum detectable effect exceeds its own
threshold, so what the evidence supports is the absence of a demonstrated
benefit, not a demonstrated harm. The confound named above is also only partly retired:
0.5B and 4B differ in domain-specific SFT as well as in scale, so the
recipe-level reading is the most parsimonious explanation of these
artifacts rather than the only one they permit.

**What we would *not* scale next: the TRM ensemble.** The 2025 recipe
included a Tiny Recursive Model ensemble, and recipe worship would say to
reproduce it. The NVARC team's own ablation says otherwise: per their
paper, ensembling TRM with their 4B model changed the score not at all
(27.22 → 27.22 public), helped their 2B model by about a point
(21.53 → 22.50, in late submissions — their in-competition ensemble
submission actually scored *lower* than their pure LLM run, at 20.28), and
TRM alone scored 7.5–10.0 — all their numbers. That makes TRM a plausible
one-to-two-point component for a mid-strength LLM and approximately zero
for a strong one. We keep the candidate-injection interface in the voting
layer (any external candidate source can be rescored under the
identical-augmentation-set rule), but the component itself is deprioritized
until the LLM side plateaus — a conclusion available to anyone who reads
the winner's ablation table instead of their architecture diagram, which
is, in miniature, this paper's argument.
