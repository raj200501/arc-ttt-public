# TRM Ensemble Integration Plan (ROADMAP item 5)

Status: research complete, implementation not started. All facts verified 2026-08-08
unless marked **[verify on GPU]**. Verification routes used this session: GitHub clones
(SamsungSAILMontreal/TinyRecursiveModels and 1ytic/NVARC, incl. the extracted
solution paper), the **Kaggle API** (dataset
metadata/files and both NVARC notebooks pulled with our project credentials — kaggle.com
web pages remain blocked by the egress proxy), and web search. Clean-room rule for this
component: the upstream TRM repo is MIT so we may vendor it verbatim; the NVARC repo has
**no license**, so anything that exists only there (their modified `eval-arc-k-10.py`
driver) is described and re-derived, never copied.

## 0. What TRM is (context)

Tiny Recursive Model — Samsung SAIL Montréal, Alexia Jolicoeur-Martineau, "Less is More:
Recursive Reasoning with Tiny Networks", [arXiv:2510.04871](https://arxiv.org/abs/2510.04871).
A single 2-layer, hidden-size-512 transformer (~7M weights) applied recursively
(H_cycles × L_cycles inner recursions, ACT halting head `q_halt`, up to `halt_max_steps`
outer steps), non-autoregressive: it emits the whole output grid in one shot per halting
step. Paper reports 45% on ARC-AGI-1 and ~8% on ARC-AGI-2 public eval. Code:
[github.com/SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
(MIT, © 2025 Samsung Electronics — LICENSE verified in clone).

NVARC's use ([1ytic/NVARC](https://github.com/1ytic/NVARC) `TRM/README.md` + `nvarc_2025.pdf`
§4, both in our clone): pretrain a variant (L_layers=2, H_cycles=3, L_cycles=4, EMA,
lr 3e-4, batch 3072, 10k epochs, 24h on 8×H100) on 4,073 puzzles × 256 augs; at
submission time, **test-time fine-tune the pretrained weights on the augmented demo
pairs of the 240 hidden tasks** (their tuned config: H_cycles=4, halt_max_steps=10,
2000 epochs, 200 warmup, lr 1e-4, batch 128, EMA, ~2h on Kaggle), then emit **10 ranked
attempts per test input** and hand them to the LLM for rescoring. TRM alone scored 7.5
(v18, 2k epochs) and 10.0 (post-deadline, 4k epochs, <4h) on the public LB.

**Honest expected-value note.** The paper's own ensemble numbers (§4.4): Qwen3-2B
21.53 → 22.50 with TRM (late submissions; their in-competition ensemble sub actually
scored *lower*, 20.28), and Qwen3-4B 27.22 → 27.22 (**zero gain**). "Most of the puzzles
solved by TRM were solved by Qwen3"; Qwen3 rescoring picked on average ~1 extra puzzle
from TRM. So this is a **+0.5–2 point** component for a mid-strength LLM and ~0 for a
strong one — worth having while our LLM is weak (TRM standalone at 10.0 may even exceed
our current LLM score), but it is not a 5-point lever. ROADMAP now lists the TRM
ensemble as DEMOTED per this plan — read it as the biggest remaining *recipe* item,
not the biggest guaranteed delta.

## 1. What TRM inference requires

### 1.1 Input serialization (verified in `dataset/build_arc_dataset.py`, MIT clone)

- Grid → flat 900-token sequence: fixed 30×30 canvas, row-major. Vocab 12:
  PAD=0, EOS=1, colors 0–9 → tokens 2–11. The grid is placed at an offset inside the
  canvas (random top-left translation for train examples, none for test), an EOS row/col
  marks its bounding box, PAD fills the rest. Input and output grids are separate
  900-token sequences (`inputs`/`labels` arrays).
- Each *augmented copy* of a puzzle is its own puzzle: id string
  `"{name}|||t{trans_id}|||{color_perm}"`, with per-copy learned puzzle embedding
  (`puzzle_emb_ndim=512`, `puzzle_emb_len=16` prefix tokens — `config/arch/trm.yaml`).
- On-disk dataset format: `train/` and `test/` dirs with
  `all__{inputs,labels,puzzle_identifiers,puzzle_indices,group_indices}.npy`,
  `dataset.json` metadata (seq_len=900, vocab_size=12), plus `identifiers.json` and
  `test_puzzles.json` at the root (verified against
  `kaggle datasets files cpmpml/arc-prize-trm-evaluation-data`).

### 1.2 Augmentation protocol (verified, same file)

- `aug()`: uniform-random dihedral transform (one of 8) × random color permutation that
  **fixes color 0** (black); dedup by SHA-256 puzzle hash; up to 5×N retries to reach N
  distinct augs. NVARC used **num_aug=128 for the test set** (their Kaggle notebook,
  cell 5) and 256 for pretraining data.
- `inverse_aug()` parses the id string and undoes color perm + dihedral, so predictions
  from every augmented copy are voted **in the original frame** — same principle as our
  `arcttt/vote.pool_predictions`, implemented via id-string bookkeeping instead of
  carrying augmentation objects.
- Random translation inside the 30×30 canvas is a *third* augmentation applied to train
  examples only (test inputs are corner-anchored).

### 1.3 Checkpoint format (verified: Kaggle API + `pretrain-no-eval.py` in NVARC clone)

- [kaggle.com/datasets/cpmpml/arc-prize-trm-031](https://www.kaggle.com/datasets/cpmpml/arc-prize-trm-031):
  **7 files**, `step_110355 … step_275886`, each **2,159,719,349 bytes (~2.16 GB)**,
  license **CC0-1.0**, owner cpmpml, collaborator sorokin (dataset id 8687355; via
  `kaggle datasets metadata`). Note their README says "10 checkpoints" — the published
  dataset has 7. NVARC submitted with **`step_220708`** (chosen by a
  leave-eval-out proxy run, paper §4.3).
- Format: raw `torch.save(model.state_dict())` of the `torch.compile`d model+loss-head
  (keys carry the `_orig_mod.` prefix, e.g. `_orig_mod.model.inner.puzzle_emb.weights`).
  Size arithmetic confirms composition: puzzle-embedding table 1,041,208 ids × 512 ×
  fp32 ≈ 2.13 GB + ~7M model weights ≈ 28 MB ≈ 2.16 GB. **~99% of the checkpoint is the
  pretrained puzzle-embedding table, which is discarded at load**: the loader resets a
  shape-mismatched table to the row-mean, broadcast to the test dataset's id count
  (`load_checkpoint()` in their script; same logic exists in upstream `pretrain.py`).
  So we only need **one** checkpoint file attached to the kernel, and only ~28 MB of it
  survives into TTT.
- Loading requires instantiating the same arch config (arch=trm, L_layers=2; H_cycles/
  L_cycles/halt_max_steps are runtime loop counts, not weight shapes, so NVARC's
  eval-time override H_cycles 3→4 is weight-compatible — verified by them running it).

### 1.4 Runtime / VRAM envelope (Kaggle 4×L4, 96 GB total)

> **Correction (2026-08-08, per KAGGLE_MECHANICS.md):** the 2026
> ARC-AGI-2 Kaggle track provides **T4/P100 only**; our actual
> interactive runs drew 2×Tesla T4 (~32 GB total). The "4×L4, 96 GB"
> envelope below — and the paper's "2000 epochs ≈ 2h" anchor — describe
> NVARC's 2025-vintage environment. Restate for 2×T4 before budgeting:
> `torchrun --nproc-per-node 2`; wall-clock (and B3's ~2.5h TRM-TTT cap /
> "all 4 GPUs" split) needs re-measurement on 2×T4, with epoch/aug-count
> reduction as the fallback knob.

- Their public TRM notebook ([kaggle.com/code/cpmpml/arc2-trm-v31](https://www.kaggle.com/code/cpmpml/arc2-trm-v31),
  pulled via `kaggle kernels pull`): builds the 128-aug test dataset in-notebook, then
  one `torchrun --nproc-per-node 4` job that TTT-trains and evaluates
  (`epochs=4000, eval_interval=4000, global_batch_size=128, lr=1e-4, lr_warmup_steps=200,
  ema=True, H_cycles=4, L_cycles=4, halt_max_steps=10, load_checkpoint=step_220708`).
- Timing anchors from the paper: **2000 epochs ≈ 2h**, 4000 epochs < 4h wall-clock on
  Kaggle for 240 tasks. Our numbers at batch 128 on L4s: **[verify on GPU]**.
- VRAM: 7M-param model, batch 32/GPU × 900 tokens — trivially fits an L4; the binding
  constraint is **time**, shared with our LLM TTT phase, plus one-off ~2.2 GB
  disk/load cost for the checkpoint. Peak RSS during dataset build (240×129 puzzles ×
  ~900×2 int arrays) is a few GB CPU RAM. **[verify on GPU]**
- Offline deps (their cell 0): `hydra-core==1.3.2 adam_atan2_pytorch==0.2.4
  argdantic==1.3.3 coolname==2.2.0` installed from a bundled-wheels Kaggle dataset;
  `numba` (used by the ARC evaluator) is already in the Kaggle image. We must add these
  wheels to our `kaggle/build_bundle.py` bundle.

### 1.5 How the 10 attempts are produced and ranked (verified in `evaluators/arc.py`, MIT)

Every augmented copy of a task produces one predicted grid + a `q_halt` confidence
(sigmoid of halt logit). The evaluator inverts each prediction to the original frame,
crops the 30×30 canvas back to a rectangle, and votes: candidates ranked by
`(vote_count, mean sigmoid(q_halt))`, top `submission_K` emitted as `attempt_1..K`.
Upstream default `submission_K=2`; NVARC's ensemble source used `submission_K=10`
(their `eval-arc-k-10.py` hard-codes it; the pure-TRM public notebook keeps K=2).
The K=10 change is a one-argument difference — trivially clean-room.

## 2. Feeding our voting/rescoring layer (`src/arcttt/vote.py` + `solve.py`)

NVARC's mechanism (paper §4.4, exact merge code **not public** — the current version of
their LLM notebook contains no TRM references, we checked): TRM's 10 attempts were
"added to the attempts generated by the [DFS] procedure … then scored by the Qwen3 model
like the other attempts." That maps 1:1 onto our existing structures:

1. Run the TRM phase first; parse its `submission.json` (attempt_1..10 per test input)
   into `dict[task_id, list[list[Grid]]]`.
2. In `solve_task()` (`src/arcttt/solve.py:38`), after `pool_predictions()` builds
   `counts`, **union in the TRM candidates** before rescoring: for each TRM grid not
   already in `counts`, add it with `found_count = w_trm` (see below); grids already
   found by DFS keep their DFS count. Everything then flows unchanged through the
   existing batched `log_probabilities` rescoring path and
   `select_attempts` (`found_count + exp(mean_log_p)` ranking).
3. Because rescoring uses one identical augmentation set for every candidate
   (`vote.rescore_candidates` docstring — the NVARC-paper property we already
   implement), TRM-originated and DFS-originated candidates are directly comparable.

Open design choice — **the found_count credit `w_trm` for TRM-only candidates is not
publicly specified by NVARC**. Options, to be A/B'd on the public eval split:
- `w_trm = 0`: pure LLM-log-prob rescoring of TRM grids (a TRM grid then wins only when
  no DFS candidate has a count ≥ 1, since exp(mean log p) ≤ 1 — probably too weak);
- `w_trm = 1` (recommended default): a TRM attempt counts like one DFS find;
- `w_trm = f(rank or vote share)`: pass TRM's own vote count through (needs a scale
  calibration; TRM votes over ~129 augs vs our DFS finds over ≤16 frames).
Also keep a **TRM-standalone fallback**: if the LLM phase dies, submit TRM's own
attempt_1/attempt_2 — worth ~10 points on its own, which may beat our current pipeline.

Interface contract to write (`src/arcttt/trm_bridge.py`): read TRM submission JSON →
`{task_id: [ [Grid, ...] per test input ]}` with validation (2-D, 1–30 per side, values
0–9), tolerant of missing tasks (TRM prints "has no predictions" and pads by repeating
attempt_1 — both behaviors verified in `evaluators/arc.py`).

## 3. Licensing / competition-rules check

- **Competition**: "External data, freely & publicly available, is allowed, including
  pre-trained models" (official Code Requirements; Rules §2.6 reasonableness clause) —
  see `docs/research/KAGGLE_MECHANICS.md` §4 (verified 2026-08-08 against
  [kaggle.com/competitions/arc-prize-2026-arc-agi-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)).
  The checkpoints are a public Kaggle dataset usable by any participant at zero cost →
  compliant. Attaching public datasets/models to the notebook is the standard pattern.
- **Checkpoints** (`cpmpml/arc-prize-trm-031`): **CC0-1.0** (public domain dedication,
  verified via Kaggle API metadata). No legal attribution requirement; we cite
  NVARC + TRM in the writeup anyway (paper-track integrity, and arcprize open-source
  rubric rewards provenance).
- **Upstream TRM code**: MIT — we may vendor `models/`, `puzzle_dataset.py`, `dataset/`,
  `evaluators/`, `utils/`, `config/` into our bundle, keeping the Samsung copyright
  notice. MIT is above the Apache-2.0/GPLv3 floor arcprize sets for third-party code
  (KAGGLE_MECHANICS §1).
- **NVARC repo: no LICENSE** → clean-room boundary. Their `pretrain-no-eval.py` /
  `eval-arc-k-10.py` are lightly modified copies of upstream MIT `pretrain.py` (wandb
  removed, submission_K=10, no-eval loop), but we cannot verify the delta is theirs to
  relicense, so: **write our own thin driver against upstream `pretrain.py`**,
  re-deriving only the *described* deltas (hyperparameters from their public README/
  paper — facts, not copyrightable expression; K=10 as an evaluator argument upstream
  already supports). Same rule for the merge logic (paper description only).
- **Our obligation**: prize eligibility requires open-sourcing our solution (CC0/MIT-0
  for our code) and a Kaggle writeup within 7 days of deadline — nothing about TRM
  changes this; the CC0 checkpoints can be redistributed/attached freely.

## 4. Implementation steps (riskiest unknowns first)

Phase R — de-risk (do before wiring anything):
1. **R1 (3h, riskiest): checkpoint load smoke test.** Lightning T4/L4: vendor upstream
   TRM, download `step_220708` via Kaggle API, build a 4-aug dataset from 5 public eval
   tasks, run our driver for ~20 epochs, confirm (i) state_dict loads (compile prefix,
   emb mean-reset fires), (ii) loss decreases, (iii) evaluator emits a valid
   submission.json. Kills the "checkpoints prove unusable" risk for ~$1 of credits.
   **[verify on GPU]**
2. **R2 (2h): torch.compile + deps on target image.** Their code `torch.compile`s by
   default (env `DISABLE_COMPILE` opts out); confirm compile works on L4 + current
   Kaggle image, else run eager and measure the slowdown. Confirm the 4 wheels install
   offline. NVARC reported version conflicts made the ensemble "very challenging" to
   co-install with the LLM stack — our mitigation: TRM runs as a **separate torchrun
   subprocess with its own sys.path**, communicating only via submission.json on disk,
   never importing into the Unsloth process. **[verify on GPU]**
3. **R3 (4h): public-eval calibration run.** Full TTT (128 augs, 2000 epochs, K=10) on
   the 120 public eval tasks on Lightning 4-GPU; record wall-clock, pass@1/2/10.
   Expected ballpark: pass@2 ≈ 9–10% (their local: 9.44% eval-held-out; 10.14% trained-on-eval,
   `TRM/README.md` stats block). If we can't reproduce ≥ ~7% here, stop and re-plan.
   **[verify on GPU]**

Phase B — build:
4. **B1 (4h): clean-room driver + K=10 evaluator wiring** (`kaggle/trm_phase.py`):
   dataset build (num_aug=128) → torchrun TTT → submission.json with 10 attempts;
   config knobs for epochs/augs/batch so we can shrink to fit the time budget.
5. **B2 (3h): `src/arcttt/trm_bridge.py` + vote.py injection** (§2), with unit tests
   mirroring `tests/test_pipeline.py` style: TRM-only candidate wins iff score clears
   DFS candidates; w_trm parameterized; malformed-grid rejection.
6. **B3 (3h): bundle + kernel budget split** (`kaggle/build_bundle.py`, entry script):
   attach checkpoint dataset + wheels; sequence phases TRM-TTT (all available GPUs —
   2×T4 on the 2026 track, see the §1.4 correction; hard cap ~2.5h wall-clock, to be
   re-measured on 2×T4, with epoch-count fallback) → LLM TTT/DFS (remaining ~8.5h) →
   merge/rescore. Add the TRM-standalone submission fallback path.
7. **B4 (2h): end-to-end dry run on Lightning** with a 10-task slice; then A/B w_trm ∈
   {0, 1} on the public eval split. **[verify on GPU]**

Phase S — ship: one Kaggle submission slot with the ensemble config only after the
public-eval A/B beats our current best config (per ROADMAP submission discipline).
Total: ~21h engineering + ~3 Lightning GPU-days of validation.

Biggest unknowns, ranked: (1) checkpoint loads and TTT-converges outside their exact
image (R1/R2); (2) 12h-budget contention — TRM's ~2h+ comes out of the LLM's TTT time,
which is currently our scoring engine (B3 measurement decides if the trade is positive);
(3) w_trm merge weighting is unpublished — pure experiment (B4); (4) run-to-run variance
(NVARC saw 1–2 points) can mask a small TRM gain in a single public-eval A/B.

## 5. Fallback if the checkpoints prove unusable

Trigger: R1/R3 fail (weights don't load, or reproduce < ~7% pass@2 on public eval).
- **F1 — pretrain our own TRM (budget option).** Trelis' "Test-time Adaptation of Tiny
  Recursive Models" ([arXiv:2511.02886](https://arxiv.org/abs/2511.02886)) shows a
  from-scratch pretrain on **1,280 public tasks, 48h on 4×H100 (~$200–300)** reaching
  ~10% public eval, then competition-time TTT at 6.67% semi-private. Uses only MIT
  upstream code + public ARC data → zero licensing exposure. Cost: ~2 GPU-days + our
  time; fits our compute budget if TRM proves valuable in R3-style experiments with
  their released recipe.
- **F2 — TRM-free ensemble diversity (zero new compute).** Keep the §2 bridge (it is
  model-agnostic: any candidate source → grids → LLM rescoring) and feed it cheap
  non-TRM candidates instead (e.g. heuristic/program-search solvers on public data).
  Preserves the architecture work; the bridge is reusable the day any candidate source
  materializes.
- **F3 — drop item 5.** Given §0's honest EV (+0.5–2 pts mid-tier, ~0 for strong LLMs),
  if R3 costs exceed a week, reallocating to ROADMAP items 1–4 (LLM TTT scaling, which
  NVARC's own ablation shows dominates) is the rational move.

## Sources

- [github.com/SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) — MIT LICENSE, `dataset/build_arc_dataset.py`, `evaluators/arc.py`, `config/` (cloned 2026-08-08)
- [arXiv:2510.04871](https://arxiv.org/abs/2510.04871) — TRM paper (abs page proxy-blocked; details via search results incl. [huggingface.co/papers/2510.04871](https://huggingface.co/papers/2510.04871))
- [github.com/1ytic/NVARC](https://github.com/1ytic/NVARC) — `TRM/README.md` (train/eval commands, checkpoint provenance, local stats), `nvarc_2025.pdf` §4 (TRM pretrain/TTT/ensembling; text extracted from nvarc_2025.pdf), **no LICENSE file** (cloned)
- [kaggle.com/datasets/cpmpml/arc-prize-trm-031](https://www.kaggle.com/datasets/cpmpml/arc-prize-trm-031) — CC0-1.0, 7 × 2,159,719,349-byte checkpoints (via Kaggle API `datasets metadata`/`files`, 2026-08-08); sibling datasets `cpmpml/arc-prize-trm-{training,evaluation}-data` also CC0-1.0
- [kaggle.com/code/cpmpml/arc2-trm-v31](https://www.kaggle.com/code/cpmpml/arc2-trm-v31) — pure-TRM submission notebook (pulled via `kaggle kernels pull`, latest version = post-deadline 4000-epoch variant)
- [kaggle.com/code/sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue](https://www.kaggle.com/code/sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue) — LLM notebook (pulled; latest version contains **no** TRM merge code — ensemble merge is unpublished)
- `docs/research/KAGGLE_MECHANICS.md` §1/§4 — competition external-model + open-source rules (verified 2026-08-08)
- [arXiv:2511.02886](https://arxiv.org/abs/2511.02886) — Trelis, "Test-time Adaptation of Tiny Recursive Models" (fallback pretraining recipe)
- `src/arcttt/vote.py`, `src/arcttt/solve.py`, `kaggle/v10/bundled_pipeline.py` — our injection points (read this session)
