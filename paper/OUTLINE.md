# Paper-track outline (draft; due Nov 8, 2026)

Working title: "A Clean-Room Reproduction Harness for Test-Time Training on
ARC-AGI-2: What Transfers, What Breaks, and What It Costs"

Angle: the paper prize rewards clarity and honest analysis, not leaderboard
rank (any linked code submission qualifies). Our differentiated contribution:
a from-scratch, fully tested reimplementation of the 2025 winning recipe with
measured ablations of each component (format fidelity, TTT, DFS decoding,
augmentation voting) and honest cost accounting (per-task s and $ measured;
$/point not yet computable by our own rules — no non-zero scored run yet).

1. Introduction — why reproduction-with-ablation beats recipe worship.
2. The recipe, decomposed — serialization format, per-task LoRA TTT,
   constrained DFS, augmentation voting; what the original writeup
   under-specifies.
3. Harness — invariants (invertible augmentations, golden-format tests,
   fail-closed loaders), pure-torch LoRA for offline scoring images.
4. Ablations (each from experiments/*.json):
   - chat-template vs raw champion format;
   - sampling vs constrained DFS;
   - corrupted vs exact DFS (the cache-aliasing bug is itself an ablation:
     how much does a *correct* search matter vs a near-greedy one?);
   - TTT on/off (teacher-forced lp(true) isolates format fit from search);
   - augmentation count sweep (dihedral only vs x color perms vs + shuffle);
   - base model scale (0.5B SFT vs champion 4B);
   - LoRA rank feasibility sweep (r=16/64/128/256 all fit a 16 GB T4;
     r=64 fastest and lightest; basis for running the champion r=256
     in-kernel — draft row 4.7).
5. Cost curves — per-task seconds and $ measured on commodity GPUs (T4,
   2×T4); $/point not yet computable by our own rules (no non-zero scored
   run yet — publish the cost side now, fill $/point when one lands);
   single-GPU vs 2-way task sharding in the kernel's own run (the scored
   file comes from that run — there is no separate scoring VM).
6. Negative results and incidents — every bug that a paper reader would
   otherwise re-hit:
   - Kaggle mount paths (recursive discovery or death);
   - peft absent on offline scoring images (pure-torch LoRA);
   - UNK-less cut tokenizers raise on out-of-vocab probes;
   - gradient checkpointing x frozen base (enable_input_require_grads) and
     x cached generation (disable before generate);
   - hand-synced bundles drift (indented intra-package imports killed a full
     scoring run while local package runs stayed green — build, don't sync);
   - HF KV caches mutate in place: per-beam cache references alias, so
     tree search must snapshot or crop-on-backtrack; an oracle test against
     cache-free full forwards is the only thing that caught it;
   - the scoring environment IS the interactive environment (6.7): the
     submission is scored from the kernel's own run output, so pin the
     accelerator (machine_shape) — an unpinned run landed on a non-bf16
     GPU and the graceful fallback converted it into a silent 0.00;
   - library API drift between dev and scoring images (6.8): the scoring
     image's newer transformers dropped the legacy KV-cache bridge and
     generic cache iteration broke; probe API surfaces explicitly and
     regression-test against mocks of versions you cannot install.
7. Limitations and what we'd scale next:
   - CORD external-validity check: per-task TTT net-neutral across 4
     paired arms at 0.5B (mean −1.3 F1, one favorable draw; format intact
     in every arm) — a preregistered negative result, k=30 arm pending;
   - the scale hypothesis: 0.5B net-neutral vs 4B lp(true) sharpening —
     confounded with SFT and task family; the preregistered scaled run
     swaps only the base model;
   - multi-frame DFS headroom (7.55× fewer forward calls) to spend on
     frames, cutoffs, or rank within the 12 h envelope;
   - TRM ensemble deprioritized per the winner's own ablation.

Requirement checklist: link a Kaggle code submission; CC BY 4.0; reproducible
from the public repo.
