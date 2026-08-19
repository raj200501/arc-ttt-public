# micro-train runbook — G7 micro-tier own-model proof on free Kaggle T4 quota

Bounded LoRA continued-training of **Qwen2.5-0.5B-Instruct** (not the
champion 4B — the point is an adapter WE own, on license-clean public
TRAINING data) as a Kaggle script kernel on the **free weekly 30h GPU
quota**. Promotional Lightning credits are untouched; this is the
"cheapest tier" G7 exit (GOALS.md — internal planning doc, not
included in the public cut), staged below Tier 1 of the internal
own-model plan.

## What the kernel does (kaggle/entry_micro_train.py → bundled_pipeline.py)

1. Loads `*training_challenges.json` + `*training_solutions.json` from
   `/kaggle/input` (TRAINING set only — never evaluation or test data as
   training input) and serializes tasks in the raw-qwen digit format the
   harness decodes.
2. Baseline held-out eval: 10 public **evaluation** tasks (sorted by
   task_id, deterministic slice), small solve budget (4 D4 frames, DFS
   20 s/frame-set, no TTT) + teacher-forced mean lp(true). Runs on the
   second T4 in parallel with training when 2 GPUs are up, else inline
   first.
3. Budget-driven training loop on cuda:0: pure-torch LoRA r=64/α=32
   rslora (no peft — kaggle-v3 lesson), bf16 via the functional probe
   (fp32 fallback, loudly), grad-accum 8, warmup 50 steps then constant
   1e-4, grad-clip 1.0, OOM/too-long/nonfinite skips counted, **atomic**
   adapter + loss-curve checkpoint every 25 optimizer steps.
4. After-arm of the same eval on the adapter-carrying model.
5. Artifacts in `/kaggle/working`: `adapter_micro.safetensors`,
   `train_log.json` (steps/loss/lr/tokens/tok-per-s + both eval arms),
   `eval_before.json`. Final `REGISTRY |` line printed with the row
   numbers.

This kernel writes **no submission.json** and must never consume the
1/day submission slot.

## Push (DO NOT run before the v8 kernel's interactive run completes)

A push replaces the kernel's interactive run — same account, but this is
a *different kernel id* (`arc-ttt-micro-train` vs `arc-ttt-v1`), so it
does not collide with the v8 output. Still: push only after v8 completes,
so the two runs don't contend for the same weekly quota window at once.

```bash
cd /workspace/arc-ttt
# gates first:
python kaggle/build_bundle.py kaggle/entry_micro_train.py kaggle/micro/bundled_pipeline.py
python kaggle/micro/smoke_micro_train.py
# push:
kaggle kernels push -p kaggle/micro
# watch:
kaggle kernels status rajskashikar/arc-ttt-micro-train
# pull artifacts when complete:
kaggle kernels output rajskashikar/arc-ttt-micro-train -p experiments/micro_train_output
```

Credentials: `~/.kaggle/kaggle.json` only — never in git (standing rule).

## Expected wall-clock and budget math

- Total kernel budget: **4.0 h** (`WALL_BUDGET_SECONDS`), self-enforced
  well under the 12 h platform kill.
  - baseline eval: ~15–30 min (parallel on GPU 1 when 2 T4s are up;
    inline it shortens the train window, capped at 45 min)
  - training: ~3.2 h (deadline = 4 h − 40 min after-eval reserve − 5 min
    save margin)
  - after-eval: ≤ 40 min reserve
- Throughput is **unknown until measured** (bf16 on T4 is emulated); the
  loop is budget-driven, so any tok/s still yields a complete artifact.
  CPU smoke measured ~4k tok/s on a toy model — meaningless for the T4
  number; the run itself is the measurement (same rule as Tier 1's
  re-measure clause in OWN_MODEL_PLAN.md, internal — not included in
  the public cut).

## Weekly quota math (free tier, 30 h GPU/week)

| Run | Wall | Quota after |
|---|---|---|
| v8 interactive (2×T4 session) | ~11 h | ~19 h left |
| micro-train (this) | ~4 h | ~15 h left |

Headroom: ~15 h for a v8 re-push or a second micro run this week.
**Open question**: whether Kaggle debits a T4×2 session at 1× or 2×
wall-clock against the 30 h — check the quota meter right after the v8
run and rescale this table if it's 2× (v8 would then be ~22 h and
micro-train must wait for the weekly reset or run on a 1×GPU shape).

## Registry row template (experiments/README.md)

```
| micro-train | Qwen2.5-0.5B-Instruct + our LoRA r=64 | continued training on ARC public TRAINING set (dihedral-8, raw format, seed 0), ~Xk optimizer steps / Y M tokens / Z tok/s on 1×T4, paired 10-task public-eval slice (no TTT, 4-frame DFS 20 s) | before A/N solved, mean lp(true) B; after C/N solved, mean lp(true) D — whatever it shows | micro_train_output/train_log.json + adapter_micro.safetensors |
```

Numbers come only from `train_log.json` (claim-integrity rule). A 0.5B
solving 0/10 both arms is an acceptable, publishable outcome — the
lp(true) delta is the primary number at this scale.

## Push-time verification items (do NOT guess these)

1. **Model source handle** — `model_sources` is set to
   `qwen-lm/qwen2.5/Transformers/0.5b-instruct/1` by analogy with the
   v8 handle format (`sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`).
   The Kaggle model family `qwen-lm/qwen2.5` **exists** (page title +
   "0.5 to 72 billion parameters" description verified 2026-08-08), but
   the exact framework/variation/version slug could NOT be verified from
   this machine (Kaggle's SPA returns 200 for bogus variation URLs too).
   Before pushing: `kaggle models instances files qwen-lm/qwen2.5/transformers/0.5b-instruct`
   (or open the model page's variation dropdown) and fix the metadata to
   the real slug + latest version number. If no Kaggle-hosted
   0.5b-instruct variation exists, fall back to uploading a private
   Kaggle model from the HF weights (Apache-2.0, attribution preserved).
2. **Competition data file names** — the kernel globs
   `*training_challenges.json` / `*training_solutions.json` /
   `*evaluation_challenges.json` / `*evaluation_solutions.json`. The v7
   log proves the mount layout
   (`/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json`)
   but only the *test* file name is directly evidenced. The kernel fails
   loudly with a tree listing if a glob misses; check the first minutes
   of the interactive run.
3. **Quota accounting for T4×2** (see table above).
4. **Second-GPU availability** — `machine_shape: NvidiaTeslaT4` gave v7/v8
   two T4s; if this kernel gets one, the baseline eval runs inline and
   the train window shrinks by ~25 min (already handled in code).
5. **transformers API drift in the scoring image** — the v7 cache
   postmortem fixes are in the bundle (decode.py probes); training itself
   uses only `model(input_ids, labels=...)` + `AutoModel.from_pretrained`,
   deliberately low-surface.

## Post-review additions (2026-08-08 late — adversarial review findings applied)
- Bundle REBUILT to include use_cache=False scoring forwards (the committed
  bundle predated commit b62c373 and would have OOM-risked the eval that
  produces the registry number) + alloc-RuntimeError ladder in the train
  loop + pre-loop checkpoint + per-task atomic before-arm writes + paired
  lp means over the both-arms task intersection (PAIRED field in the
  REGISTRY line — use THAT number for the registry row, not the raw means).
- Push-time verification items ADDED: (a) a failed script run does NOT
  publish /kaggle/working — the pre-loop checkpoint and alloc catches are
  the mitigation, not the wall margins; (b) confirm image torch >= 1.13
  (we use torch.cuda.OutOfMemoryError, present since 1.13, instead of the
  >=2.4-only top-level spelling); (c) the 2-GPU spawn path is untested by
  any offline gate — read the worker exitcode line in the log; (d) model
  handle verify command must use the metadata's exact case
  (Transformers/0.5b-instruct).
