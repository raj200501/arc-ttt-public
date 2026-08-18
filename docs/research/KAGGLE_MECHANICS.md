# ARC Prize 2026 — Operational Mechanics (ARC-AGI-2 Kaggle track focus)

**All facts below verified by direct fetch of live pages and Kaggle's competition API on 2026-08-08 UTC** (arcprize.org pages, Kaggle internal API `GetCompetition`/`ListPages`/`GetLeaderboard` for competition ID 133469, and clones of the official GitHub data repos). Anything not directly verified is marked.

## 1. Competition structure

**$2M total, 3 tracks** (https://arcprize.org/competitions/2026):

- **ARC-AGI-3 (flagship, interactive)** — $850K: $700K Grand Prize for first agent scoring 100% on the ARC-AGI-3 game environments (rolls forward if unwon); $75K guaranteed Top Score awards ($40/15/10/5/5K); $75K Milestone prizes (June 30 and Sept 30, 2026, $25/10/2.5K each). Kaggle: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3. Agents explore hidden game environments with no instructions. Not covered further here.
- **ARC-AGI-2 (static reasoning)** — $700K (details below). Kaggle: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2
- **Paper Prize** — $450K: Top Paper $75K guaranteed ($50/20/5K) + $375K "Outstanding Papers" pool for papers scoring >4.5/5 on the rubric. Papers must link to a real Kaggle code submission (either track); score need not be high. Kaggle: https://www.kaggle.com/competitions/arc-prize-2026-paper-track (https://arcprize.org/competitions/2026/paper)

**Key dates** (all 23:59 UTC, from Kaggle Timeline + API): start **March 25, 2026**; **entry + team-merger deadline Oct 26, 2026** (API: `2026-10-26T23:59:00Z`; notebook publishing also locks then); **final submission deadline Nov 2, 2026** (`2026-11-02T23:59:00Z`); papers due Nov 8; results Dec 4, 2026.

**Teams:** max 5; mergers allowed if combined submission count ≤ days-elapsed × 1/day. **Open source:** all leading participants must open-source to be prize-eligible (removed from competition otherwise). arcprize.org demands your own code under a permissive license (CC0 or MIT-0; third-party code at least Apache-2.0/GPLv3-class); the Kaggle rules page states "Winner License Type: Open Source - CC BY 4.0". Solutions must be attached to a Kaggle Solution Writeup **within 7 days** of the deadline. Kaggle identity verification is required (`requiresIdentityVerification: true`).

## 2. ARC-AGI-2 track specifics

- **Prizes ($700K; API notes $550K guaranteed + $150K bonus):** Progress Prizes $275K (1st $75K, then 50/40/35/25/20/15/15K across 8 places); Grand Prize $275K for the best Solution Writeup judged on 6 rubric criteria (accuracy, universality, progress, theory, completeness, novelty, 0–5 each); Bonus $150K unlocked by ≥85% on the leaderboard, split among up to 5 qualifying teams ($75/25/20/20/10K); rolls to 2027 if unmet.
- **Compute:** Code competition, notebook submissions only (`onlyAllowKernelSubmissions: true`). **CPU or GPU notebook ≤ 12 hours** (API: `maxCpuRuntimeMinutes`/`maxGpuRuntimeMinutes` = 720). **Upgraded accelerators: Kaggle L4x4 machines (4× NVIDIA L4, 96 GB total GPU memory)** are available, restricted to notebooks attached to this competition, consuming GPU quota at 2× the T4x2/P100 rate. The page still carries "NOTE: We are currently exploring options to provide better compute." Runtimes are obfuscated (±up to 10 min variance; `rerunMaxStaggerMinutes: 10`).
- **Internet: none during evaluation** — no API models (GPT/Claude/etc.); L4 sessions must have internet disabled even in development.
- **Submissions:** 1 per day (`maxDailySubmissions: 1`); you select 2 final submissions (`numScoredSubmissions: 2`); output file must be **`submission.json`** (≤20 GB submission size). **Format:** dict keyed by task_id → list (one entry per test input, in order) of `{"attempt_1": grid, "attempt_2": grid}`; both attempts required for every task_id; exact-match scoring — 1 if either attempt matches ground truth, averaged over all test outputs.
- **Evaluation sets:** notebooks are rerun against `arc-agi_test_challenges.json` containing **240 unseen private tasks** (the file visible pre-submission is a placeholder from the public eval set). Public leaderboard uses **50% of test data** (`leaderboardPercentage: 50`); final standings use the other 50%, withheld until verified.
- **Leaderboard (public, fetched 2026-08-08):** 1st **nvbanana 67.50**, 2nd rabbithole 47.78, 3rd Junhua Yang 37.22, then a tight pack ~33–34.6 (ranks 4–15). 1,398 teams, 7,309 joined users, 12,135 submissions. URL: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/leaderboard

## 3. Datasets

- **ARC-AGI-2:** https://github.com/arcprize/ARC-AGI-2 — cloned and verified: **1,000 public training tasks** (`data/training`, includes ARC-AGI-1 tasks + new) and **120 public evaluation tasks** (`data/evaluation`); one JSON per task with `"train"` (demo pairs) and `"test"` fields; grids are lists-of-lists of ints 0–9, 1x1 to 30x30. Two hidden sets exist: **semi-private** (for testing commercial APIs) and **fully private** (Kaggle competition), both difficulty-calibrated to the public eval. Kaggle mirrors the data as `arc-agi_{training,evaluation}-{challenges,solutions}.json` + `sample_submission.json` on the competition Data tab.
- **vs ARC-AGI-1** (https://github.com/fchollet/ARC-AGI, cloned: **400 training + 400 evaluation tasks**): same grid format/size range, but 2.5× more training tasks; ARC-AGI-2 (introduced March 2025) is substantially harder — average human performance on public eval was 66% in their test sample; every eval task solved by ≥2 humans within 2 attempts; designed to resist brute-force program search. Technical report: http://arcprize.org/blog/arc-agi-2-technical-report. Note the ARC-AGI-2 repo README says "3 trials" in one legacy paragraph, but the competition rule everywhere else is **2 attempts**.

## 4. Pretrained models / external data

Official Code Requirements: "External data, freely & publicly available, is allowed, **including pre-trained models**." Rules §2.6: external data/models must be publicly available and equally accessible to all participants at no/minimal cost, or satisfy "reasonableness" (a dataset costing more than a prize is unreasonable). No stated parameter-size cap — the practical cap is the 12h/96GB offline rerun. AMLT tools allowed with a compliant license. **Uploading fine-tuned checkpoints as Kaggle Datasets/Models and attaching them to your notebook is the standard, permitted pattern** (they must be public and open-sourced with your solution for prize eligibility — the mechanism itself is verified by the rules; the "standard pattern" characterization is from 2024/25 practice).

## 5. Entry checklist

1. Kaggle account + phone/identity verification.
2. Join (accept rules) at https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2 **before Oct 26, 2026**; free entry.
3. Clone data (GitHub repo above or Kaggle Data tab); read the ARC-AGI-1&2 technical guide linked from https://arcprize.org/competitions/2026/arc-agi-2 and play tasks at https://arcprize.org/play.
4. Fork a public notebook from the competition Code tab — since this is an explicit **relaunch of ARC Prize 2025** (https://www.kaggle.com/competitions/arc-prize-2025), 2025 solutions/notebooks (e.g., the ARChitects' winning open-source release) port directly. (I could not enumerate the Code tab anonymously — top-notebook names unverified.)
5. Commit notebook offline, ≤12h, emitting `submission.json`; submit (1/day); pick 2 finals before Nov 2, 2026 23:59 UTC.

Sources: [arcprize.org/competitions/2026](https://arcprize.org/competitions/2026), [arc-agi-2 track page](https://arcprize.org/competitions/2026/arc-agi-2), [arc-agi-3 track page](https://arcprize.org/competitions/2026/arc-agi-3), [paper prize](https://arcprize.org/competitions/2026/paper), [Kaggle ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2) (overview/rules/leaderboard via API, comp ID 133469), [Kaggle ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3), [Kaggle paper track](https://www.kaggle.com/competitions/arc-prize-2026-paper-track), [github.com/arcprize/ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2), [github.com/fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI). All accessed 2026-08-08 UTC.


## Correction (2026-08-08, learned from submission 55338854)
- The submission score comes from the submission.json produced by the
  kernel version's OWN (interactive) run — resolution took ~30 min, far
  too fast for any scoring rerun. The interactive environment IS the
  scoring environment; there is no separate L4x4 rerun for this track.
- The 2026 ARC-AGI-2 track offers T4 / P100 (per docs.arcprize.org
  build_notebook options: cpu, t4, p100; rtx6000 reserved for ARC-AGI-3).
  The L4x4 note earlier in this file was 2025-vintage.
- kernel-metadata.json accepts "machine_shape" (or `kaggle kernels push
  --accelerator`); default resolved to a non-bf16 GPU (P100-class) which
  hard-fails bf16 compute. Pin "NvidiaTeslaT4" — same GPU class as every
  Lightning validation run.
