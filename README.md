# arc-ttt

Test-time training (TTT) lab targeting **ARC Prize 2026** — the ARC-AGI-2
Kaggle track, where per-instance test-time adaptation remains the dominant
approach — plus cost-vs-accuracy TTT curves on enterprise-shaped tail tasks.

**Status (2026-08-17): the preregistered k=30 adaptation gate decided GO —
mean +46.5 micro-F1 over 30-shot prompting across three novel-schema
seeds (+36.0 / +49.0 / +54.4 vs a +5 bar frozen 2026-08-12 before any
data; receipt-level sign test 156W/0L (p < 1e-15); CI excludes zero;
`experiments/novel_schema_summary_2026-08-12.json`).** Stated per the
spec's claim rule, always beside the CORD negative: on CORD receipts —
a domain the base model already knows — the same adaptation recipe
FAILED its preregistered gates at all three scales tested (Addendum A:
−7.3 / −11.5 / −4.5 F1). Adaptation buys novelty, not general quality,
and we publish our negatives. Replication: 10/10 fresh tenants at k=10,
+41.5 pooled (569W/1L; cuda/bf16 per B.8). The k=30 gate pairs ran
CPU/fp32 on free Kaggle kernels; artifacts carry full receipt trails,
including `resumed` stamps from the checkpoint/resume system that
survived repeated infrastructure kill-strikes during the gate.

**Status (2026-08-12): 1.67 public, unchanged by v9 — the recall-bound
widening was a clean preregistered null; next lever is the DFS time
budget. First nonzero was 1.67 public on
ARC-AGI-2's hidden set** (v8; ~4/240 tasks at 150/240 real-prediction
coverage). The road there is documented failure by failure: v6 scored
0.00 (wrong-architecture GPU; accelerator pin via `machine_shape`, paper
§6.7), v7 scored 0.00 at 40/240 coverage (transformers cache-API
incident, fixed with explicit API probes + regression tests, paper
§6.8), v8 closed both and scored. Honest read: the pipeline is proven
end-to-end; per-attempt hit rate (~2.7%) makes solver quality the
binding constraint, and that is the current work. 83 offline tests
pass. The full pipeline — augmentation sweep → per-task LoRA TTT →
constrained DFS decoding → invert → vote/rescore → submission — is
GPU-validated end-to-end with the 2025 champion's public 4B checkpoint.
Teacher-forced diagnostics show the checkpoint assigns per-token
probabilities of 85–97% (lp −0.16..−0.03) to true solutions under our
serialization; current iteration targets candidate recall (search) and TTT
sharpening. No claims beyond the artifacts in `experiments/`.

## Verify the headline in 60 seconds

Don't trust our summary — recompute it. Zero dependencies:

```
python3 scripts/verify_verdict.py
```

It rebuilds every statistic of the k=30 gate from the raw per-receipt
records (per-arm means, paired deltas, sign test, receipt-level and
cluster-level CIs, validity windows, attrition) and cross-checks the
published summary, exiting nonzero on any mismatch. To re-run the
underlying experiment itself, any free Kaggle account suffices — the
kernel entries under `kaggle/` are the exact runners that produced the
artifacts. If you find any claim in this README not backed by its cited
artifact, open an issue; we publish our corrections (spec B.9) with the
same prominence as our results.

## What's in the harness

- **Clean-room reproduction** of the NVARC 2025 winning recipe from its
  public writeups (their repo has no license; no code copied): raw Qwen
  16-token serialization, per-task LoRA (pure torch — scoring images lack
  peft), rslora, leave-one-out TTT corpora, dihedral x color-permutation
  augmentations with exact inverses, cumulative-NLL-bounded DFS decoding,
  count+likelihood candidate selection.
- **Oracle-tested decoding:** the DFS is verified token-exact against a
  cache-free brute-force enumerator (this test caught a KV-cache aliasing
  bug that silently degrades naive implementations to near-greedy search).
- **Deterministic kernel builds:** `kaggle/build_bundle.py` compiles
  `src/arcttt` + an entry into the single-file Kaggle kernel; multi-GPU
  task sharding with atomic per-task checkpointing and graceful
  degradation on non-bf16 GPUs.

## The plan (short version)

1. **Step 0 (~$0):** Reproduce the open NVARC/TTT baseline lineage on the
   ARC-AGI-2 Kaggle track using subsidized Kaggle compute. ✅ done:
   first nonzero scored 2026-08-10 (1.67 public, v8) after two
   published-postmortem 0.00s.
2. **Step 1 (3–6 months):** Iterate toward a genuine leaderboard placement;
   publish a cost-vs-accuracy TTT curve on one enterprise-shaped tail task.
3. **Step 2:** The placement, the cost-vs-accuracy curve, the open repo,
   and the ARC paper-track submission (due Nov 8) together establish a
   verifiable public record of the capability.
4. **Steps 3–4:** Productize the adaptation layer with design partners,
   then scale it.

## Ground rules (carried over from prior work)

- **No fabricated results.** Scores exist only when a run produced them;
  every reported number links to its artifact in `experiments/`.
- **Fail-closed experiment discipline:** preregistered gates, pinned
  versions, machine-readable run records.
- **Honest framing:** this repo is the competition entry and the research
  lab; product- and company-level claims are made only where an artifact
  backs them — the status line above says exactly where things stand.

## Repository layout

- `src/arcttt/` — the harness: tasks, augmentations, serialization,
  pure-torch LoRA, TTT loop, constrained DFS, voting, solver.
- `tests/` — 83 offline tests (tiny in-test models; no downloads).
- `experiments/` — machine-readable run records + the registry README.
- `kaggle/` — bundle builder, kernel entries, kernel metadata.
- `demo/` — the CORD-receipt adaptation demo: endpoint script, captured
  transcript, rendered before/after page.
- `scripts/` — dataset fetch and eval helper scripts.
- `paper/` — ARC paper-track outline (reproduction + ablations + costs).
- `docs/research/` — competition mechanics and recipe notes (web-verified).

## License

MIT (see `LICENSE`). ARC Prize rules require winning solutions to be
open-sourced; building in the open is the plan.
