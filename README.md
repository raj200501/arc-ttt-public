# arc-ttt

**Per-tenant small-model adaptation for document extraction, with
preregistered, reproducible evals — failures published beside passes.**
Every headline number reconciles to a machine-readable artifact
(`VERDICT.md` is the map). Preregistration ordering: the spec's later
gates (Addenda D/E/F onward) are Bitcoin-anchored via OpenTimestamps
before their data existed; the original gate's 2026-08-12 freeze
predates the first anchor and rests on git history — stated plainly,
and the gates from D onward are chain-anchored pre-data — with one
precision that belongs here rather than in a footnote: Addendum E's +5
bar and decision rule are in the chain-anchored snapshot, while the
E-r2 measurability amendment (the compacted geometry and the token
screen) was git-committed before its data and anchored the following
day. Verification
is one command:

    python3 scripts/verify_verdict.py
    python3 scripts/verify_from_primary.py experiments/novel_schema_f_*.json

**Latest gate (2026-08-20): Addendum E PASSED** — +40.4 F1 seed-mean
over six fresh shape-varying tenants against a +5 bar frozen before the
data existed, 340W/5L/15T over 360 paired documents, zero excluded.
Recompute it: `python3 scripts/addendum_e_summary.py`.

**In a hurry? [`EVIDENCE.md`](EVIDENCE.md) is the whole ladder on one
page** — five preregistered gates in the order they happened (two of
them failures), the comparison that cuts against the result, the cost
table stated against interest, and the one blind-holdout run, each
number carrying the caveat that makes it true.

The harness began as a test-time-training entry for ARC Prize 2026
(that origin, its scores, and its failure log are documented below —
nothing is scrubbed); the enterprise adaptation gates are the active
program.

**Status (2026-08-17): the preregistered k=30 adaptation gate decided GO —
mean +46.5 micro-F1 over 30-shot prompting across three novel-schema
seeds (+36.0 / +49.0 / +54.4 vs a +5 bar frozen 2026-08-12 before any
data; receipt-level sign test 156W/0L/2T over 158 scored of 180 designed (p < 1e-15); CI excludes zero;
`experiments/novel_schema_summary_2026-08-12.json`).** Stated per the
spec's claim rule, always beside the CORD negative: on CORD receipts —
a domain the base model already knows — the same adaptation recipe
FAILED its preregistered gates at all three scales tested (Addendum A:
−7.3 / −11.5 / −4.5 F1). Adaptation buys novelty, not general quality,
and we publish our negatives. Replication: 7 fresh tenants at k=10, pooled with
the 3 gate tenants: +41.5 pooled, 569W/1L (cuda/bf16 per B.8). The k=30 gate pairs ran
CPU/fp32 on free Kaggle kernels; artifacts carry full receipt trails,
including `resumed` stamps from the checkpoint/resume system that
survived repeated infrastructure kill-strikes during the gate.



**The protocol has run once, end to end (2026-08-20):** a labeled
DRESS REHEARSAL — an adversarial AI agent authored 50 out-of-
distribution waybills, kept its gold, and scored our blind single
submission once: 0.8792 mean micro-F1, 30/30 valid JSON, hard tier
0.679, failure taxonomy published (agent-authored corpus, NOT a real
tenant — the row in VERDICT.md carries the full label).
`experiments/blind_rehearsal_2026-08-20.json` has per-document scores.

## Check the preregistration ordering

**Preregistration you can check without trusting us:** the eval spec's
SHA-256 is anchored to the Bitcoin blockchain via OpenTimestamps —
proofs committed beside byte-exact snapshots of the revisions they
stamp (`docs/research/snapshots_ENTERPRISE_EVAL_SPEC_*.md` +
matching `.ots`). Verify with
`ots verify docs/research/ENTERPRISE_EVAL_SPEC.md.2026-08-19T0119Z.ots -f docs/research/snapshots_ENTERPRISE_EVAL_SPEC_2026-08-19T0119Z.md`
(attestations complete once the Bitcoin confirmation lands). The
Addendum D and E/F freezes therefore have independently checkable
ordering: bars first, data second.

## Verify the headline in 60 seconds

Don't trust our summary — recompute it. Zero dependencies:

```
python3 scripts/verify_verdict.py
```

It rebuilds every statistic of the k=30 gate from the raw per-receipt
records (per-arm means, paired deltas, sign test, receipt-level and
cluster-level CIs, validity windows, attrition) and cross-checks the
published summary, exiting nonzero on any mismatch.

The corpus generator itself is a public entry point — regenerate any
tenant's documents + gold deterministically and inspect them:

```
python3 scripts/export_novel_prompts.py --seed 1 --k 10 --limit 3 \
    --prompts-out /tmp/docs.jsonl --gold-out /tmp/gold.jsonl
```

To run the adaptation demo or any model-loading path (the verify and
scoring scripts above are stdlib-only and need none of this):

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
bash demo/run_endpoint_demo.sh   # ~6-10 min on a 4-core CPU box
``` To re-run the
underlying experiment itself, any free Kaggle account suffices — the
kernel entries under `kaggle/` are the exact runners that produced the
artifacts. If you find any claim in this README not backed by its cited
artifact, open an issue; we publish our corrections (spec B.9) with the
same prominence as our results.

## ARC Prize origin (documented history)

**Status (2026-08-15): 1.67 public ×3 — the recall-bound widening (v9)
and the DFS time-budget raise to 90 s/task (v10) were both clean
preregistered nulls; budget levers are exhausted, candidate-generation
quality is the binding constraint, and leaderboard climbing is formally
deprioritized (`experiments/kaggle_v10_scored_2026-08-15.json`).
First nonzero was 1.67 public on
ARC-AGI-2's hidden set** (v8; ~4/240 tasks at 150/240 real-prediction
coverage). The road there is documented failure by failure: v6 scored
0.00 (wrong-architecture GPU; accelerator pin via `machine_shape`, paper
§6.7), v7 scored 0.00 at 40/240 coverage (transformers cache-API
incident, fixed with explicit API probes + regression tests, paper
§6.8), v8 closed both and scored. Honest read: the pipeline is proven
end-to-end; per-attempt hit rate (~2.7%) makes solver quality the
binding constraint — a multi-week solver program, deprioritized per the
v10 verdict in favor of the enterprise gates and the paper track. 167 offline tests
pass. The full pipeline — augmentation sweep → per-task LoRA TTT →
constrained DFS decoding → invert → vote/rescore → submission — is
GPU-validated end-to-end with the 2025 champion's public 4B checkpoint.
Teacher-forced diagnostics show the checkpoint assigns per-token
probabilities of 85–97% (lp −0.16..−0.03) to true solutions under our
serialization; current iteration targets candidate recall (search) and TTT
sharpening. No claims beyond the artifacts in `experiments/`.

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

1. **Done:** Build the adaptation + eval harness and prove the loop
   end-to-end on subsidized compute (ARC track: first nonzero scored
   2026-08-10 after two published-postmortem 0.00s; leaderboard
   climbing since formally deprioritized — see the origin section).
2. **Done:** The preregistered enterprise ladder on the harness — the
   k=30 novelty gate (GO), the CORD negative (published), document-only
   serving (D FAIL → F PASS), measured cost rows.
3. **Now:** Real documents. Design-partner / blind-holdout runs on
   corpora we didn't generate (the anchored protocol in
   `docs/research/BLIND_HOLDOUT_PROTOCOL.md` is the standing offer;
   `scripts/make_challenge.py` is the challenger-side kit — it splits
   your labeled JSONL into a challenge package on your machine, so
   gold never leaves it, and later scores the single submission with
   the pinned scorer), plus the GPU serving crossover measurement and
   one scale rung up.
4. **Then:** Productize the per-tenant adapt-measure-verify loop with
   design partners; the ARC paper-track submission (due Nov 8)
   documents the harness lineage.

## Ground rules (carried over from prior work)

- **No fabricated results.** Scores exist only when a run produced them;
  every reported number links to its artifact in `experiments/`.
- **Fail-closed experiment discipline:** preregistered gates, pinned
  versions, machine-readable run records.
- **Honest framing:** this repo carries both the active adaptation
  program and the ARC Prize competition entry it grew out of; product-
  and company-level claims are made only where an artifact backs them —
  the status line above says exactly where things stand.

## Repository layout

- `src/arcttt/` — the harness: tasks, augmentations, serialization,
  pure-torch LoRA, TTT loop, constrained DFS, voting, solver.
- `tests/` — 167 offline tests (tiny in-test models; no downloads).
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
