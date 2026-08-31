# arc-ttt

**A markdown code fence makes OpenAI's `evals` steganography monitor
report that it detected nothing.**

`evals/elsuite/steganography/monitor.py:72` asks a monitor model whether a
response hides a payload and parses the answer with `json.loads`. If the
model wraps its JSON in a fence the parse fails, the loop retries with
**the same messages** up to five times, and the function returns
`("", 0.0)` — where `0.0` means *not suspicious*. A monitor that
correctly answered `1` is recorded as having found nothing, and the
sample scores in the steganographer's favour.

That prompt does say *"do not generate any other text"*, so a fenced
answer disobeyed it. The finding is about which way the default falls
when a model disobeys, not that the package is wrong.

## What one fence is worth

Same 30 held-out documents, same model, same prompt, same scorer. The
only difference between the columns is three backticks.

| Qwen2.5, field list only | fenced | as emitted | fence-stripped |
|---|---|---|---|
| 0.5B  | 30/30 | `0.0000` | **0.6350** |
| 1.5B  | 30/30 | `0.0000` | **0.7375** |
| 3B    | 30/30 | `0.0000` | **0.8958** |
| 0.5B, with 20 demonstrations | **0/30** | 0.7836 | 0.7836 |
| 1.5B, with 20 demonstrations | **0/30** | 0.8804 | 0.8804 |

Two things fall out, and the second is why this survives in shipped code.

**The tax grows with capability.** The better the model, the more the
wrap costs — it had more to lose. All three print as `0.0000`.

**Demonstrations suppress it.** Given twenty examples of bare JSON the
same checkpoints wrap nothing; given a bare field list they wrap
everything. The tax lands entirely on the *cheap* prompt regime — the one
you reach for to make a small model economical — and is invisible in the
expensive regime most harnesses run.

That second claim compares two prompts differing in three ways at once
(examples, ~4,300 tokens vs ~196, many turns vs one). **Addendum R** in
[`VERDICT.md`](VERDICT.md) is preregistered to find out which one is
doing the work, and will withdraw the sentence if it is not the examples.

## Is your evaluation doing this?

One file, standard library, no install, Python 3.9+:

```
python3 tools/fencecheck.py scan  path/to/your/repo
python3 tools/fencecheck.py score predictions.jsonl
```

`scan` reports code that parses model JSON with no fence handling **and**
turns the failure into a zero or a silent skip — only the conjunction,
because parsing strictly is not a defect on its own. `score` reads your
saved outputs and tells you how many are valid JSON your scorer rejects.
Exit status 1 on a finding, so it drops into CI. See
[`tools/README.md`](tools/README.md).

## How common is it?

**6 confirmed instances of 35 candidate sites, read by hand, across 34
published packages** — in `evals`, `instructor` and `ragas`. All 29
rejections are published with reasons in
[`experiments/fence_census_hand.json`](experiments/fence_census_hand.json):
most candidates turned out to be protocol frames, telemetry, HTTP
bodies, credentials, ground-truth strings, or places where the fence is
handled a level up.

**Quote 6 of 35, not a rate over the ecosystem.** This is a lower bound
among the packages searched, and it is not a claim that any package is
buggy. Two mechanical classifiers failed their own preregistered accuracy
gates before this one was done by hand (60%, then 70%, against an 80%
floor); both withheld tallies stay published.

## Why this repository found it

Everything above came out of an eval harness built to test a different
idea, which it then killed.

The thesis was that per-tenant adaptation makes a 0.5B good enough for
private document extraction. Under gates frozen before each arm ran: a
hosted API beat the adapted model outright; a bare field list beat the
cost argument; document grounding moved 0.8833 to 0.8833; and a
Qwen2.5-3B with **no adaptation** and a 196-token prompt matched it at
0.8958 against 0.8833. [`VERDICT.md`](VERDICT.md) says so in the words
frozen beforehand, and [`CORRECTIONS.md`](CORRECTIONS.md) carries every
withdrawal with a date.

The 3B arm that ended it first reported `0.0000`. That is where the fence
came from.

---

# The adaptation work, in full

Everything below is the original project, unedited except where a later
result withdrew a claim. It is kept complete because the withdrawals are
part of the record, and because the harness described here is what found
the fence.

**Adapt a small open model (Qwen2.5-0.5B) to one tenant's document
schema, and measure whether it beat that same model's own prompt —
under gates frozen before the data existed, with the failures published
beside the passes.**

Five preregistered gates, two of them FAIL. Every headline number
reconciles to a machine-readable artifact; [`VERDICT.md`](VERDICT.md) is
the map and [`EVIDENCE.md`](EVIDENCE.md) is the whole ladder on one
page. Two commands check the arithmetic against the raw records, with
nothing installed:

    python3 scripts/verify_verdict.py
    python3 scripts/verify_from_primary.py experiments/novel_schema_f_*.json

**What this measured, and it is the strongest thing here (gate 5, Addendum E).**
On **six tenant schemas the base model has never seen**, per-tenant weight
adaptation beats that same model's own 30-shot prompt by **+40.4 micro-F1**
(0.5726 → 0.9761) across **360 paired documents** — 340W/5L/15T, cluster
CI95 [+31.0, +49.7] — against a **+5 bar frozen before the data existed**,
with every arm re-scored from its raw predictions and **all 6 of 6 tenants
clearing the rule individually**. **And what the prompt gets wrong is not
format.** Its JSON is valid **360/360**, its key-path precision and recall
are both **1.00**, and its key set equals gold's on 60/60 documents in every
tenant — so rebuilding its output under gold's key paths, which is what a
schema-constrained decoder guarantees for free (plus a second oracle more
generous than any decoder that exists), closes **0.0% of that gap**. The
cheapest rival explanation for the headline is measured, not argued, and it
explains none of it. The effect also **survives the strongest cheap control
anyone proposed** — making the JSON key the document's own label, documents
byte-identical across arms — at **+18.75** (40W/2L/18T, p=2.1e-10).

**Read this next, because it is the ceiling on all of the above (2026-08-22).**
On the only realistic corpus here — 30 held-out freight waybills, gold
published — **the cheapest hosted API tier scores 0.9708–1.0000 across four
runs of the same arm** (mean 0.9865, zero invalid JSON every time; hosted
inference is not deterministic at temperature 0, so one run is a sample, not
a measurement). Our adapted 0.5B scores 0.8833 on the same documents and wins
**at most one of thirty in any run** (0W/14L/16T to 1W/13L/16T). That
comparison was preregistered with its readings frozen first, and it landed on
the one that costs the most: **on documents that can be sent to a hosted API,
this loses on accuracy, and nothing else in this repository changes that.**
**On cost the claim is worse for us and is withdrawn.** This page said $0.89
ours against $1.55 theirs and called it a trade; the $1.55 assumed the hosted
arm must carry all 20 demonstrations in every request, and Addendum J refutes
that from our own artifacts. **With no demonstrations at all — just the
tenant's field list declared, a 25x smaller payload — it scores 0.8930 for
about $0.36 per 1,000 documents, above our adapted 0.8833 at our measured
$0.89.** Give it two demonstrations and it reaches 0.9722 for ~$0.40. **Our own
side then turned out to be 3.7x too expensive for a reason that was
ours:** we had measured it serving one document at a time. At batch 16
the same adapter serves the same 30 documents for **$0.1406 per 1,000 at
an unchanged 0.8833** — every batched prediction byte-identical to the
batch-1 run. So the standing comparison is **2.6x cheaper and 0.0097
worse**, for batch workloads only (per-document latency gets 4x worse),
with both sides priced cold. A real trade, not a win, and not the cost
story this repo used to tell. **And the gap is
not an implementation artifact** — Addendum K built document grounding to
close it, froze the recipe on the training split, scored the holdout once,
and got 0.8833 again; only **4 of our 28 field errors are reachable by
document grounding at all**, while 6–7 of the hosted arm's 6–7 are. What
survives is narrower and is now the whole claim — **workloads where a hosted
API is not an option**, where the question is whether an on-prem small model
is good enough for one tenant's schema, and where this comparison is not
available at any price. Everything here should be read as being about that
segment and no other.

**2026-08-25 — THAT SEGMENT NO LONGER SUPPORTS THE CLAIM EITHER (Addendum O, preregistered, reading (a)).** The paragraph above narrowed this project to *workloads where a hosted API is not an option* and said adaptation is what makes a 0.5B viable there. **It is not.** `Qwen2.5-3B-Instruct`, an open checkpoint any on-prem buyer can run, with **no adaptation of any kind** and only the tenant's field list in a 196-token prompt, scores **0.8958 with 0 of 30 invalid** against our adapted 0.5B's 0.8833 — 8W/8L/14T, p=0.60, a dead heat. Reading (a) publishes in the words frozen before the arm ran: *a 3B needs neither our adaptation nor a large prompt, and the on-prem cost claim is dead, not narrowed.*

**And the result that had been holding this up was a markdown fence.** Addendum N published "given only a field list, a 1.5B does not produce a usable object at all — 0.0000, 30 of 30 invalid" and was cited as the asymmetry that made adaptation earn its keep on-prem. A reproduction run reproduced that number exactly, stored the predictions this time, and they are correct extractions inside fenced json code blocks. Fence-stripped: **0.7375, zero invalid.** N is withdrawn. This page had already met fenced JSON on the hosted arm one paragraph below, handled it correctly, and written down why — and the scale-rung runner written three days later did not inherit that rule. The un-repaired zero was the number that flattered us.

**What is left, and it is all that is left:** matching our quality with an unadapted open model takes a 3B, about 6x the parameters, which on the same CPU box costs about **$2.97 per 1,000 documents against our $0.5143 at the same batch size**. Cost at fixed quality — real, measured, much weaker than what this page claimed yesterday, and **not yet measured at matched batching**, which is the first thing a technical reader should ask and which we have not run. See `VERDICT.md` Addenda N and O and `experiments/fence_rescore.json`.

`experiments/waybill_market_baseline_gemini-3.5-flash-lite_matchedturns_2026-08-22.json` is the citable run — its demonstrations mirror our own arm's turn structure exactly and it needed no output repair. The packed-turn run beside it reaches the same numbers only after stripping a markdown fence from four outputs, a repair our own arms never got; without it that run scores 0.8667, *below* our adapted arm. Both are banked, and the difference is stated because an outside reviewer found it.

**And the corpus's own difficulty is now priced (Addendum H, preregistered, decided 2026-08-22).** Making the JSON key the document's own label — what real tenant schemas look like — while changing nothing else takes the delta from **+41.1 to +18.75** (40W/2L/18T, p=2.1e-10). The effect **survives** the strongest cheap control proposed against it, and **the arbitrary mapping was worth +22.3 F1, more than half of it.** Removing the distractor lines costs another +12.3. Read with the waybills (+9.97, +4.14 granting nothing) and CORD (negative), the series says the same thing each time: the more a corpus looks like a customer's, the smaller the effect. Expect the bottom of that range, not the top.

**What it buys, and where it does not.** On the six novel-schema tenants
whose baseline predictions are stored, the 0.5B baseline emits perfect
JSON in exactly the right keys — key-path precision and recall both
**1.00** across all 360 documents — and fills **43% of the fields with
the wrong content** (six-tenant mean; 27%–51% across tenants). Adaptation
fixes the content, and a schema-constrained decoder — the obvious free rival —
closes **0%** of that gap. On a realistic freight-waybill corpus, where
a 20-shot prompt already scores 0.78, the paired test **FAILED** our own
bar and what remains is mostly output format, which that same free
decoder plausibly takes. Both mechanisms were measured against our own
interest and both ship with the scripts that produce them. There are
**zero customers** and every corpus here is synthetic or agent-authored.

[![verify](https://github.com/raj200501/arc-ttt-public/actions/workflows/tests.yml/badge.svg)](https://github.com/raj200501/arc-ttt-public/actions/workflows/tests.yml)

On every push, a GitHub-hosted runner — not this laptop — recomputes the gate-1,
gate-3 and gate-5 verdicts from the per-receipt artifacts and re-scores a
gate-4 artifact against regenerated gold. (The CORD negative in gate 2
has no recompute step — stated so the badge does not imply more coverage
than it has.) The log is public and needs nothing
installed. **This is not an independent replication:** our code, our
artifacts, our workflow. It audits the arithmetic and the scoring, not
the data distribution — that is what the blind-holdout offer is for.
Mutation tests (`tests/test_verify_scripts.py`) pin that the check can
actually fail.

**Preregistration ordering, stated precisely rather than implied.** The
spec's later gates (Addenda D/E/F onward) are chain-anchored
(OpenTimestamps, Bitcoin) before their data existed. The original gate's
2026-08-12 freeze predates the first anchor and rests on git history we
control. One further precision: Addendum E's +5 bar and decision rule
are in the chain-anchored snapshot, while its E-r2 measurability
amendment (the compacted geometry and the token screen) was
git-committed before its data and anchored the following day.

**Latest gate (2026-08-20): Addendum E PASSED** — +40.4 F1 seed-mean
over six fresh shape-varying tenants against a +5 bar frozen before the
data existed, 340W/5L/15T over 360 paired documents, zero excluded.
Recompute it: `python3 scripts/addendum_e_summary.py`.

**Every number we have withdrawn or revised is on one page:**
[`CORRECTIONS.md`](CORRECTIONS.md) — including the ones that moved a
headline, and the defects we found in our own verification path.

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
the 3 gate tenants: +41.5 pooled, 569W/1L/30T over 600 (cuda/bf16 per B.8). The k=30 gate pairs ran
CPU/fp32 on free Kaggle kernels; artifacts carry full receipt trails,
including `resumed` stamps from the checkpoint/resume system that
survived repeated infrastructure kill-strikes during the gate.

**The protocol has run once, end to end (2026-08-20):** a labeled
DRESS REHEARSAL — an adversarial AI agent WE RAN OURSELVES, in the same
working session on the same host, authored 50 out-of-distribution waybills,
withheld its gold from us, and scored our blind single submission
once: 0.8792 mean micro-F1, 30/30 valid JSON, hard tier
0.679, failure taxonomy published (agent-authored corpus, NOT a real
tenant — the row in VERDICT.md carries the full label).
`experiments/blind_rehearsal_2026-08-20.json` has per-document scores.

**And the paired baseline that number was missing has now been run —
it FAILED (2026-08-21).** Matched greedy decode, same 30 documents:
prompted 0.7836, adapted 0.8833, delta +9.97 — twice the +5 bar — but
the sign test disagreed (8W/5L/17T, p=0.29) and the preregistered rule
needs both, so it publishes as a failure. 17 of 30 documents tie, and
63% of the delta comes from 3 documents where the prompted arm emitted
invalid JSON. On this realistic corpus the measured benefit is
**output-format reliability, not extraction accuracy** — narrower than
the headline gates support. Read 0.8792 as an adapted score, not as
evidence that adapting beat prompting there.
`experiments/blind_rehearsal_baseline_2026-08-21.json`.

**And the rival explanation for *that* has now been measured, and this
corpus cannot rule it out (2026-08-22).** If the benefit is output format,
then a JSON-grammar-constrained decoder buys it for free — no
adaptation, no per-tenant weights, no training. `python3
scripts/format_counterfactual.py` bounds that from the published raw
arms, stdlib-only: delete the prompted arm's 12 off-schema keys (which
a schema-constrained decoder cannot emit) and the delta falls +9.97 →
+9.26; restrict to the **27 documents both arms parsed**, where format
cannot be the explanation because neither arm failed format, and it is
**+4.14 with a 5W/5L/17T sign test** granting nothing at all (+3.34 if the baseline is also key-pruned) — an interval of [−3.5, +11.8] at n=27, which supports neither explanation — under our own +5 bar and
directionally a coin flip; grant an impossible fixer that makes the
prompted arm perfect on the 3 it could not parse and it is **−0.74**.
Post-hoc analysis, not a gate, and labeled as such in the artifact. It
leaves gates 1/4/5 untouched — the prompted baseline there is
0.4333–0.6208 and format failures are not what separate those arms —
and it means the honest v1 ships constrained decoding with the
adaptation layer measured **on top of** it. That comparison is now a
PENDING row in VERDICT.md with its bar and all three readings frozen
before the arms exist, including the one that says there is no product.
`experiments/format_counterfactual_2026-08-22.json`.

**And that objection was then pushed onto the headline gates, where it
does not land — measured, not asserted.** Across both arms, invalid JSON
is **0 of 158** on gate 1, **0 of 360** on gate 5, and **1 of 158** on
gate 4 (on our arm), so restricting to documents both arms parsed changes
nothing: +46.5 → +46.5, +40.4 → +40.4, +24.0 → +24.5. Valid JSON is not
the same as the right keys, so gate 5's baseline was also rebuilt under
gold's key paths keeping its own values — what a schema-constrained
decoder guarantees — and again forgiving nesting mistakes. **Neither
repair moves it: 0.5726 → 0.5726 → 0.5726 against an adapted 0.9761, 0%
of the gap closed**, because the baseline's key-path precision and recall
are already 1.00 on all 360 documents and it simply puts the wrong
content in 43% of the fields it names correctly — a six-tenant mean over a **27%–51%** spread, and arithmetically 1 − the baseline's 0.5726 rather than an independent measurement. On the headline gates
adaptation is buying value-level extraction accuracy, which no decoder
supplies. `PYTHONPATH=src python3
scripts/schema_conformance_decomposition.py`,
`experiments/schema_conformance_decomposition_2026-08-22.json`.

**Look at it rather than run it.** `demo/waybill_field_audit.html` is a
single self-contained page — open it in a browser, nothing to install —
showing all 30 held-out documents field by field: the source document,
what each arm returned, and which of the tenant's eight fields each one
got. Every value in it is the real recorded prediction; the page is
regenerated from the artifacts by `scripts/build_field_audit.py`, so it
is a view of the evidence rather than a drawing of it. The ties and the
five losses are in the same rail as the wins, and it opens on the FAIL
verdict. Start with document `m-2201`: the prompted model reads the
waybill correctly and then answers in a schema it invented, while the
adapted model returns the tenant's fields. That is the clearest picture
of what this actually buys, and of what it does not.

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

## Run it on your own documents

Recomputing our numbers only proves our arithmetic. This points the same
machinery at YOUR data: one JSONL of labeled documents (`{"id", "text",
"gold"}` per line), one command, and you get the paired comparison every
gate row is built on — the prompted baseline against the adapted model,
matched decode, scored per document against your own gold with the
pinned scorer.

```
python3 scripts/try_your_documents.py --docs mydocs.jsonl
```

Stated by the script itself, beside its own result: it is **not blind**
(it reads your gold to score), it fixes **no bar in advance** (you can
re-run until you like the number — so could we), and it is one corpus,
one schema, one run. If your prompted baseline is already saturated it
says outright that there is no headroom and nothing here for you to buy.
The version that is *evidence* is the blind-holdout offer in
[`CHALLENGES.md`](CHALLENGES.md).

To run that, the adaptation demo, or any model-loading path (the verify
and scoring scripts above are stdlib-only and need none of this):

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
bash demo/run_endpoint_demo.sh   # ~6-10 min on a 4-core CPU box
```

To re-run the
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
v10 verdict in favor of the enterprise gates and the paper track. 341 offline tests
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
- `tests/` — 341 offline tests (tiny in-test models; no downloads).
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
