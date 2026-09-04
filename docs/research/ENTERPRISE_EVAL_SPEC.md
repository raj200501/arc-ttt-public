# ENTERPRISE_EVAL_SPEC — First Enterprise-Shaped Evaluation (Design)

*Status: DESIGN — no run has happened; every number below is either a cited
external fact or a labeled estimate. Written 2026-08-08 in response to
REDTEAM.md Q3 ("zero quality evidence on any enterprise-shaped task") and the
FINAL_PICK.md Step-1 deliverable ("cost-vs-accuracy curve on one
enterprise-shaped tail task"). Rules of this document: every dataset claim
carries a source URL; nothing here claims transfer from ARC — transfer is
exactly the hypothesis this eval exists to test.*

---

## 0. What this eval must and must not do

**Must:** produce one honest, preregistered, artifact-backed point (then curve)
answering: *on a narrow repeated structured task with 5–30 demonstrations, does
per-task LoRA TTT on a small open model close the quality gap to a few-shot
frontier API — and at what adaptation cost?*

**Must not:** borrow credibility from ARC results, run on the unlicensed NVARC
checkpoint (see §2.4), or present a grid-cosplay of a text task as "enterprise
evidence."

**The honest architectural admission up front:** the current harness is
grid-native end to end. `tasks.py` hard-codes `Grid = tuple[tuple[int,...],...]`,
10 colors, a 30×30 cap; `serialize.py` emits digit rows; `decode.py` builds a
16-token grid vocabulary for constrained DFS; `augment.py` is dihedral × color
permutations. A text extraction task does not fit this harness today. Section 2
analyzes the two ways to close that gap and recommends one.

---

## 1. Task selection

### 1.1 What "enterprise-shaped" means here (from FINAL_PICK.md Step 1)

Rare-schema document extraction: a repeated, narrow task where a customer has a
handful of labeled examples (5–30), the schema is fixed, and the same endpoint
is then hit thousands of times. The eval task must therefore support: (a) a
fixed output schema, (b) few-shot adaptation sets sampled at k ∈ {5, 10, 30},
(c) a held-out test set large enough for a stable metric, (d) a license clean
enough to put in a public artifact attached to a commercial pitch.

### 1.2 Finalist A — CORD (receipt → structured JSON). **THE PICK.**

- **What:** Consolidated Receipt Dataset for post-OCR parsing (Indonesian
  receipts). Each receipt has OCR annotations (per-word text + boxes), 30
  semantic classes in 5 superclasses (`menu`, `void_menu`, `subtotal`,
  `void_total`, `total`), and a ground-truth parse (`gt_parse`) — i.e. the
  dataset ships both the input text (OCR words, so we need no OCR engine) and
  the target JSON.
  Source: https://github.com/clovaai/cord (README) and
  https://huggingface.co/datasets/naver-clova-ix/cord-v2
- **License:** CC BY 4.0 — "This work is licensed under a Creative Commons
  Attribution 4.0 International License" (verified 2026-08-08 at
  https://github.com/clovaai/cord/blob/master/README.md). Commercial use OK
  with attribution. This is the cleanest license among the finalists.
- **Size:** 800 train / 100 dev / 100 test released (of 1,000 annotated;
  an 11,000+ superset is described in the paper but not fully released).
  Paper: https://openreview.net/forum?id=SJl3z659UH
- **Why it maps to the "5–30 examples → adapted endpoint" story:** it *is* the
  story — one fixed schema, structured output, plenty of held-out instances.
  We sample k ∈ {5,10,30} adaptation receipts from train and treat the schema
  as one "task instance" (see §3.1). Text-only post-OCR parsing is also the
  dataset's original stated purpose, so we are not abusing it.
- **Frontier-API baseline cost (rough token math, labeled estimate):**
  OCR text per receipt ≈ 300–800 tokens (call it ~500); output JSON ≈ 200–400
  tokens (~300). A k=10 few-shot prompt ≈ 10×800 + 500 instructions + 399 test
  input ≈ **~9k input / ~0.3k output tokens per call**; at k=5 ≈ ~5.5k input.
  At Claude Sonnet 5 pricing ($3/M in, $15/M out —
  https://platform.claude.com/docs/en/pricing) that is **≈ $0.02–0.03 per
  instance**; Opus 5 ($5/$25) ≈ $0.035–0.05. A full 100-receipt test sweep ≈
  0.6–1.0M input + 30k output tokens ≈ **$2–5 per config per model** (half
  that with the Batches API). The whole baseline grid (2 models × 3 k-values)
  is **under $30**. Notably, the per-instance figure independently lands
  inside an assumed $0.02–0.05 per-instance frontier range (an
  assumption, not a measurement) — this run
  would replace that assumption with a measured number.

### 1.3 Finalist B — SROIE (ICDAR 2019 Task 3, receipt key-field extraction). Backup.

- **What:** scanned receipts with OCR transcriptions; extract exactly 4 fields
  (company, date, address, total). Source:
  https://rrc.cvc.uab.es/?ch=13 (official) and the paper
  https://arxiv.org/abs/2103.10213
- **Size:** 626 train / 347 test images with annotations.
- **License:** the weak point. The official RRC portal requires registration
  and does not publish a clean redistribution license; community mirrors
  (e.g. https://github.com/zzzDavid/ICDAR-2019-SROIE, and HF mirrors such as
  https://huggingface.co/datasets/darentang/sroie) claim CC BY 4.0 on their
  redistributed annotations, but the provenance chain is murkier than CORD's
  first-party statement. Given REDTEAM Q5 already flags us for building on
  unclear rights, do not make this the headline dataset.
- **Mapping:** excellent shape (fixed 4-field schema, big test set, short
  outputs), arguably *too easy* — 4 flat fields may saturate for both the
  adapted model and the frontier baseline, compressing the very gap we want
  to measure. Use as a cheap secondary task if CORD results warrant a second
  point.
- **Frontier baseline cost (estimate):** ~2.5–4k input / ~60 output tokens per
  call at k=10 (receipts ~250–400 tokens, outputs tiny) → ≈ **$0.01/instance**
  (Sonnet 5); full 347-receipt sweep ≈ **$3–5**.

### 1.4 Finalist C — WikiTableQuestions (table QA). Rejected for round 1.

- **What:** 22,033 natural-language questions over 2,108 Wikipedia tables.
  Source: https://github.com/ppasupat/WikiTableQuestions ; paper
  https://arxiv.org/abs/1508.00305 ; HF mirror
  https://huggingface.co/datasets/wikitablequestions
- **License:** CC BY-SA 4.0 (per the GitHub repo / HF dataset card). Clean,
  but ShareAlike — fine for an eval artifact.
- **Why rejected:** it breaks the wedge's shape. Questions are heterogeneous
  free-form QA, not a repeated fixed-schema transform; "5–30 examples then
  thousands of identical-schema instances" has no natural analog (per-table
  question counts are ~10, and the *skill* varies per question, not per
  table). A win here would not support the IDP story, and a loss would not
  refute it. Keep on the shelf for a later "structured transform" eval.
- **Cost if ever run (estimate):** tables ~300–2,000 tokens; k=10 prompt
  ≈ 5–15k input / ~30 output tokens ≈ $0.02–0.05/instance (Sonnet 5).

### 1.5 Considered and rejected outright

- **FUNSD** (form understanding, 199 documents): license is explicitly
  "solely for non-commercial, research and educational purposes"
  (https://guillaumejaume.github.io/FUNSD/work/) — disqualifying for an
  artifact attached to a commercial pitch, and 199 docs is small anyway.
- **DocVQA**: registration-gated distribution, and free-form visual QA breaks
  the fixed-schema story the same way WikiTableQuestions does
  (https://www.docvqa.org/).
- **Synthetic JSON schema mapping** (self-generated): infinitely available and
  perfectly shaped, but self-authored benchmarks attached to a pitch invite
  exactly the skepticism REDTEAM Q3 models. Public data first; synthetic only
  as a supplementary ablation, clearly labeled.

---

## 2. The architectural decision

### 2.1 Option (a): grid-ify a structured task into the existing ARC harness

Encode text/fields into 10-color grids so the current pipeline runs unchanged.

**What survives:** everything — serialization, DFS with the 16-token grid
vocabulary, dihedral × palette augmentations, voting, the kernel bundle.

**What is lost:** the task itself. A receipt's OCR text cannot be represented
in a ≤30×30 grid over 10 symbols without destroying its content (a 30×30 grid
holds 900 symbols of a 10-symbol alphabet; a receipt is hundreds of tokens
over a full vocabulary). Any encoding contrived to fit (hashing chars to
colors, laying out field positions spatially) yields a synthetic puzzle whose
solution demonstrates nothing about document extraction. Dihedral augmentation
of such a grid is semantically meaningless. The result would be a
manufactured "enterprise" number — precisely the kind of claim AGENTS.md's
integrity rules exist to prevent, and REDTEAM's partner would shred it in one
question ("show me the receipt your grid encodes").

**Verdict: rejected.** Not an engineering shortcut; an honesty failure.

### 2.2 Option (b): a text-mode TTT path. **RECOMMENDED.**

Same recipe shape — per-task LoRA on a small instruct model, leave-one-out
corpus construction, multi-sample generation, count+likelihood
voting/rescoring — with text (OCR lines in, canonical JSON out) instead of
grids, and without the grid-specific pieces (digit serialization, grid-vocab
DFS, dihedral/palette augmentations).

Module-by-module reuse audit (engineering estimates, not commitments):

| Module | Grid coupling | Reuse in text mode | Est. hours |
|---|---|---|---|
| `lora.py` | none — wraps any `nn.Linear` | unchanged | 0 |
| `model.py` TTT loop (`adapt`: encoding, label masking to final assistant turn, length-sorted padded batching, grad checkpointing, per-task inject/remove) | only via `Grid`/`grid_to_text` at call sites | reuse with a generic example type; the `ChatTurn` machinery is already text | 4–8 |
| `serialize.py` `ttt_training_examples` (leave-one-out + order shuffle) | typed on `Task`/`Pair` | generalize over `(input_text, output_text)` pairs; logic identical | 2–4 |
| `vote.py` (pool → rescore → select) | `Grid` used only as a hashable key; `Augmentation.invert` | key on canonicalized JSON string (hashable); identity/permutation "augmentations" with exact inverses (see below) | 2–4 |
| `model.py` `log_probabilities*` (batched candidate rescoring) | via `grid_to_text` in turn construction | reuse with text serializer | 1–2 |
| `decode.py` constrained DFS + grid vocab | total — 16-token vocabulary assumption | **not reused** in v1 (greedy + sampled generation instead); a JSON-grammar-constrained decoder is a possible v2 (est. 15–25 h, not needed for the first point) | 0 (v1) |
| `augment.py` dihedral × palette | total | **not reused.** Text analogs with exact inverses: demonstration-order shuffles (already in the loop via `shuffle_seed`) and JSON key-order permutation (invert = canonicalize). This is the single biggest recipe loss vs ARC — say so in the writeup | 2–4 |
| `tasks.py` schema/scoring | total | new `TextTask` loader (JSONL) + field-level scorer (§3.2) | 4–6 |
| `cli.py` / `serve.py` | via task schema | add a `--mode text` path; the `/adapt` endpoint story ("POST examples, get adapted predictions") carries over directly | 3–5 |
| CORD data prep (HF/parquet → OCR-line text + canonical `gt_parse` JSON, k-shot samplers, pinned revision + hashes) | n/a | new | 6–8 |
| Experiment runner + machine-readable artifacts (match `experiments/` discipline) | n/a | new | 3–5 |

**Total: roughly 27–46 hours** of engineering before the first GPU run —
comparable to what the registry shows this repo shipping in a day or two, and
all of it off the competition's critical path (none of it touches the kernel).

### 2.3 Recommendation and reasoning

**Option (b).** It tests the actual thesis (the recipe's *shape* — per-task
LoRA + LOO corpora + vote/rescore — transfers; not that receipts are secretly
ARC grids), reuses the genuinely portable 60% of the harness, and produces an
artifact a diligent skeptic can poke without it collapsing. The known losses —
no rich augmentation group, no constrained decoding in v1 — are stated
limitations, not hidden ones, and both have text-mode successors if the first
point justifies them.

### 2.4 Base-model licensing constraint (REDTEAM Q5 fallout)

The champion 4B NVARC checkpoint has **no license** (OWN_MODEL_PLAN.md;
AGENTS.md clean-room rule). Competition rules permit its use *in competition*;
an enterprise eval attached to a pitch is not in competition. **The enterprise
eval runs only on cleanly licensed bases:** verified per-model, not
per-family — Qwen2.5-Instruct 0.5B and 1.5B are Apache 2.0
(https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct); Qwen2.5-3B-Instruct
is NOT (Qwen Research License, `license:other`) and is excluded
[correction dated 2026-08-08; original text claimed the family
blanket-Apache]. Ladder: 0.5B (already pipeline-validated on T4 in this
repo, per `experiments/t4_smoke_2026-08-08.json`), 1.5B, and
Qwen3-4B-Instruct-2507 (verified apache-2.0) as the curve budget allows
(frozen in Addendum A). This also makes the eval a rehearsal for the owned-weights product
path rather than another dependency on legal fog.

*Forward pointer (added 2026-08-08): the scaled-run model ladder — including a
license correction to this section's "family" claim (Qwen2.5-**3B**-Instruct is
under the Qwen Research License, `license:other`, **not** Apache-2.0; the top
rung is therefore Qwen3-4B-Instruct-2507, Apache-2.0) — is frozen as a
preregistration in **Addendum A** below, before any scaled-run token is
generated.*

---

## 3. Evaluation protocol

### 3.1 Task instance, splits, and adaptation sets

- **Task instance:** the CORD receipt schema (one instance for now — the
  product story is many schemas; the eval buys the first honest point, and
  the writeup must say "n=1 schema" explicitly).
- **Adaptation sets:** k ∈ {5, 10, 30} receipts sampled from CORD train with
  seeds {0, 1, 2} (3 resamples per k, reported as mean ± range — adaptation
  variance is part of the result, not noise to hide).
- **Dev:** CORD's 100 dev receipts for any hyperparameter choice (LoRA rank,
  epochs, samples-per-vote). Frozen before test.
- **Test:** CORD's 100 test receipts, touched once per preregistered config.
  No test-driven iteration; if we iterate, dev only, and the test shot is
  re-preregistered.

### 3.2 Metric

- **Primary: field-level micro-F1** over extracted (field-path, normalized
  value) pairs against `gt_parse` (numeric normalization for prices/counts,
  whitespace/case fold for names).
- **Secondary: exact-match rate** of the full canonicalized JSON (the "no
  human in the loop" number an IDP buyer cares about), and invalid-JSON rate.
- Published document-parsing work on CORD (e.g. Donut,
  https://arxiv.org/abs/2111.15664) reports tree-edit-distance accuracy on
  the *image→JSON* task; our text-only post-OCR variant is **not directly
  comparable** and the writeup must not juxtapose our numbers with theirs as
  if it were.

### 3.3 Arms

1. **Adapted small model:** Qwen2.5-Instruct (0.5B first) + per-task LoRA TTT
   on the k examples (LOO corpus × order-shuffles), then greedy + sampled
   generation, vote/rescore, top-1 submitted.
2. **Un-adapted small model, k-shot prompt** (isolates the TTT delta — the
   claim is "adaptation closes the gap," so this arm is mandatory).
3. **Frontier API, k-shot prompt:** Claude Sonnet 5 and Claude Opus 5, same k
   examples in-context, JSON output requested. (Structured-output modes are a
   legitimate frontier-side strength; report with and without if budget
   allows.)
4. *(Optional, cheap)* Frontier zero-shot with schema description only.

### 3.4 Cost-curve axes (the deliverable chart)

- **X:** one-time adaptation cost per task instance, in measured GPU-seconds ×
  measured $/hr (T4 via Lightning, same measurement discipline as
  the cost-accounting rule used throughout this project — never quoted
  without the config), swept via k and epochs.
- **Y:** field-level F1 on held-out test.
- **Reference lines:** frontier k-shot F1 (flat lines with their measured
  $/instance recurring cost); un-adapted small model F1 (floor).
- **Companion table:** break-even instance count per config =
  adaptation-$ / (frontier-$-per-instance − adapted-$-per-instance), computed
  only from measured numbers, presented as a range across configs per the
  cost-quoting rule (ranges across configs, never point estimates).

### 3.5 Preregistered success/failure criteria (fixed BEFORE any GPU run)

Committed to this file, with config hashes, before the first adaptation run;
results land in `experiments/` as machine-readable JSON win or lose.

- **G-E1 (pipeline):** ≥95% of adapted-model outputs parse as valid JSON on
  dev. Fail → fix harness before any test shot; no quality claims meanwhile.
- **G-E2 (adaptation effect):** adapted model beats un-adapted same-model
  k-shot by ≥5 F1 points (mean over seeds) at k=10 on test. This is the
  minimum claim the whole wedge rests on.
- **G-E3 (competitiveness):** adapted small model reaches ≥90% of the best
  frontier k-shot F1 at k=30 for "parity"; exceeds it for "win."
- **G-E4 (cost):** measured marginal inference cost of the adapted model is
  ≥10× below the measured frontier per-instance cost (sanity check of the
  assumed $0.02–0.05 per-instance frontier range — an assumption, not a
  measurement — with real numbers).
- **Kill/iterate rule:** if G-E2 fails after 3 preregistered dev-side
  iterations (augmentation count, LoRA rank, epochs — each logged), publish
  the negative result and stop scaling this eval; the wedge claim loses its
  "adaptation closes the gap" line until a config passes.

### 3.6 Minimal first experiment (~1 GPU-hour, one publishable point)

- **Config:** Qwen2.5-0.5B-Instruct (cached, pipeline-validated), k=10,
  seed 0; LOO corpus (10 sequences) × 3 order-shuffles = 30 training
  sequences; LoRA r=16 (rslora, as in the ARC path), 2 epochs; greedy + 4
  sampled candidates; vote/rescore; **dev-only** (50 of the 100 dev receipts)
  for this smoke point.
- **Budget (estimates):** adaptation ≈ minutes on T4 at 0.5B; 50 greedy+
  sampled evaluations of ~300-token outputs ≈ tens of minutes → fits in ~1
  T4-hour ≈ $0.5–1.5 at measured Lightning rates. Frontier comparison on the
  same 50 receipts ≈ **$1–3** (both Claude models, k=10, Batches API).
- **Output:** one `experiments/cordtext_smoke_<date>.json` with F1/EM for
  arms 1–3, timing, config, dataset revision hash — the first non-ARC row in
  the registry, whatever the numbers say. Only after this lands do we spend
  on the full k-sweep and the test shot.

---

## 4. What we may honestly claim at each outcome (AGENTS.md rules)

Scope preamble mandatory in every use: *one public dataset, one schema,
text-only post-OCR extraction, small open models, measured configs stated.*

- **Win (G-E2 + G-E3-exceed + G-E4):** "On CORD receipt extraction (n=1
  enterprise-shaped schema), a per-task-adapted Apache-2.0 0.5B/1.5B model
  beat a k-shot frontier baseline by X F1 at a measured Y× lower marginal
  cost; adaptation itself contributed Z F1 over the un-adapted model
  [artifact link]." **Not claimable:** "works on your documents," anything
  about vision/layout tasks, anything implying n>1 schemas, any ARC-to-
  enterprise transfer beyond this measured instance.
- **Parity (G-E2 + G-E3-90%):** "Adaptation closed most of the gap to
  frontier k-shot quality (within X F1) at Y× lower marginal cost on one
  public extraction task [artifact]." The wedge survives as an economics
  story; any public quality claim must say "approaches," not "beats."
- **Loss (G-E2 passes, G-E3 fails):** claimable: "TTT measurably improves the
  small model on a real extraction task (+Z F1) but does not yet reach
  frontier quality; here is the measured gap and the cost frontier." That is
  a legitimate research artifact and must be published in `experiments/`
  like every other honest zero in this repo.
- **Loss (G-E2 fails after iteration budget):** claimable: only the negative
  result. The enterprise quality claim reverts to "open empirical question,
  first negative evidence at 0.5–1.5B scale, next lever is model scale" — and
  the cost model's quality caveat becomes the headline, not the footnote.
  **Not claimable:** any suggestion that the ARC pipeline validates the
  enterprise wedge.

---

## 5. Source URLs (all access-dated 2026-08-08)

- CORD repo + license: https://github.com/clovaai/cord (README license
  statement verified); HF: https://huggingface.co/datasets/naver-clova-ix/cord-v2 ;
  paper: https://openreview.net/forum?id=SJl3z659UH
- SROIE: https://rrc.cvc.uab.es/?ch=13 ; https://arxiv.org/abs/2103.10213 ;
  mirrors https://github.com/zzzDavid/ICDAR-2019-SROIE ,
  https://huggingface.co/datasets/darentang/sroie
- WikiTableQuestions: https://github.com/ppasupat/WikiTableQuestions ;
  https://arxiv.org/abs/1508.00305 ;
  https://huggingface.co/datasets/wikitablequestions
- FUNSD (rejected, non-commercial): https://guillaumejaume.github.io/FUNSD/work/ ;
  https://arxiv.org/abs/1905.13538
- DocVQA (rejected): https://www.docvqa.org/
- Donut (CORD SOTA context, not directly comparable):
  https://arxiv.org/abs/2111.15664
- Frontier pricing used in estimates: https://platform.claude.com/docs/en/pricing
  (Claude Sonnet 5 $3/$15 per MTok — $2/$10 introductory through 2026-08-31;
  Claude Opus 5 $5/$25; Claude Haiku 4.5 $1/$5)
- Base model license: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
  (Apache 2.0)

---

## Addendum A — Preregistered scaled run (frozen 2026-08-08, before execution)

*Status of this addendum: FROZEN before the first token of any arm below is
generated. The dev-scale variance sweep this design responds to is complete
except one arm: the k=30 seed-1 **adapted** arm is still running (its k-shot
counterpart is on disk: `experiments/cord_k30_seed1_kshot_2026-08-08.json`,
mean F1 0.6716). Whatever that fifth pair returns, it lands in the dev-sweep
record and does not alter this preregistration — it was written without
knowledge of that result. Changes to this addendum follow §A.5 only.*

### A.1 Hypothesis (exact, falsifiable)

**H-SCALE:** At fixed recipe (per-task LoRA TTT, LOO corpus, hyperparameters
frozen in §A.2), the paired adaptation delta — adapted minus un-adapted
same-model k-shot, field-level micro-F1, k=10, CORD held-out dev slice —
**increases monotonically with base-model scale over the ladder
{0.5B, 1.5B, 4B-class}, and passes G-E2 (≥ +5 F1 points, paired mean over
seeds, at k=10 on the held-out dev slice) by the top rung.**

Falsified if: the k=10 paired mean delta does not increase from 0.5B to the
top rung, or no rung reaches +5 F1. Partially supported if deltas increase
but no rung passes G-E2 (report as "scale-positive, threshold-fail" — not a
pass).

**Motivating evidence (all numbers from artifacts or cited as theirs):**

1. **0.5B is net-neutral.** `experiments/cord_variance_summary_2026-08-08.json`:
   4 paired arms, all-pairs mean delta **−1.3 F1** (−0.0126), range −6.5 to
   +12.7, only 1/4 positive (the original seed-0 smoke pair). The spec's §3.5
   loss branch fired and was honored in all materials the same night.
2. **4B-class sharpens.** `experiments/t4_champ_diag_2026-08-08.json`: on the
   champion 4B ARC checkpoint, one-epoch TTT sharpened teacher-forced
   lp(true) on **3/5 comparable tasks** (5/7 test pairs; best −0.120 →
   −0.029), degraded it on two; solve count unchanged (recall-limited).
3. **NVARC's own scale step (their numbers, cited as theirs).** With the same
   published TTT recipe, their 2B variant scored **22.22%** public vs the 4B
   competition entry's **27.64%** public (`docs/research/NVARC_RECIPE.md`;
   the 2B is their post-deadline variant on the LLM part of Qwen3-VL-2B).

**What this design removes and what it keeps (stated up front):**

- **Removed — recipe completeness.** Every dev arm above ran greedy top-1
  with **no vote/rescore layer active** (caveat recorded in the variance
  summary artifact). The scaled run turns vote/rescore ON at all scales,
  including a 0.5B re-run arm, so recipe completeness is no longer confounded
  with scale.
- **Removed — within-ladder domain-SFT.** All three ladder models are stock
  instruct checkpoints with no receipt/domain SFT, so *within this ladder*
  scale is not confounded with domain fine-tuning.
- **Kept — scale vs domain-SFT in the motivating evidence.** Evidence items
  (2) and (3) come from heavily domain-SFT'd checkpoints; their "bigger is
  better" signal confounds scale with domain SFT, and this design cannot
  separate those without owned domain-SFT'd weights at multiple scales
  (that is OWN_MODEL_PLAN Tier 2 territory). Said plainly: a pass here shows
  scale helps *this* recipe on stock instruct models; it does not decompose
  NVARC's 2B→4B gain.
- **Kept (new, forced by license) — family generation at the top rung.**
  Qwen2.5-3B-Instruct carries the **Qwen Research License** (`license:other`
  on its HF card, verified 2026-08-08 via the HF API), not Apache-2.0 — §2.4
  disqualifies it for a pitch-attached artifact. The top rung is therefore
  **Qwen3-4B-Instruct-2507 (Apache-2.0, verified 2026-08-08)**. The top rung
  consequently differs in model generation (Qwen2.5 → Qwen3) as well as
  scale. The 0.5B→1.5B step remains within-family and license-clean.

### A.2 Design (frozen)

**Models (licenses verified via HF API 2026-08-08):**

| Rung | Model | License |
|---|---|---|
| 0.5B (re-run) | Qwen/Qwen2.5-0.5B-Instruct | Apache-2.0 |
| 1.5B | Qwen/Qwen2.5-1.5B-Instruct | Apache-2.0 |
| 4B-class (top) | Qwen/Qwen3-4B-Instruct-2507 | Apache-2.0 |

**Arms:** per rung, k ∈ {5, 10, 30} × seeds {1, 2, 3}, paired
(adapted-TTT vs un-adapted same-model k-shot) = 18 arms per rung, 54 total.
**20 held-out receipts per arm** from `demo/cord_validation.jsonl` (the same
100-receipt dev pool and per-seed reshuffle procedure as the variance sweep;
deltas are paired within seed, F1 levels not comparable across seeds — same
caveat as the dev artifact).

**Protocol: SAME as the variance sweep in every respect EXCEPT the single
preregistered change — vote/rescore ON.** Generation is greedy + 4 sampled
candidates (5-candidate pool; sampling temperature 0.7, the `model.py`
config default), pooled on canonicalized-JSON key, count + likelihood
rescored, top-1 submitted (§2.2 `vote.py` adaptation). This applies to
**both arms** (adapted and k-shot) at **all scales, including the 0.5B
re-run**, so the paired delta isolates adaptation, not decoding, and the
scale comparison stays clean. The dev arms' greedy-only protocol was a
recipe-completeness confound; this is its removal, not a tuning knob.

**Hyperparameters (frozen at the dev-sweep values, from
`cord_variance_summary_2026-08-08.json` `config`):**

| Parameter | Frozen value |
|---|---|
| LoRA rank r | 16 |
| LoRA alpha | 32 |
| TTT epochs | 1 |
| eval_n per arm | 20 |
| max_new_tokens | 512 |
| max_seq | 4096 (k=5, k=10); 8192 (k=30) |
| Corpus | LOO + order-shuffles, as in dev sweep |
| Candidates | 1 greedy + 4 sampled (T=0.7), vote/rescore top-1 |
| Metric | field-level micro-F1 (§3.2), plus EM and invalid-JSON rate |

**Decision rule:** G-E2 evaluated **per rung** on the k=10 paired mean delta
over seeds {1,2,3} on the dev slice: pass iff mean(adapted − k-shot) ≥ +5 F1
points. k=5 and k=30 arms are reported (curve shape) but do not gate.
**One test shot on CORD test-100 ONLY if dev G-E2 passes at any rung** —
exactly one shot, at the smallest passing rung, config k=10 × seeds {1,2,3},
vote/rescore ON, touched once per §3.1.

### A.3 Compute + schedule

- **0.5B re-run + 1.5B:** CPU / free-quota. Measured 0.5B dev arms ran
  353–982 wall-seconds each on CPU (variance summary `run_wall_seconds`);
  the ~5× candidate count and 1.5B's ~3× parameter cost keep the two lower
  rungs within free CPU/Kaggle-CPU budgets — no paid GPU.
- **4B-class rung:** needs GPU — **next week's Kaggle 30h weekly T4 quota**,
  estimated **~10–30 T4-hours total** (champion-4B TTT tasks ran 458–965
  s/task on T4 per `t4_champ_diag_2026-08-08.json`; 18 arms × 20 receipts
  with a 5-candidate pool fits the quota with headroom for OOM retries).
  Kaggle non-negotiables apply (no credentials in git; no submission slots
  involved — this is eval compute only).
- **Frontier k-shot baseline grid:** < $30 API per §1.2 (2 Claude models ×
  3 k-values, Batches API), run once against the same held-out receipts.
- **Target: all dev arms complete within 7 days of freeze (by 2026-08-15);**
  test shot (if earned) and artifact write-up in the same week.
- **Artifacts:** one machine-readable JSON per arm plus a
  `cord_scale_summary_<date>.json` roll-up in `experiments/`, same
  discipline as the variance sweep, win or lose.

### A.4 Outcome branches (pre-written)

1. **Pass at 4B-class only (H-SCALE supported at top rung).** Publish the
   per-scale delta curve + summary JSON. Materials update: public materials
   gain a measured "adaptation delta emerges with
   scale" line (numbers only from the roll-up artifact, scale ladder and
   family-generation caveat stated); break-even is re-computed at
   4B-class serving costs; test shot proceeds at the 4B-class rung. The 0.5B
   net-neutral stays published and cited — it is the curve's floor, not an
   embarrassment.
2. **Pass at 1.5B (with or without 4B-class also passing).** Strongest
   commercial outcome: the wedge works at a CPU-serveable scale. Same
   publications as branch 1 plus a cost-curve point at 1.5B serving costs;
   test shot at 1.5B (smallest passing rung); demo/endpoint story upgrades
   from "mechanism" to "mechanism + measured quality gain."
3. **Fail everywhere (no rung reaches +5 F1 at k=10).** The negative result
   attaches to **this recipe at small open-instruct scale on this schema** —
   publish it in `experiments/` and the paper's negative-results section like
   every other honest zero. **The wedge claim retires to economics + privacy
   only** (no quality-gap claim in any material) **until 4B-class owned,
   domain-SFT'd weights exist — the internal own-model plan's Tier 2
   ("own-SFT", ~$2K–13K)**, which is exactly the asset that removes the
   scale-vs-domain-SFT confound this addendum keeps. Materials update: the
   enterprise claim reverts to §4's loss language ("open empirical question, negative
   evidence at 0.5B–4B-class stock-instruct scale, next lever is owned
   domain-SFT weights"); the cost model's quality caveat becomes the headline.
   In all three branches, every updated document cites the roll-up artifact
   and nothing else.

### A.5 Amendment rule

This addendum changes **only by dated, appended amendment** (a new
"Amendment A.N (YYYY-MM-DD)" block below this section stating what changed
and why, before any affected arm runs). It is never edited in place.
Results, whatever they are, are reported against the text as frozen on
2026-08-08.

---

## Addendum B — Novel-schema gate (frozen 2026-08-12T19:40Z, before any run)

Written before a single record was generated or a single arm executed. The
generator (`src/arcttt/novel_schema.py`, 399 tests) exists; no corpus, no
scores, and no arms exist at the time of writing. Everything below is a
pre-commitment, and B.6 records what may be claimed at each outcome so no
branch can be re-argued after the numbers land.

### B.1 Why a second gate at all

Addendum A's gate failed at 0.5B (−7.3 F1) and 4B (−4.5 F1). Those are real
results and they are NOT retracted or reinterpreted by this addendum. But
they were collected in the regime least favourable to the hypothesis: CORD
is public, near-certainly in pretraining, and its schema is ordinary
commercial vocabulary. The prompted arm therefore starts out knowing what a
receipt is and what its fields are called, and per-request adaptation cannot
add knowledge the model already has. G-E2 largely establishes that
"prompting already knows CORD", which is close to a tautology and is NOT the
claim the product makes.

The product claim is narrower and testable: adaptation buys you NOVELTY —
a schema no pretraining supplies. This gate tests exactly that and nothing
else.

### B.2 The single changed variable

Exactly ONE thing changes from Addendum A: the corpus is a synthetic novel
schema instead of CORD. Model, LoRA config, epochs (1), optimiser, decode
settings, scorer, pairing and seed discipline are all held at Addendum A's
frozen values. In particular epochs stays at 1 even though "one epoch is too
few" is a live alternative explanation for G-E2 — changing two variables
would make a positive result uninterpretable. Epoch count is a SEPARATE
future gate, not this one.

### B.3 Decision point (declared now, not after)

- **The gate is k=30.** Pass iff the mean paired delta (adapted − k-shot) in
  field-level micro-F1 at k=30, over seeds {1,2,3}, is ≥ **+5.0 F1**.
- **k=10 is a comparability point, NOT a decision point.** It exists only to
  sit alongside Addendum A's k=10 numbers. If k=10 comes out positive and
  k=30 does not, the gate FAILS and the k=10 number may not be promoted.
  This sentence exists because the 4B k=5 arms came out at +6.9 F1 with an
  interval excluding zero and still turned out to be two receipts (p=0.38);
  the trap is now pre-committed against rather than caught afterwards.
- Both statistics must agree, per the rule already shipping in
  `cord_paired_power.py`: a pass requires the parametric interval AND the
  sign test to point the same way. A mean that survives only until its two
  largest winners are removed is not a pass.

### B.4 Power (the Addendum A defect, fixed by construction)

Addendum A's gate could not resolve its own threshold: MDE 13.1 F1 at 0.5B
and 11.3 F1 at 4B against a 5 F1 bar, and CORD's 100-row validation split
could not supply the ~115 distinct paired receipts required at ANY eval_n.
That ceiling was a property of the dataset, and a synthetic corpus does not
have it.

**eval_n is therefore raised to 60 per seed** (180 paired records across
three seeds), chosen so that at the per-receipt spread observed on CORD
(sd ≈ 0.19) the MDE lands near 4 F1 — below the 5 F1 bar the gate tests.
This is the first gate in this project that can actually resolve the effect
it is testing. The realised MDE will be recomputed from the observed spread
and reported alongside the result; if it comes out above 5 F1 the gate is
reported as UNDERPOWERED rather than as a pass or a fail.

### B.5 Validity gates (checked BEFORE the delta is read)

A null is only informative if the task was measurable at all. Both of these
are checked first, and if either trips, the run is reported as
uninformative and the delta is NOT interpreted:

- **Floor.** If the k-shot arm scores < 0.15 mean micro-F1, the base model
  cannot do the task at all and a zero delta means "too hard", not "no
  benefit". 0.5B is small enough that this is a genuine risk.
- **Ceiling.** If the k-shot arm scores > 0.95, there is no headroom for
  adaptation to demonstrate anything and the corpus is too easy.

### B.6 Outcome branches (pre-written)

- **PASS (≥ +5 F1 at k=30, both statistics agreeing, validity gates clear).**
  Claimable: "per-request adaptation beats in-context prompting by X F1 on
  schemas absent from pretraining, preregistered, at n=180 paired records."
  This is the wedge's first positive quality evidence and it would justify
  re-opening the raise. It must ALWAYS be reported next to the CORD
  negative, framed as what it is: adaptation buys novelty, not general
  quality.
- **FAIL (< +5 F1).** Combined with Addendum A this is close to decisive
  against the quality thesis as implemented: adaptation would then have
  failed both where the model already knows the domain AND where it cannot
  possibly know it. The honest conclusion is that one epoch of per-request
  LoRA does not buy extraction quality at these scales, and the company
  case rests on infrastructure and measurement rather than on model quality.
  This branch must be stated to Raj in those words.
- **UNINFORMATIVE (a validity gate trips).** Report as such, fix the
  corpus difficulty or the rung, and re-run. An uninformative run is NOT a
  fail and may not be counted as evidence either way.

### B.7 Execution constraints

Runs on CPU (no GPU quota until 2026-08-15). 0.5B first: it is the cheapest
rung and the one with the most complete Addendum A comparison. 4B is NOT
scheduled here — the 4B CPU attempt produced no arm at all in a full
session, so 4B on CPU is known non-viable at the frozen config.

**Not launched at the time of freezing.** The 1.5B Addendum A rung is mid-
flight, and a second concurrent Kaggle kernel risks contending with it. The
frozen preregistered work has priority over the exploratory gate.

**Execution revision B.7-r1 (2026-08-14T07:05Z).** The CPU execution named
above was measured infeasible, not merely slow: kernel v1 was OOM-killed at
the first k=30 adaptation (LOO context makes ~7.5k-token sequences; fp32
logits over a 152k vocab exceed the CPU session's RAM — full record in
`experiments/novel_schema_cpu_oom_postmortem_2026-08-14.json`). Execution
moves to T4/bf16 after the 2026-08-15T00:00Z quota refresh. Nothing frozen
in B.2–B.6 changes: same generator, seeds, eval_n, LoRA config, epochs,
decode, scorer, pairing, decision point and validity gates. All arms will
be cuda/bf16, so every pair remains internally environment-homogeneous;
cross-dataset comparisons to Addendum A's CPU/fp32 0.5B rung inherit the
same device/dtype caveat already recorded for the 4B rung.

**Execution revision B.7-r2 (2026-08-15T11:30Z).** The T4 run completed all
nine kshot/k=10 arms but all three k=30 ADAPTED arms OOMed: a 7.5k-token
training sequence's full logits over the 152k vocabulary do not fit T4
memory alongside the backward pass. Fix: `chunked_loss_tokens=512` — the
same shifted fp32 cross-entropy computed over 512-token slices with a
two-phase backward. This is math-identical (per-token CE is additive;
gradients pinned equal to the labels path by tests/test_chunked_loss.py
across dividing and non-dividing chunk sizes) and therefore an execution
revision, not a protocol change. Interim state at freeze of this note:
gate UNDECIDABLE (0/3 k=30 pairs); the k=10 comparability point reads
+45.0 F1 mean, CI [+41.9,+48.1], sign test 171W/0L/9T — recorded and,
per B.3, NOT promotable regardless of magnitude.

**Execution revision B.7-r3 (2026-08-15T11:45Z).** B.7-r2's chunked loss
was necessary but not sufficient: with gradient checkpointing verified
engaged and allocator fragmentation eliminated, the k=30 trunk backward
still dies on the T^2 attention buffer — this torch build's
memory-efficient SDPA kernel requires sm80+ for bf16, so on the sm75 T4
every available path materializes full T^2 attention. Revision: the k=30
arms run in fp16, where the memory-efficient kernel IS sm75-eligible and
attention memory is linear in sequence length. Pair homogeneity holds
because the k=30 KSHOT arms are re-run in fp16 alongside (the entry purges
non-fp16 k=30 artifacts before the arm loop); every artifact now stamps
its dtype. The k=10 pairs remain bf16/bf16. Cross-k comparisons therefore
carry a dtype caveat, exactly like Addendum A's cross-rung device caveat;
within-pair comparisons — the only thing the gate reads — are clean.

**Execution revision B.7-r4 (2026-08-15T21:20Z).** The GPU path is
abandoned for the k=30 pairs after six instrumented attempts: the scoring
image defeats HF gradient checkpointing, a per-layer torch.utils.checkpoint
monkeypatch, AND torch.autograd.graph.save_on_cpu (held GPU memory
byte-stable at ~11.3 GB throughout — mechanisms verified working on
transformers 5.15 locally), and fp16 introduced a device-side assert on
seed 3. The k=30 pairs move to CPU/fp32 — the path every Addendum A CPU
rung already validated end-to-end, whose only k=30 blocker (the
full-sequence logits in the loss) is exactly what the gradient-pinned
chunked cross-entropy removes. Both sides of every k=30 pair are produced
on CPU/fp32 (the entry purges all GPU-era k=30 artifacts); k=10 pairs
remain cuda/bf16; the summary's homogeneity refusal enforces all of this
mechanically. Slower, with no novel failure modes — after six of them,
that is the point.

### B.8 — Replication sweep of the comparability point (frozen 2026-08-15T21:25Z, before any run)

Role: EXPLORATORY REPLICATION. This section does not touch the gate (B.3:
k=30, seeds {1,2,3}) and cannot substitute for it. Its question: is the
k=10 comparability result (+45.0 F1, 171W/0L/9T over tenants 1-3) a
property of three lucky vocabularies, or of the task class?

Design, frozen before any record was generated:
- Seeds 4,5,6,7,8,9,10 — seven NEW tenants from the same generator, same
  schema geometry (8 fields, 2 groups, 4 distractors).
- k=10 only; eval_n=60; model, LoRA config, epochs=1, decode, scorer,
  pairing identical to B.2's frozen values; cuda/bf16 (the environment the
  original k=10 arms ran in — pairs remain internally homogeneous).
- Reported with the same two-statistics standard (interval + sign test),
  pooled and per-tenant. Claimable, if it holds: "replicated across ten
  invented schemas, n≈600 paired records" — as a robustness statement
  about the COMPARABILITY point, always labeled as such, never as the
  gate.
- If any tenant flips negative, that is reported with the same prominence
  as the wins. The sweep's value is that it can fail.

### B.7-r5 (2026-08-16 ~05:40Z) — arm-scoped kernel packaging

Execution revision only; no frozen protocol value changes. The v11 full-chain
CPU kernel was cancelled ~9h into its run (not by us; cause unknown — Kaggle
UI or system) with zero k=30 arms saved, and its log shows a single k=30
adapted arm still incomplete at cancellation. Since Kaggle batch kernels only
persist output at natural session end, a k=30 PAIR may not fit one 12h CPU
session. Repackaging: one kernel per (k=30, seed, arm) unit — six kernels
s{1,2,3}{a,k} — each finishing well inside the cap. The seed-2/3 pair shards
(B.7-r4 era) continue running and race the arm-scoped kernels under the
first-terminal-wins duplicate policy frozen with the shards. Kaggle's
concurrent batch-CPU limit is 5; s2k/s3a/s3k queue behind free slots.
Computation per arm is bit-comparable to the pair shards (same entry code
path, ARM_ORDER restricted).

#### B.7-r5 correction (2026-08-16 ~06:30Z)

The r5 premise "output persists only at natural session end" is wrong for
cancellation: the s2 pair shard was cancelled ~6.5h in (cause unknown, same
pattern as v11) and its output WAS saved, including the first completed k=30
gate arm (seed-2 adapted, cpu/fp32, 0.9934, 60/60 receipts, adapt 11498s).
v11's cancel saved nothing because no k=30 arm had completed, not because
cancellation discards output. The arm-scoped packaging decision stands on
its real justification: one ~6h arm per kernel clears the 12h cap with
margin, where a ~12h pair does not. Measured CPU timing (first datum):
adapt ~3.2h, full adapted arm ~6h at k=30/eval_n=60.

#### B.7-r5 duplicate resolution, seed-2 adapted (2026-08-16 ~19:15Z)

The s2 pair shard (first terminal, 06:15Z) and the s2a arm kernel (19:05Z)
produced same-environment (cpu/fp32) adapted arms with different numbers:
0.9934 vs 0.9836. Resolved by the first-terminal-wins rule frozen in the
shard entries before any k=30 data existed: 0.9934 stays banked; 0.9836 is
preserved at experiments/duplicates/ as the same-env reproducibility datum.
Immateriality check, stated at resolution time: seed-2 kshot is 0.5033, so
the pair delta is +49.0 or +48.0 under either value — the +5 bar is cleared
by ~10x and the choice cannot affect the verdict. The ~1-point run-to-run
spread on identical cpu/fp32 environments is hereby noted as the observed
CPU nondeterminism scale; it is small against the +5 bar but belongs in any
future power calculation.

### B.7-r6 (2026-08-16 ~20:15Z) — cancellation-proof arm kernels

Execution revision only; frozen protocol values untouched. Trigger: the
external canceller struck again at ~19:30Z, killing both seed-1 arm kernels
at ~13.3h with arms unfinished (nothing saved; unlike s2's shard there was
no completed artifact to persist). Countermeasure, added to every arm entry:
(1) after adaptation, the trained LoRA state (all lora_a/lora_b tensors) is
saved to novel_ckpt_*_adapter.pt with adapt_seconds in a meta file; (2)
every scored doc is appended (flush+fsync) to novel_ckpt_*_docs.jsonl with
its unrounded micro-F1; (3) on launch, checkpoint files seeded from the
attached dataset are consumed — the adapter is restored bit-identically
instead of retrained, journaled docs are not re-decoded, and counters are
rebuilt from unrounded values. Checkpoint filenames deliberately avoid the
novel_schema_* prefix so the banker, artifact seeding filter, and r4 purge
never match them. Artifacts stamp "resumed" for transparency. Resume path
covered by an offline journal-reconstruction test; adapter save/load uses
the same inject_lora/remove_lora API as training. Deployed by delete+repush
(Kaggle holds a slot per running session, so an in-place version push is
rejected at the 5-session cap): s1a, s1k, s3kb run checkpointed; s3k v1
continues unprotected as s3kb's race partner; s3a (13h in, near completion)
was deliberately not touched.

#### B.7-r6 board note (2026-08-17 ~03:45Z)

s3a (the one arm kernel deliberately left un-checkpointed because it was
13h into its run) was cancelled by the external canceller before writing
its artifact — its log shows adaptation completed at ~2.9h and decode was
in progress; ~13h of compute lost. Relaunched at 03:40Z as v2 with the
r6 checkpointed entry (adapter saved post-adapt, docs journaled), ETA
~16:40Z if undisturbed, sweep-resistant either way. Every kernel on the
board now runs the checkpointed entry except s3k v1, whose loss is covered
by its checkpointed race partner s3kb.

#### B.7-r6 first live resume (2026-08-17 ~13:05Z)

The canceller swept four of five kernels during the 06:20-12:49Z window.
Checkpoint recovery, as designed: s1a saved its adapter + 54/60 docs,
s1k 43/60, s3kb 37/60 (s3k v1 ran unprotected and lost everything - its
racer's journal covers the arm). All checkpoint files versioned into the
resume dataset and all four kernels relaunched at ~13:10Z; expected
remaining work is ~1-3.5h per arm instead of 13h. One operational
lesson recorded: `kaggle datasets metadata` returns a file without the
`id` field and `kaggle datasets version` then fails with "ID or slug
must be specified" AFTER kernels were already pushed - the fixed order
is version-first-then-push, and resume_arm_kernel.sh already encodes it.

#### s3a second cancellation + scripted resume (2026-08-17 15:45Z)
Canceller struck s3a v2 at ~12h; this time checkpointed: adapter + 47/60
docs recovered, resume_arm_kernel.sh executed the full pull->version->
repush flow, ~2h remain (ETA ~17:45Z). Verdict ETA ~18:00Z.

#### Board closed (2026-08-17 20:12Z)

s3kb (the checkpointed race partner) completed after the verdict: seed-3
kshot 0.4354 vs the banked s3k 0.4333 — same-environment spread 0.0021.
Preserved at experiments/duplicates/ per first-terminal-wins. The gate's
two free replication data now bound CPU run-to-run spread at 0.0021-0.0098
across arms — two orders of magnitude under the +46.5 measured effect.
All six k=30 arms banked, verdict GO, board complete.

## Addendum C — Scale-rung gate (frozen 2026-08-17T20:35Z, before any run)

Frozen the evening the Addendum B verdict landed (GO, mean +46.5,
156W/0L/2T over 158), before any rung above 0.5B has produced a single k=30
novel-schema record and before any GPU beyond Kaggle's free tier has
been rented. The company-shape decision of the same date (lab + quiet
data partners) makes this the next public artifact: the scale curve.

### C.1 The question

Does the Addendum B effect survive model scale? The known threats, named
now: (a) bigger models have higher k-shot baselines (less headroom — the
B.5 ceiling gate will bite earlier); (b) Addendum A showed adaptation
LOSING at 1.5B and 4B on CORD, so scale is a live risk to the thesis,
not a formality; (c) environment changes with scale (GPU dtype), which
is why pair homogeneity is per-rung.

### C.2 The single changed variable

Per rung, exactly ONE thing changes from Addendum B's frozen protocol:
MODEL_ID. Rungs: **1.5B (Qwen2.5-1.5B-Instruct, free Kaggle T4, fp16
both arms)**, then funded rungs **7B / 14B / 32B** (rented GPUs; dtype
per rung recorded, both arms identical). k=30, seeds {1,2,3}, eval_n 60,
epochs 1, LoRA r=16/alpha=32, decode settings, scorer, pairing and seed
discipline all stay at B's frozen values. The corpus generator is
seed-identical to B: the SAME three tenants, so rung deltas are
attributable to scale alone.

### C.3 Decision points (declared now)

- Each rung passes iff mean paired delta at k=30 over seeds {1,2,3} is
  ≥ +5.0 F1, parametric interval AND sign test agreeing (B.3 rule).
- B.5 validity windows apply per rung ([0.15, 0.95] on the kshot arm).
  Given threat (a), a ceiling trip at a large rung is reported as
  UNINFORMATIVE-CEILING and the corpus difficulty is escalated by a
  preregistered amendment BEFORE that rung re-runs; the 0.5B result is
  never re-litigated by it.
- **The curve claim is gated separately:** "the effect holds at scale"
  may be stated only when ≥3 rungs (0.5B counts as one) have passed
  with validity clear. One extra rung passing is reported as that rung
  passing, nothing more.
- A rung that fails is reported in Addendum A's language: adaptation
  as implemented does not buy quality at that scale on this corpus. No
  averaging across rungs; no dropping of failed rungs from the curve
  figure. The figure shows every rung run, pass or fail.

### C.4 Execution constraints

- 1.5B rung: free tier, arm-scoped checkpointed kernels (B.7-r6
  machinery verbatim), T4 fp16 both arms — the fp16 path validated on
  the 4B CORD arms. May start any time; costs $0.
- 7B+ rungs: rented, funded by rung-1 raise; each rung's environment
  (GPU, dtype, image) recorded in every artifact; both arms of a pair
  on identical hardware. Budget envelope stated in
  SCALE_EVIDENCE_ROADMAP.md ($50-150K total); per-rung cost is
  published with the artifact.
- Banker, first-terminal-wins, duplicate preservation, resumed stamps:
  all B.7-r5/r6 rules apply unchanged.

#### C.4 note (2026-08-17 20:45Z): 1.5B rung staged, blocked on quota
Both seed-1 1.5B arm kernels (c15-s1a/s1k) are built, verified and
metadata-pinned to GPU; the weekly 30h GPU quota is exhausted, so they
push at the next quota reset. Frozen protocol unaffected — execution
timing is not part of the freeze.

### B.9 — Post-verdict corrections and scoping (2026-08-17 ~20:30Z)

An adversarial diligence pass ran the evening of the verdict. Its
confirmed findings are recorded here, against ourselves, with the same
discipline the verdict was produced under.

**B.9.1 Configuration scoping of the headline claim (the material one).**
Both arms of every pair carry the full k-demo prompt at decode time
(text_task_to_messages includes all train demos; the adapted arm differs
only by the trained LoRA). The +46.5 therefore measures adaptation ADDED
ON TOP of in-context prompting at 0.5B — not a document-only serving
mode. The document-only configuration (adapter + bare document, the mode
the payload-asymmetry economics describe) has NO quality measurements.
Claim rule, binding immediately: the GO number and the 20-58x payload
number may not be stated as properties of one configuration. The
document-only quality gate is hereby named as a REQUIRED future
preregistered gate (Addendum D when frozen) and is the first experiment
of the funded program.

**B.9.2 Attrition disclosure.** Designed receipt n was 180 (60 x 3
seeds). Scored n is 158: the seed-2 pair excluded the SAME 22 documents
in both arms (prompt over the 8192-token cap returns no prediction and
the document is excluded from the mean, not scored 0). Correction to
B.7-r5's note: seed-2 adapted banked with 38/60 scored, not "60/60
receipts". Robustness, computed from the artifacts (2026-08-17 21:00Z check):
scoring all 22 excluded documents as 0.0 in BOTH arms gives seed-2 a
delta of +31.0 and a GATE MEAN of +40.5 — the verdict does not depend
on the exclusions. Every future summary must carry an explicit
attrition field.

**B.9.3 The s3kb "replication" is contaminated and is struck as a
bound.** s3k and s3kb consumed the same 37-document checkpoint journal
(the ckpt stem is arm-keyed, not kernel-keyed), so 53/60 of their
per-document scores are identical by construction. The 0.0021 spread is
NOT a run-to-run bound and the board-closed note's use of it is
retracted. The only genuine same-environment duplicate is seed-2
adapted: spread 0.0098. That single pair is the honest nondeterminism
estimate until a deliberate replication runs.

**B.9.4 Cluster-honest inference.** The receipt-level CI [42.9, 49.4]
pools 158 receipts sharing 3 adapters/tenants — the same pooling
optimism this project criticized on CORD (ge2_power_finding). At the
cluster level the evidence is n=3 seed deltas (+36.0/+49.0/+54.4,
sd~9.4), CI roughly [23, 70]. Both levels clear the +5 bar; both are
now quoted together wherever the CI appears, and "p=0.0" is restated as
p < 1e-15 (float underflow, not a probability of zero).

**B.9.5 Corpus-design limitation (stated, not fixed).** All tenants
share one deterministic schema geometry (group/value-kind assignment by
index modulo); B.8's ten tenants are vocabulary re-rolls of that shape.
A live alternative mechanism for the effect is "0.5B is weak at
long-prompt ICL on unfamiliar formats" — consistent with the frontier
context arm scoring 1.00 on the same corpus. The claim stands as
preregistered (the gate tested what it said it tested) but the corpus
diversity escalation in Addendum C.3 is now REQUIRED for any rung, and
the qualifier "at 0.5B, over the same model's in-context arm" travels
with the number everywhere. The one attack that FAILED verification:
schema leakage — describe() is artifact-only and no path from the
label-key mapping into any prompt exists (verified in code).

**B.9.6 Resume disclosure.** 4 of 6 gate arms are resumed runs
(seed-1 both, seed-3 both; stamped resumed:true in the artifacts).
Adapter restore is bit-identical; decode sampling is unseeded, so a
resumed run's sample stream differs from an uninterrupted one — this is
ordinary run-to-run nondeterminism, bounded per B.9.3. "Resumes
bit-identically" phrasing outside the adapter context is retracted.

**B.9.7 Duplicate-resolution optics.** Both first-terminal-wins
resolutions favored the delta (banked seed-2 adapted was the higher
duplicate, +0.98; banked seed-3 kshot the lower, +0.21). The rule
predates the data and the combined effect (~1.2 points against a +46.5
mean) is immaterial, but the fact is recorded because a skeptic will
notice it before we mention it — so we mention it.

## Addendum D — Document-only serving gate (frozen 2026-08-18T18:05Z, before any run)

The B.9.1 obligation, discharged: the gate that tests whether adapted
quality survives the serving configuration the payload economics assume
(adapter present, NO examples in the request). Frozen before any
document-only decode has ever been run in this project. Free-tier
CPU/fp32 — this gate needs no funding, only the idle CPU slots.

### D.1 Arms

Per seed {1,2,3}, one new arm: **doconly-adapted** — the gate-run
adapter loaded, decode over the same 60 documents with a prompt
containing ONLY the test document (include_demos=False; k recorded as
30 because the adapter was trained on the k=30 corpus). Environment
cpu/fp32, decode settings otherwise at B's frozen values. Adapter
provenance, disclosed: seeds 1 and 3 restore the EXACT gate-run
adapters bit-identically from the published novel_ckpt files; seed 2's
gate pair predates the checkpoint layer, so its adapter is retrained
with the identical frozen recipe (run-to-run adapter spread is bounded
by B.9.3's honest estimate, ~1 F1 on the one genuine duplicate).
Artifacts: novel_schema_d_0.5b_k30_seed{n}_doconly_2026-08-18.json.

### D.2 Reads (declared now)

- **Read 1 — retention (primary).** Per-seed retention delta =
  mean(doconly-adapted) − mean(B adapted-with-prompt), on the
  intersection of scored documents. PASS iff the seed-mean retention
  delta is ≥ −5.0 F1 (quality within 5 points of the prompted adapted
  arm) with the sign test not contradicting (losses may exceed wins by
  at most the 5-point-equivalent; a seed-mean drop worse than −5 is a
  FAIL and the payload economics claim is retired as stated in B.9.1).
- **Read 2 — the unified business claim.** doconly-adapted vs the
  banked B kshot-with-prompt arm: PASS iff seed-mean delta ≥ +5.0 F1
  with interval and sign test agreeing (B.3 discipline). A pass means
  quality AND the 20-58x payload advantage hold in ONE configuration —
  the claim the product page may then state without the adjacent-
  configurations caveat.
- Attrition: doc-only prompts are shorter, so exclusions can only
  shrink; all comparisons run on per-seed scored-index intersections
  with counts published (B.9.2 discipline).

### D.3 Outcome branches (pre-written)

- Both reads PASS: the B.9.1 caveat is retired from all materials and
  replaced by the measured one-configuration claim; this is public
  artifact #2 of the lab cadence.
- Read 1 FAILS: stated verbatim in every material that carried the
  payload economics: "document-only quality does not survive; the
  serving-cost claim applies only with demos included, whose payload
  advantage is smaller and is republished accordingly." No spin.
- Read 2 alone fails while Read 1 passes: retention holds but the
  baseline comparison weakens; reported as such, the wedge re-scoped.

### D.4 Execution

B.7-r5/r6 machinery verbatim: arm-scoped checkpointed kernels, doc
journals, banker-style first-terminal-wins, resumed stamps. Estimated
runtime: seeds 1/3 decode-only; seed 2 adds one adapt. All on the five
free CPU slots.

### D.5 Comparability arm (declared 2026-08-19T04:20Z, before any run;
### NOT a decision arm)

**doczero** — the same document-only decode with NO adapter and NO
demos (epochs 0, base 0.5B, bare document prompt), per seed. Purpose:
isolate the adapter's full contribution in the serving configuration
(doconly − doczero) and give Read 1/2 their floor context. Explicitly
comparability-only under the B.3 discipline: whatever these arms show,
they cannot pass or fail Addendum D — the decision reads are D.2's, as
frozen. Artifacts: novel_schema_d_0.5b_k30_seed{n}_doczero_2026-08-18.json
(same D date; the arm inherits D's freeze context). Runs on the idle
CPU slots beside the gating arms.

## Addendum E — Diverse-geometry gate (frozen 2026-08-19T04:55Z, before any run)

The B.9.5 escalation, discharged as its own preregistered gate: does
the Addendum B effect survive when the schema SHAPE itself varies per
tenant, not just the vocabulary? Frozen before any diverse-geometry
document has ever been generated for evaluation.

### E.1 The single changed variable

geometry="diverse" in the corpus generator (committed and tested before
this freeze; fixed mode proven byte-identical to the banked B tenants):
group count 2-4, field count 6-12, per-field value kinds and group
assignments all seed-derived — every tenant a different shape. All else
at B's frozen values: 0.5B, k=30, eval_n 60, cpu/fp32, epochs 1,
LoRA r16/a32, scorer, pairing, B.5 validity windows, B.3 two-statistics
rule, B.9.2 attrition discipline.

### E.2 Decision

Seeds {101,...,106} — six NEW pairs (twice B's seed count; the seed
range is disjoint from every prior experiment). PASS iff the seed-mean
paired delta (adapted − kshot) is ≥ +5.0 F1 with interval and sign test
agreeing. Cluster CI over 6 seeds reported beside the receipt level.
A pass retires the shared-geometry objection with n doubled; a fail is
reported as the boundary of the effect (novelty may be carried by the
fixed shape) — either outcome is publishable and the branches bind.

### E.3 Execution

B.7-r5/r6 arm-scoped checkpointed kernels, one arm per kernel, launched
in waves on free CPU slots as the Addendum D board drains (12 arms
total; ~2-3 waves). Artifacts:
novel_schema_e_0.5b_k30_seed{n}_{arm}_2026-08-19.json.

### D.6 Interim observation (2026-08-19 03:25Z — seed 3 landed; NOT a verdict)

Seed-3 doconly: 0.0000 mean F1, 0/60 valid JSON, 60/60 scored (no
decode failures). Local reproduction with the restored adapter captured
the mechanism: given a bare document, the adapted model produces a
prose summary of the document's fields ("Based on the information
provided, I can summarize...") rather than the tenant JSON. The
adapter, trained exclusively on LOO sequences WITH demonstration
context (text_ttt_training_examples), encodes the schema mapping as
context-conditioned behavior; a bare document elicits the base model's
assistant prose. The D.2 reads bind only when all three seeds land, but
the mechanism makes the direction unambiguous. If confirmed, D.3's
failure branch publishes verbatim, and the corrected experiment below
is already frozen.

## Addendum F — Document-only-TRAINED adapters (frozen 2026-08-19T03:30Z, before any run)

The engineering response to D.6, preregistered within hours of the
observation: train the adapter ON the serving configuration. For each
train pair, the training sequence is (user: document) -> (assistant:
tenant JSON) — no LOO context, no demos. Decode: document-only, as in
D. Everything else at B's frozen values (0.5B, k=30 corpus seeds
{1,2,3}, eval_n 60, cpu/fp32, LoRA r16/a32, scorer, B.5/B.9.2
discipline). Artifacts:
novel_schema_f_0.5b_k30_seed{n}_docadapted_2026-08-19.json, carrying
raw predictions (primary-evidence storage, as Addendum E).

### F.1 Reads (declared now)

- Read 1: docadapted (doc-only trained, doc-only served) vs the banked
  B kshot-with-prompt arm — PASS iff seed-mean delta ≥ +5.0 F1,
  interval and sign test agreeing. A pass IS the unified claim
  (quality + 20-58x payload in one measured configuration), replacing
  the failed D path.
- Read 2 (comparability, non-gating): docadapted vs B
  adapted-with-prompt — how much of the demo-context quality the
  serving-trained adapter retains.
- Failure branch, pre-written: if Read 1 fails, the honest public
  statement is that per-tenant adaptation of a 0.5B requires demos at
  serving time in BOTH training regimes tested; the payload economics
  are then republished on the demos-included payload ratio only, and
  the unified claim is retired until a larger rung or a different
  recipe passes it.

### F.2 Execution

New training-example builder (doc-only), unit-tested before any run;
arm-scoped checkpointed kernels on free CPU (adapt ~3h + decode per
seed); B.7-r6 machinery; OTS-anchored spec hash covers this freeze.

### F.3 Interim observation (2026-08-19 06:15Z — seed 1 landed; NOT a verdict)

f-s1 docadapted: 0.9407 mean F1, 60/60 valid JSON, 37/60 exact;
adapt time 271s (vs ~3h for LOO-context adapters — doc-only training
sequences are ~40x cheaper). Against banked seed-1 arms: +32.0 over
kshot-with-prompt (F.1 Read 1 bar: +5), −4.1 vs adapted-with-prompt
(Read 2 comparability). First prediction-carrying artifact:
PRIMARY-VERIFIED — all 60 stored predictions re-scored against
regenerated gold match recorded scores. F.1 binds when seeds 2 and 3
land.

### D.7 VERDICT: FAIL (decided 2026-08-19 06:33Z, all three seeds banked)

Read 1 (retention): seed-mean −98.4 F1 vs the −5.0 bar — FAIL.
Read 2 (unified claim): seed-mean −51.9 F1, 0W/155L/3T — FAIL.
D.5 comparability: adapter contribution over no-adapter is +0.0 — in
document-only serving, the demo-trained adapter does nothing.
Per D.3, stated verbatim in every material that carried the payload
economics: "document-only quality does not survive; the serving-cost
claim applies only with demos included, whose payload advantage is
smaller and is republished accordingly." Mechanism: D.6 (prose
collapse; adapters trained with demonstration context encode the
schema as context-conditioned behavior). The corrective, preregistered
before this verdict (Addendum F, doc-only-TRAINED adapters), has its
first seed banked at 0.9407 with 60/60 valid JSON (F.3 interim,
PRIMARY-VERIFIED); F's verdict binds on seeds 2 and 3.

### F.4 Interim observation (2026-08-19 06:35Z — seed 2 landed; NOT a verdict)

f-s2 docadapted: 0.5282, 59/60 valid JSON, adapt 272s,
PRIMARY-VERIFIED. Against banked seed-2 arms: +2.5 vs kshot (below the
per-seed pace of the +5 bar), −46.5 vs adapted-with-prompt. Two honest
notes recorded before seed 3 decides the read: (a) the doc-only arm
scored all 60 documents where the B arms excluded 22 for length — the
serving mode reaches documents the demo-stuffed prompt cannot; the F.1
read runs on scored-index intersections as frozen; (b) the seed spread
(0.9407 vs 0.5282) suggests doc-only training quality depends on
document length/complexity — a real scoping finding either way.

### F.5 VERDICT: PASS (decided 2026-08-19 06:37Z, all three seeds banked, all PRIMARY-VERIFIED)

Read 1 (unified claim, docadapted vs banked kshot-with-prompt on
scored-index intersections): seed-mean +24.0 F1 (per-seed +32.0/+5.5/
+34.7) vs the +5.0 bar; receipt-level mean +26.6, CI low +22.3 > 0;
sign test 126W/19L/13T — all three frozen conditions met. PASS.
Read 2 (comparability): −22.4 seed-mean vs adapted-with-prompt
(−4.1/−43.5/−19.7) — the measured quality cost of document-only
serving, strongly seed-dependent. Honest scoping carried with the
claim: seed-2 (the long-document tenant) passes marginally (+5.5 on
its n=38 intersection) and pays the largest retention cost; the
serving mode's sweet spot is compact schemas. Adapt cost in this
regime: ~272s per tenant (~40x cheaper than LOO-context training).
The B.9.1 caveat is hereby retired per D.3/F.1: the quality claim and
the 20-58x payload economics hold in ONE measured configuration, with
raw predictions published and independently re-scorable.

### E.4 UNINFORMATIVE + Amendment E-r1 (frozen 2026-08-19T06:45Z, before any E-r1 run)

Seed-101 adapted arm: 60/60 "no completion", adapt_seconds=1.0 — every
LOO training sequence AND every decode prompt exceeded the frozen
8192-token budget. The E.1 diverse ranges (fields up to 12, distractors
up to 7) are structurally unmeasurable at k=30 under B's frozen
max_sequence_tokens for large-geometry draws. Per the B.6 uninformative
branch: reported as such (the artifact stays banked as the record; it
is not evidence either way), and the corpus is re-scoped BEFORE any
further diverse data:

**E-r1:** geometry "diverse-compact" — groups 2-3, fields 6-9,
distractors 3-5 (still shape-varying per tenant, unlike fixed mode);
seeds {201..206} (fresh range; the 101-106 series is retired to avoid
mixing protocols); every other value unchanged from E.1/E.2 including
the +5 bar, both-statistics rule, and primary-evidence storage. The
in-flight 101/102-series kernels are cancelled — their data would be
unpoolable under the amendment.

### E-r2 (frozen 2026-08-19T06:50Z, before any E-r2 run) — tokenizer-verified seeds

E-r1's rough character-based budget estimate was wrong (recorded
against ourselves): pseudoword corpora tokenize at ~1.4 chars/token,
and the production tokenizer shows B's own fixed corpus ran at 8,068
of 8,192 tokens — 124 of headroom. E-r1 seeds 201/202 measured
8,438-8,707 tokens (201: LOO max 8,536, decode 8,693-8,707; 202: LOO
max 8,438, decode 8,593-8,609) and produced the same all-excluded
failure as E.4. [Corrected 2026-08-19: the range first written here,
8,539-8,700, was a transcription slip; the measurement is now banked
and rerunnable — `experiments/novel_schema_er_seed_screen_2026-08-19
.json`, regenerated by `scripts/screen_er_seeds.py`.]

Fix, with the screen stated precisely so it cannot be mistaken for
cherry-picking: candidate seeds ascending from 201 are screened by
TOKEN COUNT ONLY (decode prompt and max LOO training sequence both
≤ 7,900 with the production tokenizer) — a measurability screen, blind
to content and to any model output; no arm had run on any accepted
seed at screening time. First six fitting: **{203, 204, 206, 207,
208, 209}** (fields 6-7, groups 2-3 — shapes still vary). All other
E-r1 values unchanged, including the +5 bar and both-statistics rule.
The screen itself is banked as
`experiments/novel_schema_er_seed_screen_2026-08-19.json` (all nine
candidate seeds, measured maxima, eligibility) and reproduces the
frozen set exactly: 201/202/205 excluded, {203,204,206,207,208,209}
eligible.

## Errata & provenance amendments (2026-08-19, from the line-by-line audit)

A full line-by-line audit of this spec against the repository's git
history, OTS anchors, and artifacts (every finding independently
reproduced before being accepted) surfaced the items below. Frozen
preregistration text is never rewritten; corrections land here, dated,
per A.5's own rule.

**P1 — Stamp provenance.** Doc-internal UTC freeze stamps were written
as forward-rounded declaration times and systematically postdate the
git commits that introduced the text by 2-16 minutes (e.g. D.5
"declared 04:20Z" committed 00:17:50Z; Addendum E "frozen 04:55Z"
committed 01:03:10Z; D.6 03:25Z / F 03:30Z committed 03:18:49Z; E.4
06:45Z committed 06:41:39Z; E-r2 06:50Z committed 06:47Z). The error
is in the conservative direction — every freeze provably EXISTED
EARLIER than its stamp — and the authoritative ordering evidence is
(1) git commit timestamps and (2) the Bitcoin-anchored snapshots:
the 0119Z snapshot already contains D.5 and Addendum E; the 0330Z
snapshot already contains D.6 and Addendum F. Treat doc stamps as
approximate declaration times; treat commits + OTS as the record.

**P2 — D.7's comparability line predates dz-s3.** When D.7 was decided
(06:33Z) the D.5 comparability read rested on doczero seeds 1 and 2;
seed-3 doczero banked at 15:07Z the same day and confirmed the same
0.0000 (artifact novel_schema_d_0.5b_k30_seed3_doczero_2026-08-18
.json). The verdict was unchanged by the third seed.

**P3 — Addendum A wording edits after its freeze.** Three passages of
Addendum A were edited in place on 08-10/08-11 (before any scaled-run
artifact existed, first artifacts 08-11 20:15Z): A.1 evidence 1
("honored in the strategy docs" -> "honored in all materials"), A.4
branch 1 (document names generalized), A.4 note on arm ordering.
Wording only; no bar, seed, n, or decision rule changed. Recorded
here because in-place edits to frozen text — however cosmetic —
violate the freeze discipline this document depends on.

**P4 — Addendum A execution vs design.** Of the 54 frozen arms, 30 ran
to score (0.5B k5/k10, 1.5B k10, 4B k5/k10 x seeds x arms), six 4B
k=30 attempts are banked as OOM error records, and the remaining 18
curve arms (0.5B k30, 1.5B k5, 1.5B k30) were never launched after
the gate had already failed at all three scales. No curve claim was
or is asserted. The k=30 seed-1 adapted CORD arm promised in the §A
preamble never completed; no artifact exists and none is claimed.

**P5 — B.7-r6 test claim.** The clause "Resume path covered by an
offline journal-reconstruction test" (line ~781) was written ahead of
the test, which did not exist at audit time. The resume path has now
been exercised operationally three times in production (er-s203k,
er-s204a, er-s204k on 2026-08-19, all recovered and completed); an
offline test is owed and tracked. The claim as originally written was
false and is retracted until the test lands.

**P6 — B.9.3 precision.** "53/60 of their per-document scores are
identical by construction" overstates: 37/60 are identical by
construction (the shared checkpoint journal); 53/60 agree in total,
the other 16 matched in independent decodes.

**P7 — §3.6 filename.** The planned smoke artifact name was never
used; delivered names are cord_smoke_2026-08-08.json and
cord_smoke_baseline_2026-08-08.json.

**P8 — Internal citations.** SCALE_EVIDENCE_ROADMAP.md, REDTEAM.md,
FINAL_PICK.md, AGENTS.md and similar names cited in this spec are
internal planning documents not shipped in the public cut; where a
number rests on one (e.g. the $50-150K funded-rung budget envelope),
the number is restated here as the claim of record.

**P9 — Sign-test tie conventions.** Two conventions coexist:
cord_paired_power.py / novel_schema_summary.py count |delta| <= 0.01
as ties (all their published counts recompute only under that rule);
verify_verdict.py and read_addendum_d.py use exact ties (1e-12).
Published counts state which convention produced them; the D.7 Read-2
count is 0W/155L/3T under the exact convention (ties now reported).

**P10 — D/E/F artifact provenance stamps.** Primary-evidence artifacts
for Addenda D/E/F carry a "spec" field naming Addendum B (the corpus
construction they inherit); their governing gates are D (frozen 08-18
T18:05Z), E/E-r2 (08-19), F (08-19T03:30Z). Mapping: novel_schema_d_*
-> Addendum D; novel_schema_e_* -> E/E-r2; novel_schema_f_* -> F.
Artifact bytes are not rewritten post hoc.

**P11 — F.4's length/complexity conjecture tested and NOT supported;
seed-2 "long-document tenant" characterization corrected (2026-08-20).**
Computed from stored F raw predictions + the regenerated corpus
(`experiments/novel_f_length_analysis_2026-08-20.json`,
`scripts/tenant_analysis.py`): documents, gold objects, and schemas are
statistically identical across the three F seeds (test docs ~150
tokens, gold 8 leaves, train docs ~208 chars for every seed), and
within-seed short/long halves score the same — so document
length/complexity does not explain seed 2's 0.5282. The B-arm
exclusions likewise are not "long documents": every seed's k=30
LOO demo prompt measures 98-100% of the frozen 8192-token budget
(seed 1: 8062-8075; seed 2: 8184-8195 — straddling the cap; seed 3:
8032-8043), so which documents were excluded is tokenization jitter at
the budget edge, and seed 2 is "the tenant whose prompts land ON the
budget line," not a long-document tenant. Two consequences, stated per
the claim rule: (a) demo-context serving at k=30 operates AT its
context ceiling for every tenant on this corpus, while doc-only
prompts run ~150 tokens (~54x headroom) — a measured structural limit
of context-carried quality; (b) the mechanism behind seed 2's doc-only
quality gap is OPEN — the F.4 conjecture is withdrawn, not replaced.

**P12 — B.9.4's receipt CI, and an unrecorded in-place edit (2026-08-21).**
(a) The receipt-level CI read [42.9, 49.4] in B.9.4 and [42.8, 49.4] in
the banked artifact. Cause: `verify_verdict.py` used the normal quantile
(1.96) for the receipt interval while using a t quantile for the cluster
interval in the same function, and `novel_schema_summary` — the
authorized reader that produced the artifact — used t throughout. The
0.033 F1-point gap sat inside that script's own 1e-3 cross-check
tolerance, so nothing ever failed. The script now uses t for both; the
correct value is **[42.8, 49.4]**, and `tests/test_readers_agree.py`
pins the readers together so a future divergence is loud.
(b) The B.9.4 figure was edited in place from 42.8 to 42.9 on 2026-08-19
with no erratum — precisely what P3 forbids. The frozen text stands as
written and this entry is the correction, per A.5.

**P13 — P11(a) rescoped (2026-08-21).** P11 stated that demo-context
serving at k=30 "operates AT its context ceiling for every tenant on this
corpus." That holds for the fixed-geometry B corpus (98-100% of the 8192
budget) but not universally: the E-r2 tenants, screened for budget fit
before any arm ran, measure 6,619-7,447 decode tokens (81-91%). The
ceiling binds where the shape is fixed. The related product limit stands
and is worth stating plainly: a 12-field schema at k=30 overflows the
frozen budget entirely (seed 101 returned 60/60 no-completion), so
demo-context serving cannot reach wide-schema tenants at that k at all.

**P14 — the quantile function was wrong for every small cluster, and an
outside reader found it (2026-08-21).** `t95()` in `novel_schema_summary`
— the authorized reader, and therefore the estimator behind every
interval this spec publishes — was a lookup table whose smallest entry
was df=39, with a fallback returning the invented constant `2.09` for
anything below it. Consequences, in order of severity:

(a) **Addendum E's cluster interval (E.5) is corrected from
[+32.75, +47.95] to [+31.01, +49.70].** E clusters over six seeds, df=5,
where t(0.975, 5) = 2.5706; the published interval was computed at 2.09,
roughly t at df≈19, making it **~19% too narrow and too narrow in the
direction that flatters the result**. The Addendum E verdict is
unchanged: the decision rule is the seed-mean against the +5 bar, the
corrected lower bound +31.0 clears it six-fold, and the sign test and
receipt interval are unaffected. Per A.5 the frozen text stands and this
entry is the correction.

(b) **P12(a) is superseded.** P12 recorded the k=30 receipt CI as
"[42.8, 49.4]" after repairing a normal-vs-t inconsistency. That repair
transcribed the constant 1.980 in place of t at df=157 (1.9752), which
is the entire reason the figure moved to 42.8. With the quantile computed
rather than copied, the correct value is **[42.9, 49.4]** — the same
value the frozen B.9.4 text carries, reached for a different reason than
P12(b)'s unlogged edit assumed. Three successive published values of one
number, each wrong for its own reason, every one of them a copied
constant.

(c) **Remedy, at the level of the class rather than the instances.** The
table is deleted. `t95()` computes the exact Student-t quantile by
bisecting its CDF over a continued-fraction regularized incomplete beta
(stdlib only, so the verify path still runs with nothing installed), and
`verify_verdict.py` imports it instead of carrying constants, so the
project has exactly one estimator. `tests/test_readers_agree.py` checks
the computed quantiles against published tables and fails on the
*presence of any hardcoded quantile* in either reader — the cause, not
the symptom.

Finder credit: reported by an outside reader who recomputed the published
interval from the artifacts rather than accepting it, per the standing
offer in `CORRECTIONS.md`.

---

# Addendum G — the adaptation-headroom law (frozen 2026-08-21, before any G data exists)

## G.1 Where this hypothesis came from, stated plainly

This addendum is **not** a fresh idea. It is a POST-HOC pattern noticed
in already-banked data on 2026-08-21, and it is written down here
*before* it is tested precisely because a post-hoc pattern is worth
nothing until it has survived a preregistered test.

The pattern: across every result this project has, the paired advantage
of per-tenant adaptation over in-context prompting at 0.5B appears to be
a **decreasing function of how well the prompted baseline already does.**

| Corpus | Prompted baseline | Paired delta |
|---|---|---|
| Novel-schema synthetic (Addendum B, k=30) | 0.4333–0.6208 | **+46.5** |
| Freight waybills, hard tier | 0.5218 | **+17.5** |
| Freight waybills, medium tier | 0.7200 | **+21.1** |
| Freight waybills, mixed tier | 0.9444 | **+0.0** |
| Freight waybills, easy tier | 0.9750 | **−2.5** |
| CORD receipts (Addendum A) | base model already knows the domain | **−7.3 / −11.5 / −4.5 (FAIL)** |

If true, this single rule explains BOTH of this project's published
failures and both of its passes, which is exactly why it deserves
suspicion rather than celebration: a story that explains everything
after the fact is the easiest kind to fool yourself with.

## G.2 The confounder this test must survive

**Ceiling effects trivially produce this correlation.** A baseline at
0.975 cannot lose more than 0.025 and cannot gain more than 0.025, so
any measure of raw delta MUST shrink as the baseline rises, whether or
not adaptation is doing anything interesting. A reader who did not spot
that would be right to discount the whole addendum.

So this addendum reports two quantities and is explicit about what each
one can and cannot establish:

- **Raw delta.** What a buyer actually gets. Ceiling effects are part of
  their reality, not an artifact to be removed. This is the gating
  statistic.
- **Captured-headroom fraction**, `(adapted − baseline) / (1 − baseline)`.
  What survives the ceiling objection. If this is FLAT across the
  baseline range, the law is a ceiling effect and this spec will say so
  in those words — still a true and actionable buying rule, but NOT
  evidence that adaptation is "better at hard documents." If it RISES as
  the baseline falls, adaptation is capturing more of the available room
  where prompting is weak, which is a stronger claim.

The captured-headroom fraction is **reported, not gating.** No threshold
is preregistered for it, because we do not have a principled prior for
one and inventing a number here would be fake precision.

## G.3 Design, frozen

- **Tenants:** three fresh seeds, **401, 402, 403**, never used in any
  prior addendum.
- **Baseline-strength dial:** demonstration count **k ∈ {1, 3, 10, 30}**.
  Fewer demonstrations makes the prompted arm weaker; this is the
  cleanest available way to sweep baseline strength while holding the
  corpus, the schema and the scorer fixed.
- **Arms, per (seed, k) cell:** the prompted baseline, and the adapted
  arm carrying the SAME k-shot prompt (spec B.9.1 scoping — adaptation
  measured ON TOP of prompting, never against a bare model).
- **Documents:** n_test = 20 per cell, disjoint from the demonstrations.
- **Decode:** GREEDY (samples=1) on BOTH arms, matched. A richer decode
  on one side would credit adaptation with the decode difference.
- **Scorer:** `score_text_output`, mean per-document micro-F1, invalid
  JSON scored 0 — unchanged from every other addendum.
- **Cells:** 3 seeds × 4 k = **12**.

## G.4 Preregistered readings

Computed by `scripts/addendum_g_summary.py`, which is the only
authorized reader of G arms.

- **(a) PRIMARY.** Spearman rank correlation between a cell's prompted
  baseline mean and that cell's raw paired delta, over all 12 cells,
  is **≤ −0.60**.
- **(b) SECONDARY.** Mean raw delta over the four lowest-baseline cells
  minus mean raw delta over the four highest-baseline cells is
  **≥ +10.0 F1**.
- **PASS** requires (a) AND (b).
- **PARTIAL** if exactly one holds; the addendum reports which, and does
  not round the result up to a pass.
- **REFUTED** if neither holds. In that case this spec states that the
  unifying story is not supported by fresh data, the post-hoc table in
  G.1 is labelled as an artifact of the corpora it came from, and
  nothing downstream may cite the law.
- **UNINFORMATIVE** if fewer than 10 of the 12 cells produce scoreable
  paired arms, or if every prompted baseline lands within a 0.10 band
  (the dial failed to move baseline strength, so the correlation is not
  measurable regardless of what it computes).

## G.5 What a PASS would and would not license

Licensed: "the measured value of per-tenant adaptation at 0.5B is
predictable in advance from a cheap measurement of the prompted baseline
on the tenant's own documents." That is a buying rule and a
qualification step, and it is what the product would sell.

NOT licensed by G alone: any claim about scales above 0.5B (Addendum H),
any claim about real customer documents (every G corpus is synthetic),
and any claim that adaptation is *better at hard documents* unless the
captured-headroom fraction in G.2 rises rather than staying flat.

## G.6 Ordering

This text is committed to git before any G arm is run, and the commit
that adds it contains no G artifacts. That is the check: if an
`experiments/novel_schema_g_*` file predates this section in git
history, the preregistration is void and this spec says so.

## G.7 Measurability amendment (2026-08-21, ZERO G cells banked)

The G.3 design as first frozen is **not measurable on the compute this
project has.** The k=30 cells need ~35 minutes each on the 4-core CPU
box, and this container has been restarting under sustained load roughly
every 20–40 minutes, so the twelve-cell sweep never reaches its end.
Three separate launches produced zero completed cells.

Amended, and the amendment is what changed rather than a rewrite:

- **Demonstration counts: k ∈ {1, 3, 10, 30} → k ∈ {1, 2, 4, 8}.**
- Everything else in G.3 stands: seeds 401/402/403, n_test = 20, both
  arms carrying the same k-shot prompt, matched greedy decode, the same
  scorer, 12 cells.
- Every bar in G.4 stands unchanged: ρ ≤ −0.60, tercile gap ≥ +10.0 F1,
  ≥10 scoreable cells, ≥0.10 baseline spread.

**Ordering, which is the only thing that makes this legitimate.** At the
time of this commit `experiments/novel_schema_g_*` matches nothing, in
the working tree and in all of git history. The bars were not moved; the
dial's rungs were. Check it:

    git log --oneline --all -- 'experiments/novel_schema_g_*'

If that command prints anything dated before this section, the
amendment is void and G must be rerun under the original G.3.

**Why this dial still tests the same claim, stated so the narrowing is
not smuggled.** The measured k=1 baseline is ~0.12 and the banked k=30
baselines are 0.4333–0.6208, so k ∈ {1,2,4,8} spans roughly 0.12 → 0.45:
a spread of ~0.33, comfortably past the 0.10 minimum the reading needs.
What is LOST is the high-baseline end of the range — the regime where
the waybill tiers showed adaptation adding nothing. So a PASS here is
evidence that the delta *falls as the baseline rises within the weak
regime*, and is **NOT** evidence about the saturated regime, where the
post-hoc table's most interesting rows (mixed 0.944 → +0.0, easy 0.975 →
−2.5) live. G.5's list of things a pass does not license is extended
accordingly: **G may not be cited as showing that adaptation stops
paying once prompting is strong.** That claim needs the high-k arms, and
they are deferred to GPU compute rather than quietly dropped.

A second, smaller gain from the amendment, recorded because it cuts in
our favour and should therefore be stated explicitly rather than
discovered: k ≤ 8 keeps every prompt far inside the 8192-token budget,
so the token-budget attrition that cost Addendum B twenty-two receipts
cannot arise here at all.

## G.8 The dial was confounded — second amendment (2026-08-21, ONE cell banked, discarded)

**Disclose the awkward part first, because a reader will find it and
should not find it from them.** This amendment was written after the
first G cell returned, and **it makes a PASS more likely, not less.**
That is the single most suspicious shape an amendment can have, so
everything about it is on the record.

### The defect

`k` was doing two jobs at once. In the G.3 design, `n_train = k` sets
both the number of demonstrations in the prompt AND the number of pairs
the adapter trains on (`adapt_text` runs leave-one-out over
`task.train`). So a low-k cell has a weak baseline *and* a barely-
trained adapter. The dial does not isolate baseline strength; it moves
adaptation strength with it, in the same direction.

That is a property of the code, not of any result. It was true when G.3
was written and would have been true whatever the first cell said.

### Why the timing still matters, and which way it cuts

The confound **suppresses** the effect G is testing for: it shrinks the
delta exactly where the law predicts the delta should be largest. The
first cell showed that plainly — `seed 401, k=1: baseline 0.3572 →
adapted 0.3667, delta +0.0095`, a weak baseline with almost no gain,
which is what a one-example adapter produces.

So removing the confound removes an obstacle to a pass. **Amending in
the direction that helps us, after seeing data, is exactly what
preregistration exists to prevent.** The mitigations, all of them
checkable:

1. **The bars in G.4 are untouched.** ρ ≤ −0.60, tercile gap ≥ +10.0 F1,
   ≥10 cells, ≥0.10 spread. Not one threshold moved.
2. **The cell run under the flawed design is not deleted.** It is moved
   to `experiments/superseded_novel/GA_COUPLED_k1_seed401_2026-08-21.json`,
   outside the reader's glob, and it is cited here.
3. **The coupled design is deferred, not disowned.** "Does adaptation
   still pay when the tenant can only supply a handful of labelled
   pairs?" is a real product question — arguably a more commercially
   important one than G's — and the coupled sweep is the right test for
   it. It is deferred to GPU compute, not quietly dropped.
4. **If G-b passes, the published claim must carry this section's
   number.** A pass says the law holds *with the adaptation set held
   constant*. It says nothing about the regime where a tenant supplies
   one or two examples, where the one cell we have shows +0.0095.

### The amended design (G-b)

- **Adaptation set: FIXED at 8 pairs** for every cell. Every adapter
  sees the same amount of training signal.
- **Prompt demonstrations: j ∈ {0, 1, 2, 4, 8}**, drawn from those same
  8 pairs, applied identically to BOTH arms.
- Seeds 401/402/403, n_test = 20, matched greedy decode, same scorer.
  **15 cells.**
- j = 0 is the document-only prompt on both arms — the Addendum D/F
  serving configuration — and it is the weakest-baseline rung.
- Everything in G.4 and G.5 stands, including the G.7 restriction that a
  pass may not be cited as showing adaptation stops paying once
  prompting is strong.

Artifacts are named `novel_schema_gb_*` so nothing from the coupled
design can enter the G-b computation by filename accident.

## G.9 j=0 is unmeasurable here; substituted, not dropped (2026-08-21, ZERO G-b cells banked)

**FINAL amendment to G. Three in one day is already more than a
preregistration should need, and the count is part of the record.**

j = 0 is the bare-document prompt. On this box it took **over forty
minutes of CPU for two documents**: with no demonstrations the model
emits prose until it hits the 512-token cap, on every document, in both
arms. A fifteen-cell sweep containing that rung does not finish.

It is also the rung whose answer this project already published.
**Addendum D** measured exactly this configuration and found both arms
collapse — 0.0000 F1, 0/60 valid JSON, adapter contribution +0.0. So
j = 0 would contribute a (baseline ≈ 0, delta ≈ 0) point that is not a
new measurement, only D repeated at greater cost.

**And removing it helps a pass**, because a zero-delta point at the
weakest baseline argues against the law. That is the second amendment in
a row that cuts our way, and it is stated here rather than left to be
noticed.

So it is **substituted, not dropped**:

- **Adaptation set: 8 → 16 pairs, fixed** across all cells.
- **Prompt demonstrations: j ∈ {0,1,2,4,8} → j ∈ {1, 2, 4, 8, 16}.**
- Still 3 seeds × 5 rungs = **15 cells**; n_test = 20; matched greedy
  decode; same scorer; **every bar in G.4 still untouched.**

The substitution is deliberately not a narrowing. It trades a rung whose
result is already banked (j=0, both arms fail) for one that extends the
sweep toward the **strong-baseline** end — the end G.7 had to give up.
That end is where the law makes its risky prediction: if adaptation
keeps paying at j=16, the correlation weakens and G fails. A rung that
can hurt us replaces a rung that could not inform us.

**What the amendment count means for citing G.** Three amendments, two
of them in our favour, all made before their data existed and all
recorded here. A reader who discounts G for that is reasoning correctly.
The claim G can support is therefore stated narrowly wherever it
appears: it is a **within-generator, 0.5B, fixed-adaptation-set** result
about a synthetic corpus, and it is not a substitute for the one thing
still missing everywhere in this repository — a real tenant's documents.

## G.10 UNRUN — compute-bound, with the cost measured rather than estimated (2026-08-22)

**Addendum G has produced no verdict, and the reason is arithmetic, not
nerve.** Zero G-b cells were banked. This section exists so that the
absence is on the record with a number attached, because "we designed an
experiment and never mention it again" is how inconvenient results get
buried, and nobody outside could tell the difference.

### The measured obstacle

Per-tenant adaptation here uses the leave-one-out corpus
(`text_ttt_training_examples`): with *n* labelled pairs it builds *n*
training sequences each carrying *n−1* demonstrations, so adaptation
cost grows as **O(n²)** in the pair count, not O(n).

Measured on this box, from the two superseded coupled-design cells that
did complete:

| pairs | demo-units (n·(n−1)) | adaptation, measured |
|---|---|---|
| 1 | 0 | **5.3 s** |
| 4 | 12 | **1112.4 s** (18.5 min) |

That is ~93 s per demo-unit. G-b fixes the adaptation set at 16 pairs =
240 demo-units, so **one arm's adaptation is ~371 minutes**. Fifteen
cells, two arms each, is on the order of a hundred hours of CPU on a
4-core box whose container recycles every 20–40 minutes. The design is
not slow here; it is impossible here.

### Why this is not fixed by a fourth amendment

The obvious fix is to adapt with the document-only recipe
(`text_docmode_training_examples`, Addendum F), which is O(n) — the
blind rehearsal adapted on 20 pairs in **54 seconds** that way. But that
would change the *adaptation recipe itself*, not a knob: Addenda B and E
were measured with the LOO recipe, and swapping it would make G
non-comparable with the results whose pattern it exists to test.

G.9 also said, in as many words, that it was the final amendment.
Breaking that one section later would confirm exactly the churn a reader
should already be worried about after three amendments in a day. **The
count stops here.**

### Status and what would change it

G is **UNRUN**, deferred to GPU compute. Everything else about it stands
and is checkable: the bars (G.4) were frozen before any data; the
confound (G.8) and the unmeasurable rung (G.9) are recorded with their
disclosure that both cut in our favour; the reader
(`scripts/addendum_g_summary.py`) exists and returns **UNINFORMATIVE**
on fewer than 10 cells, which is what it does today.

The two cells that did complete under the *superseded coupled* design
are preserved at `experiments/superseded_novel/GA_COUPLED_*` and are
**not** evidence for or against the law — the coupled dial moved
adaptation strength along with baseline strength, which is precisely why
it was superseded.

**Nothing anywhere in this repository may cite the adaptation-headroom
law.** The post-hoc table in G.1 remains what it was on the day it was
written: a pattern across four corpora, generated after the fact, never
tested. It is a reason to run G, not a result.

## G.11 The two coupled cells point AGAINST the law, and G.10 cited one of them for its runtime only (2026-08-22)

**This section exists because an outside reader found that the previous
section used one attribute of a file while withholding another, and the
withheld one cuts against us.** That is the exact failure
`CORRECTIONS.md` had already named — *"a disclosure whose honest half was
added while the misleading half stayed"* — committed by the person who
wrote that sentence, four sections later.

### The numbers that were not stated

Both superseded coupled-design cells, in full:

| cell | prompted baseline | adapted | paired delta | captured headroom |
|---|---|---|---|---|
| k=1, seed 401 | 0.3572 | 0.3667 | **+0.0095** | 0.0148 |
| k=4, seed 401 | 0.5663 | 0.9187 | **+0.3524** | 0.8126 |

The adaptation-headroom law predicts the paired delta **falls** as the
prompted baseline **rises**. Observed: the baseline rises (0.357 →
0.566) and the delta rises with it, by a factor of **37**. On the
captured-headroom fraction — the statistic G.2 introduced specifically
to survive the ceiling-effect objection — it rises from 0.015 to 0.813.

**Both cells that exist point the opposite way to the hypothesis, on
both statistics.**

### What G.8 and G.10 did with that

G.8 justified discarding the coupled design by arguing the confound
*"suppresses the effect G is testing for… The first cell showed that
plainly."* That reading survives only while k=1 is the only cell. The
k=4 cell completed at 02:08 the following morning and inverts it.

G.10 then cited that same k=4 cell — **for its adaptation runtime
(1112.4 s), in a table used to argue that G is compute-bound.** Its
result appears in no narrative document in this repository. Using a
file's timing while omitting its finding is disclosure operating as a
controlled instrument, and it is not defensible by pointing at the
sentence that says the cells are "not evidence for or against."

### What this does and does not establish

It does **not** refute the law. Two cells, one seed, and the coupled
design is genuinely confounded — adaptation strength moved with baseline
strength, so k=4's adapter saw four times the training signal of k=1's,
which is a sufficient explanation for the rise on its own. That is
exactly why the design was superseded, and the reasoning in G.8 stands
on its own terms.

It does establish two things that are now on the record:

1. **The only G-family data that exists runs against the hypothesis.**
   Anyone citing the G.1 post-hoc table must also carry this table. The
   prohibition in G.10 — nothing in this repository may cite the law —
   is reinforced, not weakened.
2. **The G.8 rationale was written from a one-cell view and reads
   differently with both cells in hand.** It is not withdrawn, because
   the confound is real regardless of direction; it is annotated, and
   this annotation is the correction.

### The rule that should have prevented it

Stated as a binding constraint rather than an intention: **any artifact
cited in this spec for any attribute must have its primary result stated
in the same section.** A runtime, a hash or a file size may not be
quoted from a cell whose outcome is left unmentioned. `CORRECTIONS.md`
carries this as a defect found in our own disclosure path, and it was
found by a reader, not by us — which is the fourth consecutive
consequential finding to arrive from outside.

---

# Addendum H — the corpus-DIFFICULTY ablation (preregistered 2026-08-22, before any arm exists)

## H.1 Why this exists, stated as the objection rather than as our answer

An outside technical reader, asked to find the objection that would end a
partner meeting, found this one, and it is better than anything we had
raised against ourselves:

> Addendum E answers whether the effect is an artifact of one fixed
> corpus **shape**. Nobody has asked whether it is an artifact of one
> fixed corpus **difficulty** — and the difficulty is set by two
> hard-coded constants that have never been ablated anywhere in this
> repository.

The constants are in `src/arcttt/novel_schema.py`:

1. **The arbitrary label→key mapping.** The document says `vokrin:` and
   the target JSON calls it `zelbat`, with no surface similarity. The
   module's own docstring calls this *"the single most important
   property: it is the part that in-context examples convey poorly and
   weight updates should convey well."* **Real tenant schemas are not
   like this.** A freight waybill says `Ship Date:` and the schema key
   is `ship_date`.
2. **Distractor lines** (`n_distractors=4`), whose labels are outside
   the schema and must be dropped.

`grep -rn n_distractors` returns only defaults and one unit test. Neither
constant is swept anywhere. And the evidence we already have is monotone
in difficulty, in the direction that hurts:

| corpus | label→key mapping | measured paired delta |
|---|---|---|
| novel-schema synthetic (gates 1/4/5) | arbitrary pseudowords | **+46.5 / +24.0 / +40.4** |
| freight waybills (rehearsal) | ordinary English | **+4.14**, sign test 5W/5L/17T — FAIL |
| CORD receipts (real, public) | ordinary English | **−7.3 / −11.5 / −4.5** — FAIL |

Read as a series, that table says the effect size may be a function of
how much of the corpus we wrote. **H is the experiment that tells us
which.**

## H.2 Design, fixed now

`mapping="mnemonic"` makes the JSON key the SAME token as the document
label. The unrelated pseudoword is still drawn from the pool, so the two
corpora differ in **nothing else**: identical documents byte-for-byte,
identical values, identical distractors, identical shuffles. Only the key
names change. (`tests/test_addendum_h_ablation.py` pins that the
documents are byte-identical across the two mappings, and that the
default path is unchanged — if the ablation altered the corpus, it would
not be an ablation.)

- **H-A (mapping):** seeds {1, 2, 3}, k=10, n_test=20, fixed geometry,
  `include_demos=True`, greedy matched decode, `n_distractors=4`.
  Arms: arbitrary-mapping control **and** mnemonic-mapping, both run
  fresh on this host, so the comparison is like-for-like rather than
  against a banked arm from different hardware.
- **H-B (distractors):** the same, with `n_distractors=0` at the
  arbitrary mapping.

The control is re-run rather than taken from the banked k=10 replication
because those arms ran cuda/bf16 (B.8) and these run cpu/fp32; pairing
across that difference would credit the ablation with a hardware change.

## H.3 The bar and the readings, frozen before the data

Same rule as every gate above: **paired seed-mean delta ≥ +5.0 micro-F1
AND the sign test agreeing.** Readings, written now:

- **(a)** The mnemonic-mapping delta clears +5.0 with the sign test
  agreeing → the effect is **not** an artifact of the arbitrary mapping.
  This is the strongest cheap control anyone has proposed against this
  result, and surviving it is the single most valuable thing H can
  produce.
- **(b)** Positive but under +5.0 → the effect is **substantially** an
  artifact of the mapping. The headline is then restated everywhere as
  bounded by mapping arbitrariness, with the mnemonic number carried
  beside it at the same size.
- **(c)** At or below zero → **the headline is an artifact of the
  generator**, and this project's pages will say that in those words.
  Gates 1/4/5 stay published with the ablation attached to each.
- **(u)** UNINFORMATIVE guard, and it binds in both directions: if the
  **mnemonic prompted baseline saturates (mean ≥ 0.95)**, the cell says
  nothing about adaptation — a task with no headroom cannot show a
  delta — and it must be reported as UNINFORMATIVE, **not** as reading
  (c). Ceiling effects are the way this ablation could look like a
  refutation while measuring nothing, and (u) exists so that outcome
  cannot be quietly banked as one. The captured-headroom fraction is
  recorded on every cell for the same reason.

H-B is read on the same ladder.

## H.4 What H cannot do

It cannot make the corpus real. Every H arm is still synthetic and still
ours; a mnemonic mapping is closer to a tenant's documents but it is not
a tenant's documents. H narrows the generator objection; only the
blind-holdout offer retires it.

## H.5 Status

**PENDING at the time of this commit.** No H arm exists. This section and
the `VERDICT.md` row are committed before the first arm is launched, and
the git timestamp of this commit is the freeze.

### H.5.1 DECIDED 2026-08-22: reading (a), and the magnitude that qualifies it

All nine cells ran. Three seeds per arm, k=10, n_test=20, cpu/fp32, every
arm re-run on this host.

| arm | prompted baseline | adapted | paired delta | pooled sign test |
|---|---|---|---|---|
| arbitrary mapping (control) | 0.5289 | 0.9396 | **+0.4107** | 56W/0L/4T, p=1.4e-17 |
| mnemonic mapping (the ablation) | 0.7875 | 0.9750 | **+0.1875** | 40W/2L/18T, p=2.1e-10 |
| arbitrary, no distractors (H-B) | 0.6892 | 0.9771 | **+0.2879** | 46W/2L/12T, p=4.2e-12 |

Per-seed mnemonic deltas: +0.1187 / +0.1625 / +0.2812.

**Reading (a) applies.** The mnemonic prompted baseline is 0.7875, below
the 0.95 saturation guard, so (u) does not fire and the cell is
informative. The delta clears the +5.0 bar with the sign test agreeing.
**The effect is not an artifact of the arbitrary label→key mapping.**

**The magnitude is the part that must travel with the verdict.** The
arbitrary mapping is worth **+22.3 F1** of the measured delta — more than
the effect that survives it. Removing the distractors costs a further
**+12.3**. The two constants that set this corpus's difficulty account
between them for most of the headline, and any citation of the control
delta must carry the mnemonic delta beside it.
`tests/test_addendum_h_verdict.py` fails if that number is ever dropped.

**H.4 still binds:** a mnemonic mapping is closer to a tenant's documents
but it is not a tenant's documents. H narrows the generator objection. It
does not retire it, and the blind-holdout offer is still the only thing
that would.

## H.6 The contingency, frozen now because (u) is the likely outcome

Written **2026-08-22 while the first mnemonic cell was still running**, and
committed before it landed, because a follow-up designed after seeing a
null is a rescue and a follow-up designed before it is an experiment.

An outside reader predicted the modal H outcome is **(u)**: with the key
equal to the document label, a 0.5B model may solve the task from ten
in-context examples, the mnemonic prompted baseline saturates at ≥0.95,
and the cell says nothing about adaptation because there is no headroom
to show a delta in. That prediction is plausible and it is why (u) exists.

**Two things are frozen here.**

**First, what saturation would itself mean, stated before we know.** If
the mnemonic baseline saturates, that is not a neutral outcome we route
around — **it is a finding, and it is against us.** It would say that on
a corpus whose labels resemble a real tenant's, plain prompting of a
0.5B model already reaches ceiling from ten examples, and there is
nothing for per-tenant adaptation to add. That belongs on the evidence
pages in those words, at the same size as the passes, **whether or not
the follow-up below recovers a measurable delta.** A ceiling that
appears the moment the corpus is made realistic is a product fact, not
just a measurement problem.

**Second, the follow-up, with its bar unchanged.** If and only if (u)
fires, run the mnemonic mapping at **k=4 and k=2** — the same corpus,
the same documents, the same scorer, the same +5.0 bar and
two-statistics rule, the demonstration count reduced so the baseline has
headroom again. Readings (a)/(b)/(c)/(u) apply unchanged at each k. No
other amendment is authorised: not a different geometry, not a different
model, not a different metric, and not a different bar. If the follow-up
also returns (u) at every k, H is recorded **UNRUN with the reason
measured**, in the same form Addendum G was, and it may not be cited in
either direction.

This is the second time this project has preregistered a contingency
rather than amending after a null. The first was Addendum G, and the
discipline held there — it was recorded UNRUN rather than quietly
reshaped. It should hold here.
