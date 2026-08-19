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
  tokens (~300). A k=10 few-shot prompt ≈ 10×800 + 500 instructions + 500 test
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
generator (`src/arcttt/novel_schema.py`, 10 tests) exists; no corpus, no
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
156W/0L), before any rung above 0.5B has produced a single k=30
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

**B.9.4 Cluster-honest inference.** The receipt-level CI [42.8, 49.4]
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
Read 2 (unified claim): seed-mean −51.9 F1, 0W/155L — FAIL.
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
of 8,192 tokens — 124 of headroom. E-r1 seeds 201/202 measured 8,539-
8,700 and produced the same all-excluded failure as E.4.

Fix, with the screen stated precisely so it cannot be mistaken for
cherry-picking: candidate seeds ascending from 201 are screened by
TOKEN COUNT ONLY (decode prompt and max LOO training sequence both
≤ 7,900 with the production tokenizer) — a measurability screen, blind
to content and to any model output; no arm had run on any accepted
seed at screening time. First six fitting: **{203, 204, 206, 207,
208, 209}** (fields 6-7, groups 2-3 — shapes still vary). All other
E-r1 values unchanged, including the +5 bar and both-statistics rule.
Seeds 201/202's excluded artifacts stay banked as the record.
