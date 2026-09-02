# Evidence, in one page

Per-tenant adaptation of a small open model (Qwen2.5-0.5B) for
document → tenant-schema JSON extraction, measured under gates that
were **frozen before the data existed** — and, from gate 3 onward,
publicly timestamped to the Bitcoin blockchain. Gate 1, the +46.5
headline, is the exception: its 2026-08-12 freeze predates the public
repository and rests on git history we control. Stated here rather than
only in the ledger at the bottom, because it is the headline.

Every number below is reconciled to a named artifact in
[`VERDICT.md`](VERDICT.md), and the failures are on this page at the
same size as the passes. If a number here disagrees with its artifact,
that is a bug — open an issue.

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

---

## The ladder, in the order it happened

| # | Question the gate asked | Verdict | The number, with what makes it true |
|---|---|---|---|
| 1 | Does adaptation beat the **same model's own 30-shot prompt** on schemas it has never seen? | **GO** | **+46.5 F1** seed-mean over 3 tenant schemas (+36.0/+49.0/+54.4) vs a +5.0 bar frozen beforehand. Sign test 156W/0L/2T — over **158 scored pairs of 180 designed**; 22 were excluded because the 30-shot prompt exceeded the frozen 8192-token budget, disclosed symmetrically. |
| 2 | Does it also help on a **real public dataset** the model already knows (CORD receipts)? | **FAIL** | **−7.3 / −11.5 / −4.5 F1** at 0.5B / 1.5B / 4B. Failed its preregistered gates at all three scales. This is the scoping result: adaptation helps where the schema is novel to the model and not where it already knows the domain. (“Novel-schema **conformance**” is how this row read until 2026-08-22, and it is measurably wrong — see the mechanism row below: on gate 5 the prompted baseline's schema conformance is already **perfect**, and every point adaptation adds is a corrected **value**.) |
| 3 | Can a demo-trained adapter serve a **bare document** (no examples in the prompt)? | **FAIL** | **0.0000 F1, 0/60 valid JSON**, all three seeds — the adapter had encoded the schema as context-conditioned behavior, so a bare document elicited prose. Adapter contribution over no-adapter: **+0.0**. Published under the pre-written failure branch. |
| 4 | Does training **on the serving configuration** fix that? | **PASS** | **+24.0 F1** seed-mean over the prompted baseline (+32.0/+5.5/+34.7; bar +5), 126W/19L/13T over 158. **Both CI levels, per this project's own binding rule:** receipt-level [+22.3, +30.9] — and **cluster-level over the three tenants [−16.0, +64.1], which INCLUDES ZERO.** The seed-mean is the decision statistic, so the cluster interval is the honest one and it is wide because n=3. Gates 1 and 5 quote both levels; this row quoted only the one that clears until 2026-08-22, when an outside auditor found it. **And one of the three tenants does not pass the frozen rule on its own: seed 2 is +5.5 on an n=38 intersection with a sign test of 18W/15L/5T (p=0.36) — the mean clears, the sign test does not, which is the exact shape that makes the waybill row a FAIL.** 11 of the 12 tenants across gates 1/4/5 pass individually; that one does not, and both figures are in `experiments/per_tenant_verdicts_2026-08-22.json`. Honest cost: **−22.4 F1** vs the demo-context arm, and the absolute document-only scores are **0.9407 / 0.5282 / 0.7798** — one of three tenants sits at 0.53, and seed 2 cleared its bar by only +5.5. ~272 s to adapt one tenant. |
| 5 | Is the effect just an artifact of **one fixed corpus shape**? | **PASS** | **+40.4 F1** seed-mean over **six fresh, shape-varying tenants** (+26.4/+46.4/+35.3/+37.6/+46.2/+50.2), cluster CI [+31.01, +49.70], receipt CI [+38.1, +42.6], sign test **340W/5L/15T**, **zero documents excluded** of 360. Seeds were screened for token budget *before* any arm ran, so the exclusions that cost gate 1 twenty-two receipts do not arise — and the price of that screen, stated: it tested **nine** candidate seeds and **rejected three** (8,438–9,003 tokens against a 7,900 bound), so **one third of fresh tenants at this geometry are not measurable at k=30 at all.** That is a limit of the product, not only of the experiment. All 12 arms re-scored from raw predictions. |

**What gates 1–5 do not show:** every corpus above is **synthetic** — deterministically generated tenant schemas, not a customer's documents. The one real dataset tried is row 2, and it failed. That gap is the point of the blind-holdout offer below.

---

## The comparison that cuts against the result

On this corpus, **a frontier model and the cheapest API tier both score 1.00 by plain prompting** (60/60 at micro-F1 1.00, same scorer). Precision the artifact itself insists on: those API arms are **k=10 on the first 20 eval documents of each of 3 tenants**, a bounded subset their own record says must never be pooled with the 60-doc kernel arms — so this "60" is 3x20, not one tenant's 60, and the arms are not run at the gates' k=30. The corpus is saturated at the top of the market; the measured wins above are *at 0.5B*, against that model's own prompt. Quality differentiation on real workloads is unproven, and claiming otherwise from these artifacts would be wrong.

**The comparison a buyer would actually make has now been run, and we lose it.** Every gate above is our 0.5B against its own prompt; no buyer has ever faced that choice. On 2026-08-22 the cheapest hosted tier was put on the 30 held-out freight waybills — our only realistic corpus, gold published, the same 20 demonstration pairs our own k-shot arm received, the same pinned scorer, temperature 0.

| on the same 30 waybills | mean micro-F1 |
|---|---|
| **gemini-3.5-flash-lite (cheapest hosted tier)** | **0.9708–1.0000 over four runs, mean 0.9865** — 0 invalid JSON in every run; 27–28 of 30 exact-match, the rest differing from gold in letter case only, which the scorer folds |
| our adapted 0.5B | 0.8833 |
| our prompted 0.5B | 0.7836 |

Paired, ours minus hosted, across the four runs: **−0.0875 to −0.1167, sign tests from 1W/13L/16T to 0W/14L/16T — our adapted arm wins at most one document of thirty, in any run.**

**A single hosted-API run is not a measurement, and the first write-up of this row treated one as a fact.** Re-running the identical arm — same model, same prompt, same documents, `temperature: 0` — does not return the identical number: hosted inference is not deterministic at temperature 0. Four banked runs give **1.0000, 1.0000, 0.9750, 0.9708 — mean 0.9865, range [0.9708, 1.0000]** — and exactly one document (`h-3307`) accounts for all of the movement. **The number is not reproducible to four decimals; the conclusion is:** the hosted model beats our adapted 0.5B in every run, on every run's sign test (worst case 1W/13L/16T, p=9.2e-04). Quote the range, never one run's mean. `scripts/market_baseline_summary.py` aggregates them, and this correction is ours — the reviewer found the defects that made us re-run, and the re-run found this. The hosted model also scored 1.000 on all three documents where our prompted arm emitted unparseable output, and on the hard-tier documents where our adapted arm scored **0.696** on the greedy decode this comparison uses (0.679 is the *voted* rehearsal arm, and mixing the two decodes is the thing `bank_rehearsal_baseline.py` exists to forbid).

This was **preregistered with its readings frozen before the arm ran**, and it landed on the one that costs the most. Stated in those frozen words: **the five-gate ladder is answering a question no buyer is asking. On documents that can be sent to a hosted API, we lose on accuracy, and nothing else on this page changes that.**

**On cost, this arm proves nothing and the claim is withdrawn to what is measured.** Every cost figure in this repository — ~$1.09 and ~$5.13 self-hosted, ~$0.22–1.03 API — was measured on the *synthetic* novel-schema corpus, at a different demonstration count and a different token profile. Transferring it to the waybill corpus would be exactly the cross-corpus conflation this page has already had to correct twice. **No cost was measured here when this was written**: the runner did not record API usage tokens and our own throughput was never measured on these documents. The hosted half has since been measured — see below; our half has not. The accuracy result stands on its own and does not need the cost claim.

**And on cost, half of it is now measured — the half that is the payload
asymmetry.** The hosted model must carry all 20 demonstrations in
**every** request; that is what makes it work, and it is not optional.
From its own usage metadata across two runs: **4,332,000 input tokens per
1,000 documents**, against a mean document of 289 characters — a
demonstration-carrying request is **~41x the payload of a document-only
one** here. Our adapted model carries none of that at serving time,
because the schema is in the weights. At the tier's quoted list price
that is **~$1.55 per 1,000 documents cold — HIGHER than the ~$1.03 this
page had been quoting**, which came from the synthetic corpus at a
smaller demonstration count and is exactly why it should never have been
carried across.

**Our own side is now measured too, with the bias in it named.** The
timing run had to inject an *untrained* LoRA — the banked rehearsal
adapter was not retained — and an untrained adapter does not know when to
stop: it generated **197 tokens per document** against the **93** the
banked adapted arm actually emitted. Unadjusted wall-clock would have put
us at ~$1.88/1k, overstating our own cost. Corrected the only honest way
available — a measured **8.42 tokens/second** on this hardware, which is
weight-independent, applied to the banked arm's real output lengths — our
document-only serving is **~$0.89 per 1,000 documents**. Both figures are
banked so nobody has to take the correction on trust.

**And then the argument they were assembled for died, in the way we
preregistered it might.** The $1.55 above is not the price of the
workload; it is the price of a *choice* — carrying all 20 demonstrations
in every request — and that choice is what the payload-asymmetry argument
assumed was forced. **Addendum J's sweep shows it is not.** Priced from
the same artifacts, at the same list rates, on the same 30 documents:

| hosted arm | prompt tokens / 1k docs | cost / 1k docs | quality |
|---|---|---|---|
| k=20 (what we priced) | 4,332,000 | ~$1.55 | 0.9708–1.0000 |
| k=5 | 1,083,000 | ~$0.58 | 0.9459 |
| **k=2** | **498,000** | **~$0.40** | **0.9722** |
| **k=0 + declared schema** | **172,000** | **~$0.36** | **0.8930** [0.8583–0.9208] |
| k=0, nothing declared (a strawman: no deployment sends it) | 106,000 | ~$0.31 | 0.0000 |
| our adapted 0.5B, batch 1 (what we published) | — | ~$0.51–0.89 | 0.8833 |
| **our adapted 0.5B, batch 16** | — | **~$0.14** | **0.8833** |

**Read the hosted rows first, then ours.** With no demonstrations at all
— just the tenant's field list, perhaps fifty tokens, a **25x** reduction
in payload — the hosted tier scores **0.8930, above our adapted 0.8833**.
That part is unchanged and it is the ceiling on everything here.

**What changed on 2026-08-23 is our side, and it changed by 3.7x.** Our
published cost was measured at batch size 1 — one document at a time —
and nothing forced that. Serving the same adapter at batch 16, greedy,
document-only, the same 30 documents cost **$0.1406 per 1,000 at an
unchanged 0.8833**. Not "within tolerance": every batched prediction was
compared to the batch-1 reference under canonical JSON and **zero
differed**. Batching changes throughput, never content.

So the honest sentence is no longer "we are dearer and worse" — that was
true of the configuration we were publishing and is false of the one we
can serve. It is: **2.6x cheaper and 0.0097 worse.** That is a real
trade where this morning there was none, and it is still a trade, not a
win. Three things it does not buy: it does not close the quality gap;
the one-time adaptation (~41.6 s/tenant, ~$0.003 per 1,000 documents
amortised over a thousand) is excluded and belongs in any honest total;
and **both sides are COLD** — a warm shared-prefix cache would cut the
hosted figure and would not change ours, so 2.6x is an upper bound on
our advantage, not a floor. The measurement that produced it also killed
int8 quantization outright (0.8833 → 0.0300, 22 of 30 outputs
unparseable) and caught a throughput benchmark of our own that had
hidden that collapse behind a forced token count. **The claim that survives is the one
Addendum I left and nothing more — workloads where a hosted API is not an
option, where this comparison is not available at any price.** The
warm-cache caveat now only cuts against us further: it would make the
hosted tier cheaper still. This hole was preregistered as Addendum J,
before its arms existed, by us, precisely because it was the obvious way
this argument could die.
`experiments/waybill_cost_2026-08-22.json`,
`experiments/waybill_cost_ours_2026-08-22.json`.

 The ladder measures a real delta over a weak baseline; that is not the same as anyone having a reason to buy.

**What survives is narrower, and it is now the whole claim: workloads where a hosted API is not an option.** There the question is not "who is more accurate" but "is an on-prem small model good enough for *my* schema" — and gates 1, 4 and 5 are the evidence that adaptation is what makes a 0.5B viable there at all. Everything on this page should be read as being about that segment and no other.

**2026-08-25 — THAT SEGMENT NO LONGER SUPPORTS THE CLAIM EITHER (Addendum O, preregistered, reading (a)).** The paragraph above narrowed this project to *workloads where a hosted API is not an option* and said adaptation is what makes a 0.5B viable there. **It is not.** `Qwen2.5-3B-Instruct`, an open checkpoint any on-prem buyer can run, with **no adaptation of any kind** and only the tenant's field list in a 196-token prompt, scores **0.8958 with 0 of 30 invalid** against our adapted 0.5B's 0.8833 — 8W/8L/14T, p=0.60, a dead heat. Reading (a) publishes in the words frozen before the arm ran: *a 3B needs neither our adaptation nor a large prompt, and the on-prem cost claim is dead, not narrowed.*

**And the result that had been holding this up was a markdown fence.** Addendum N published "given only a field list, a 1.5B does not produce a usable object at all — 0.0000, 30 of 30 invalid" and was cited as the asymmetry that made adaptation earn its keep on-prem. A reproduction run reproduced that number exactly, stored the predictions this time, and they are correct extractions inside fenced json code blocks. Fence-stripped: **0.7375, zero invalid.** N is withdrawn. This page had already met fenced JSON on the hosted arm one paragraph below, handled it correctly, and written down why — and the scale-rung runner written three days later did not inherit that rule. The un-repaired zero was the number that flattered us.

**What is left, and it is all that is left:** matching our quality with an unadapted open model takes a 3B, about 6x the parameters, which on the same CPU box costs about **$2.97 per 1,000 documents against our $0.5143 at the same batch size**. Cost at fixed quality — real, measured, much weaker than what this page claimed yesterday, and **not yet measured at matched batching**, which is the first thing a technical reader should ask and which we have not run. See `VERDICT.md` Addenda N and O and `experiments/fence_rescore.json`.


**The one protocol difference a critic could name was removed rather than caveated.** Our k-shot arm gets its demonstrations as alternating chat turns; the first run packed them into one turn. Repeated with the turn structure mirroring our own arm's exactly, the result is **identical, document by document: 1.0000, 0W/14L/16T.** Both runs are banked.

**Which of the two runs is the citable one, and why it matters.** The first run stripped a markdown ```` ``` ```` fence from four of the hosted model's outputs before scoring. **Our own arms never received that repair** — `run_challenge.py` does a bare `json.loads` and records a null on failure — so granting it to the hosted arm alone is an asymmetry in the hosted model's favour. Measured: without the fence strip that run scores **0.8667 with 4 invalid JSON, below our adapted arm's 0.8833.** The matched-turn run emitted **zero fences**, so its 1.0000 needs no repair at all and is the number this result rests on. **The matched-turn artifact is the citation; the packed run is context.** Both artifacts now bank the fenced document list and the un-repaired mean beside the headline. An outside reviewer found this, and it is the closest anything has come to overturning the result.

Scope, stated rather than used as an escape: 30 documents, one corpus, agent-authored with OCR-like damage injected by its author rather than real scans, one model, one run at temperature 0. Contamination does not explain it either: the gold first entered any repository at 2026-08-21T23:05Z, about seven hours before this arm ran, the public cut carrying it is still unpublished, and the values are generated — no training cycle closes in that window. **A different leakage path, unrebutted, and it inflates the hosted arm:** these documents *and their gold* were authored by an LLM agent, to normalization conventions (casefold, numeric canonicalization) that the challenge kit disclosed to it before it labelled. A hosted frontier model reproducing that gold is therefore partly a measure of agreement between LLMs, and a 0.5B cannot exploit shared conventions the same way. That makes 1.0000 an over-estimate of what a hosted model would score on real scans — which cuts against the strength of our own conclusion, not in favour of it. It is stated because the earlier paragraph closed the memorisation door and left this one open, and an outside reviewer walked through it. It is nonetheless the **third corpus in a row saturated at 1.00 by a hosted model** — though the honest statement of that pattern is that **all three of our corpora are machine-generated**, which is a fact about our corpora before it is a fact about the market — frontier on the synthetic corpus, cheap tier on the synthetic corpus, cheap tier here. The pattern is the finding, not the caveat. The runner is `scripts/run_market_baseline_waybills.py` and the corpus and gold are published, so anyone can re-run it.

**And in the configuration that could actually be sold, we lose to the cheap API on quality outright.** Every delta on this page is a within-model comparison: the same 0.5B against its own prompt. The absolute number a buyer would receive is the document-only served score — **0.9407 / 0.5282 / 0.7798, mean 0.750** — against the cheap API tier's **1.00** by plain prompting on the same corpus. There is no arm anywhere in this repository benchmarking adaptation against what a buyer would otherwise deploy. The claim is about a delta over a weak baseline, not about being the best available extractor, and those are not the same claim.

**And the prompted baseline is much stronger on realistic documents than the headline corpus implies.** On the synthetic novel-schema corpus the same model's 30-shot prompt scores 0.4333–0.6208, which is what makes +46.5 possible. On the freight-waybill corpus a 20-shot prompt scores **0.7836**. The gap the product has to earn is far narrower where the documents look real — measured, not estimated, in the paired run below.

**Cost, also against interest** (rates dated 2026-08-19, arithmetic in `VERDICT.md`):

| Serving path | Cost per 1,000 documents | Measured quality, SAME adapter and corpus |
|---|---|---|
| Self-hosted CPU, greedy decode | ~$1.09 | **0.7686** |
| Self-hosted CPU, voted decode (the decode that passed gate 4) | ~$5.13 | **0.7798** |
| Cheap API tier, cold single requests | ~$1.03 | **1.00** (60/60 at micro-F1 1.00; the field this came from was named `exact` and was not exact match — corrected 2026-08-22) |
| Cheap API tier, warm shared-prefix caches + batch | ~$0.22 | **1.00** (same arm) |

Both self-hosted rows are the *same* seed-3 document-only adapter on the
*same* 60 documents (`novel_serving_throughput_cpu_2026-08-19.json`,
`novel_greedy_quality_2026-08-19.json`), so they are directly comparable
and the comparison is unkind twice over. **Voted decode costs 4.7x more
than greedy and buys +0.011 F1** — the decode that passed gate 4 is not
buying quality on this corpus, it is buying variance reduction. And **at
essentially the same price — ~$1.09 against ~$1.03 — self-hosted scores
0.77 while the cheap API scores 1.00.**

Two precisions, both against us. This table quoted **0.954–0.985** for
the voted row until 2026-08-22; that range is the *demo-context* k=10
arm, a different serving mode that ships the examples in every request
and has none of the payload advantage the economics depend on. Quoting
it here was the mode conflation `CORRECTIONS.md` says was withdrawn on
2026-08-19, recommitted by the person who wrote that entry, and found by
an outside auditor. And the API rows are k=10 on **20 eval documents per
tenant** (3 x 20 = 60 receipts), a bounded subset their own artifact says
must never be pooled with the 60-doc kernel arms — the "60" here is not
the same 60 as everywhere else on this page.

Today the cheap API wins on cost *and* on quality **on the synthetic novel-schema corpus these figures were measured on — they do not transfer to the waybills**, cold caches
included; the corpus is saturated at the top of the market, so "1.00" is
a statement about the corpus as much as about the API. The GPU crossover
is the next cost measurement and publishes either way.

---

## What the corpus's difficulty was worth (Addendum H, preregistered, decided 2026-08-22)

The strongest objection anyone raised against the headline was that it
might be an artifact of the corpus's **difficulty**, not of adaptation.
Two constants set that difficulty and neither had ever been swept: the
**arbitrary label→key mapping** (the document says `vokrin:`, the JSON
calls it `zelbat`, no surface similarity) and **four distractor lines**.
Real tenant schemas are the opposite — a waybill says `Ship Date:` and
the key is `ship_date`.

The ablation makes the JSON key the document's own label and changes
nothing else: the documents are byte-identical, same values, same
distractors, same shuffles, pinned by a test. Three seeds per arm, k=10,
every arm re-run on this host.

| corpus | prompted baseline | adapted | paired delta | sign test |
|---|---|---|---|---|
| arbitrary mapping (as the gates run it) | 0.5289 | 0.9396 | **+41.07** | 56W/0L/4T |
| **mnemonic mapping** (realistic labels) | 0.7875 | 0.9750 | **+18.75** | 40W/2L/18T, p=2.1e-10 |
| arbitrary mapping, no distractors | 0.6892 | 0.9771 | **+28.79** | 46W/2L/12T |

**The preregistered reading is (a): the effect is not an artifact of the
mapping.** The mnemonic baseline does not saturate (0.7875), so the cell
is informative, and +18.75 clears the +5 bar with the sign test agreeing.
This is the strongest cheap control that was proposed against this
result and it survives.

**The more useful half is the magnitude. The arbitrary mapping is worth
+22.3 F1 — more than half the measured delta.** Removing the distractors
costs a further +12.3. The two design choices that make this corpus hard
account between them for most of the headline.

**And the series is coherent rather than convenient:**

| corpus | how realistic | delta |
|---|---|---|
| arbitrary labels + decoys | least | **+41.1** |
| realistic labels + decoys | more | **+18.8** |
| freight waybills (realistic labels and decoys) | most | **+9.97**, and **+4.14** granting nothing |
| CORD receipts (real, public) | real | **negative** |

The more a corpus looks like a customer's, the smaller the effect. That
was already visible across the corpora; H is the first arm to *measure*
one axis of it rather than infer it. **The number a buyer should expect
is the bottom of that table, not the top.**

---

## What adaptation actually buys, measured rather than asserted

The waybill result above says the benefit there is output format, and that
a free constrained decoder is the better explanation for it. The obvious
next question is whether that objection also eats gates 1, 4 and 5 — the
headline. It does not, and this is measured from the banked arms, not
argued:

| gate | invalid JSON, either arm | delta as measured | delta restricted to documents both arms parsed |
|---|---|---|---|
| 1 (novel schema, k=30) | **0 of 158** | +46.5 | **+46.5** |
| 4 (document-mode serving) | 1 of 158, on **our** arm | +24.0 | **+24.5** |
| 5 (six fresh geometries) | **0 of 360** | +40.4 | **+40.4** |

Zero format failures means the format restriction is a no-op: there is nothing on these gates for a constrained decoder to fix — and the denominator is post-exclusion: gate 1's 22 designed receipts that exceeded the frozen token budget produced no completion at all in either arm, which is the most complete output failure there is. The restriction is silent on those 22 and speaks only for the 158 scored.

Valid JSON is not the same as the *right keys*, so gate 5 — the only gate
whose baseline arm stores its predictions, which is why this half of the
finding is stated for gate 5 and not generalised to gates 1 and 4 — was
pushed one step further.
The 30-shot baseline was rebuilt under the gold schema's key paths, keeping
its own values, which is exactly what a schema-constrained decoder
guarantees; and then again while forgiving nesting mistakes, which is more
than any real decoder gives you. **Neither repair changes its score at
all** — because across all 360 documents the baseline's key-path precision
and recall are already **1.00** and its key set is exactly gold's on
**60/60 documents per tenant**. It emits the tenant's schema perfectly and
puts the wrong content in **43% of the fields** — a six-tenant mean over a **27%–51%** spread, and arithmetically 1 − the baseline's 0.5726 rather than an independent measurement.

So the mechanism is the opposite of the waybill one, and the contrast is
the actual scoping result of this project: **where the prompted baseline
already knows the shape, adaptation is buying value-level extraction
accuracy that no decoder can supply; where the baseline is close on
content (waybills, 0.78), what is left is mostly format, and a decoder
plausibly takes it.** Both are post-hoc analyses of banked data, labeled
as such in their artifacts, and both ship with the scripts that produce
them: `scripts/format_counterfactual.py` and
`scripts/schema_conformance_decomposition.py` (stdlib and the generator;
gold is regenerated, never read from a stored copy).

---

## The protocol, and the one time it has run

The standing offer: **you** split your own documents on your own machine, keep the gold labels, and score a single submission with a scorer pinned to a commit named in advance. Terms were frozen and timestamped before any challenger existed (`docs/research/BLIND_HOLDOUT_PROTOCOL.md`, OTS-anchored snapshot).

It has been executed once, as a **labeled dress rehearsal**: an adversarial AI agent **I ran myself** authored 50 freight waybills outside this generator's distribution, withheld its gold from me, and scored one blind submission — **0.8792 mean micro-F1, 30/30 valid JSON**. By its author's difficulty tiers: easy 0.950, medium 0.931, mixed 0.944, **hard 0.679**. Its named failure modes: **no unit arithmetic** ("11.5 short tons" → 1150, not 23000), **role-order inversion** (shipper/consignee swapped when the layout inverts), and **fluent confabulation under heavy OCR** ("Ta11ahassee" → "Tampa") — the dangerous one in production, published here because it is true. The rehearsal's recorded verdict: *"it survived contact, it did not win it."*

**The corpus was agent-authored, the challenger was an AI agent, and I operated it — same session, same host. Gold was withheld in process and hash-committed before my submission, but that is procedural blindness, not third-party custody. This is a protocol-integrity and transfer signal, not a real tenant's documents.** The rehearsal also caught two gaps in the deliverables paperwork (a commit reference the challenger could not fetch; a base model pinned by mutable name), one now hard-gated in code (the runner refuses to emit an unpinned submission) and one documented in the runner.

---

## Check it yourself

```bash
git clone https://github.com/raj200501/arc-ttt-public && cd arc-ttt-public
python3 scripts/verify_verdict.py          # recomputes gate 1 from raw per-receipt records
python3 scripts/addendum_e_summary.py      # recomputes gate 5 under its frozen rule
python3 scripts/read_addendum_d.py         # recomputes the failure in row 3
```

**Then run the three that argue against us**, which take about a second
between them and need nothing installed:

```bash
python3 scripts/format_counterfactual.py               # the rival explanation, on the realistic corpus. It wins there.
PYTHONPATH=src python3 scripts/schema_conformance_decomposition.py   # the same objection on the headline gates. It closes 0%.
python3 scripts/per_tenant_verdicts.py                 # the frozen rule per tenant: 11 of 12 pass, and one does not.
```

**Or run it on your own documents.** One JSONL of your labeled documents, one command, and you get the same paired comparison every gate row above is built on — prompted baseline vs adapted, matched decode, scored against your gold with the pinned scorer, per-document:

```bash
python3 scripts/try_your_documents.py --docs mydocs.jsonl   # needs torch; adapts weights
```

It is **not** blind, it fixes **no** bar in advance, and it is one corpus — it prints all three of those next to its own result, and if your prompted baseline is already saturated it says plainly that there is nothing here for you to buy. The version that is *evidence* is the blind-holdout offer below.

All three verdict scripts are dependency-free. To go past arithmetic to primary evidence,
`python3 scripts/verify_from_primary.py experiments/novel_schema_f_*.json` re-scores every
stored prediction against gold regenerated from the deterministic corpus generator —
it checks the *predictions*, not the summaries. 381 offline tests, no downloads:
`python3 -m pytest tests/ -q`.

---

## Honest ledger

Every claim withdrawn or revised, with its cause and date, is collected
in [`CORRECTIONS.md`](CORRECTIONS.md) — self-correction only counts as
evidence if it is countable.

- **Zero customers.** No real-workload win exists. The blind-holdout offer is open precisely because that is the missing evidence.
- **The rehearsal's paired baseline FAILED its bar, and it is the most useful thing measured this week.** 0.8792 was an adapted score with nothing to subtract from it, so the missing arm was run against a rule frozen first. Matched greedy decode, same 30 waybills: prompted **0.7836**, adapted **0.8833**, delta **+9.97** — twice the +5 bar, CI excluding zero — but the sign test disagreed (**8W/5L/17T, p=0.29**), so the two-statistics rule returns **FAIL** and the row says FAIL. The rule caught what it is for: 17 of 30 documents tie, and **63% of the delta comes from 3 documents where the prompted arm emitted invalid JSON**. On this realistic corpus the measured benefit is concentrated in **output-format reliability, not extraction accuracy** — narrower than the headline gates support, and it is the first paired number this project has on document-shaped data.
- **Then we named the rival explanation for that and measured it against ourselves — and it wins on this corpus.** If the benefit is output format, a **JSON-grammar-constrained decoder buys it for free**: no adaptation, no per-tenant weights, no training. Nothing in this repository had ever named that. `scripts/format_counterfactual.py` bounds it from the published raw arms, stdlib-only. Grant the prompted arm the free half of constrained decoding — delete its 12 off-schema keys, which a schema-constrained decoder cannot emit — and the delta falls **+9.97 → +9.26**. Restrict to the **27 documents both arms parsed**, where format cannot be the explanation because neither arm failed format: **+4.14 granting nothing at all, sign test 5W/5L/17T** (+3.34 if the baseline is also key-pruned) — below our own +5 bar and directionally a coin flip. Grant an impossible fixer that makes the prompted arm *perfect* on all three unparseable documents: **−0.74**. **The rival explanation is not refuted — and neither is ours. At n=27 the interval is [−3.5, +11.8] and the sign test is 5W/5L (p=0.62): nothing is supported here in either direction.** Saying "the rival wins" was itself an over-claim, made against our own interest, and an outside reader caught it; an over-claim is an over-claim whichever way it points. What is true is narrower and worse for us than a clean loss would be: the one realistic corpus we have cannot distinguish adaptation from a free decoder, and the reading that models a decoder that actually exists (C) sits at **+0.47 with the sign test pointing the wrong way**. It does not touch gates 1/4/5, where the prompted baseline is 0.4333–0.6208 and format is not what separates the arms. It does mean the honest v1 ships constrained decoding and **the adaptation layer has to earn its keep on top of it** — now a PENDING row in [`VERDICT.md`](VERDICT.md) with its bar and all three readings frozen before the arms exist, including the one that says there is no product.
- **A published interval was wrong in our favour, and an outside reader found it.** Gate 5's cluster CI read [+32.75, +47.95] because the quantile table in our own reader invented a value for six seeds instead of computing t at df=5. Corrected here to [+31.01, +49.70] — ~19% wider. The verdict is untouched (the lower bound clears the +5 bar six-fold), the cause is in [`CORRECTIONS.md`](CORRECTIONS.md), and the estimator is now computed rather than looked up.
- **One tenant at 0.53** in the serving mode the economics depend on, and its mechanism is **OPEN** — an earlier explanation (document length) was tested against the banked data, was not supported, and was withdrawn on the record rather than quietly revised.
- **The headline is the least verifiable number on this page, and that is worth saying plainly.** Gate 1's arms predate primary-evidence storage: they record a per-receipt `micro_f1` and **no predictions**, so `verify_from_primary.py` cannot re-score +46.5 against regenerated gold — it reports the artifact as unverifiable and does not silently pass it. What can be checked is the arithmetic (`verify_verdict.py` re-aggregates the stored per-receipt scores) and the corpus (the generator is deterministic and in the repo); what cannot is the step from model output to score. Gates 4 and 5 store raw predictions and are primary-verifiable end to end. Combined with the row-1 freeze resting on private git history, **the number this project is named after is the one with the weakest verification path**, and re-running those arms with predictions stored is the fix — it is ~9 CPU-hours for the prompted arms alone and has not been done.
- **Everything above is 0.5B, and that is the objection most likely to be fatal.** The delta exists partly *because* 0.5B is weak at in-context learning — a stronger base model is better at the very baseline this result is measured against, so the margin should shrink by construction. How much is the open question. The 1.5B/2B rung is now a **PENDING row in [`VERDICT.md`](VERDICT.md) with its bar and all three preregistered readings frozen before the arms exist**, including the one that says the headline is a small-model artifact. Failed rungs stay on the figure.
- **The ARC Prize track this harness grew out of scored 1.67% public** across three submissions and is formally deprioritized. It is disclosed, not featured.
- **Preregistration ordering:** gates from row 3 onward are SHA-256 chain-anchored (OpenTimestamps, Bitcoin) with byte-exact snapshots shipped. One precision rather than a footnote: gate 5's +5 bar and decision rule are in the chain-anchored snapshot, while its measurability amendment (the compacted geometry and the token screen) was git-committed before its data and anchored the following day. The row-1 freeze (2026-08-12) predates the public repository and rests on private git history — stated plainly rather than implied.
