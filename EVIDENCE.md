# Evidence, in one page

Per-tenant adaptation of a small open model (Qwen2.5-0.5B) for
document → tenant-schema JSON extraction, measured under gates that
were **frozen and publicly timestamped before the data existed**.

Every number below is reconciled to a named artifact in
[`VERDICT.md`](VERDICT.md), and the failures are on this page at the
same size as the passes. If a number here disagrees with its artifact,
that is a bug — open an issue.

---

## The ladder, in the order it happened

| # | Question the gate asked | Verdict | The number, with what makes it true |
|---|---|---|---|
| 1 | Does adaptation beat the **same model's own 30-shot prompt** on schemas it has never seen? | **GO** | **+46.5 F1** seed-mean over 3 tenant schemas (+36.0/+49.0/+54.4) vs a +5.0 bar frozen beforehand. Sign test 156W/0L/2T — over **158 scored pairs of 180 designed**; 22 were excluded because the 30-shot prompt exceeded the frozen 8192-token budget, disclosed symmetrically. |
| 2 | Does it also help on a **real public dataset** the model already knows (CORD receipts)? | **FAIL** | **−7.3 / −11.5 / −4.5 F1** at 0.5B / 1.5B / 4B. Failed its preregistered gates at all three scales. This is the scoping result: adaptation buys *novel-schema conformance*, not general extraction quality. |
| 3 | Can a demo-trained adapter serve a **bare document** (no examples in the prompt)? | **FAIL** | **0.0000 F1, 0/60 valid JSON**, all three seeds — the adapter had encoded the schema as context-conditioned behavior, so a bare document elicited prose. Adapter contribution over no-adapter: **+0.0**. Published under the pre-written failure branch. |
| 4 | Does training **on the serving configuration** fix that? | **PASS** | **+24.0 F1** seed-mean over the prompted baseline (+32.0/+5.5/+34.7; bar +5), CI low +22.3, 126W/19L. Honest cost: **−22.4 F1** vs the demo-context arm, and the absolute document-only scores are **0.9407 / 0.5282 / 0.7798** — one of three tenants sits at 0.53, and seed 2 cleared its bar by only +5.5. ~272 s to adapt one tenant. |
| 5 | Is the effect just an artifact of **one fixed corpus shape**? | **PASS** | **+40.4 F1** seed-mean over **six fresh, shape-varying tenants** (+26.4/+46.4/+35.3/+37.6/+46.2/+50.2), cluster CI [+32.75, +47.95], receipt CI [+38.1, +42.6], sign test **340W/5L/15T**, **zero documents excluded** of 360. Seeds were screened for token budget *before* any arm ran, so the exclusions that cost gate 1 twenty-two receipts do not arise. All 12 arms re-scored from raw predictions. |

**What gates 1–5 do not show:** every corpus above is **synthetic** — deterministically generated tenant schemas, not a customer's documents. The one real dataset tried is row 2, and it failed. That gap is the point of the blind-holdout offer below.

---

## The comparison that cuts against the result

On this corpus, **a frontier model and the cheapest API tier both score 1.00 by plain prompting** (60/60 exact, same protocol, same scorer). The corpus is saturated at the top of the market; the measured wins above are *at 0.5B*, against that model's own prompt. Quality differentiation on real workloads is unproven, and claiming otherwise from these artifacts would be wrong.

**Cost, also against interest** (rates dated 2026-08-19, arithmetic in `VERDICT.md`):

| Serving path | Cost per 1,000 documents |
|---|---|
| Self-hosted CPU, greedy decode | ~$1.09 (quality 0.7686, one seed measured) |
| Self-hosted CPU, voted decode (the decode that passed gate 4) | ~$5.13 |
| Cheap API tier, cold single requests | ~$1.03 |
| Cheap API tier, warm shared-prefix caches + batch | ~$0.22 |

Today the cheap API wins on cost on this corpus, cold caches included. The GPU crossover is the next measurement and publishes either way.

---

## The protocol, and the one time it has run

The standing offer: **you** split your own documents on your own machine, keep the gold labels, and score a single submission with a scorer pinned to a commit named in advance. Terms were frozen and timestamped before any challenger existed (`docs/research/BLIND_HOLDOUT_PROTOCOL.md`, OTS-anchored snapshot).

It has been executed once, as a **labeled dress rehearsal**: an adversarial AI agent authored 50 freight waybills outside this generator's distribution, kept its gold, and scored one blind submission — **0.8792 mean micro-F1, 30/30 valid JSON**. By its author's difficulty tiers: easy 0.950, medium 0.931, mixed 0.944, **hard 0.679**. Its named failure modes: **no unit arithmetic** ("11.5 short tons" → 1150, not 23000), **role-order inversion** (shipper/consignee swapped when the layout inverts), and **fluent confabulation under heavy OCR** ("Ta11ahassee" → "Tampa") — the dangerous one in production, published here because it is true. The challenger's own verdict: *"it survived contact, it did not win it."*

**The corpus was agent-authored and the challenger was an AI agent — this is a protocol-integrity and transfer signal, not a real tenant's documents.** The rehearsal also caught two gaps in the deliverables paperwork (a commit reference the challenger could not fetch; a base model pinned by mutable name), both now hard-gated in code — the runner refuses to emit an unpinned submission.

---

## Check it yourself

```bash
git clone https://github.com/raj200501/arc-ttt-public && cd arc-ttt-public
python3 scripts/verify_verdict.py          # recomputes gate 1 from raw per-receipt records
python3 scripts/addendum_e_summary.py      # recomputes gate 5 under its frozen rule
python3 scripts/read_addendum_d.py         # recomputes the failure in row 3
```

Both verdict scripts are dependency-free. To go past arithmetic to primary evidence,
`python3 scripts/verify_from_primary.py experiments/novel_schema_f_*.json` re-scores every
stored prediction against gold regenerated from the deterministic corpus generator —
it checks the *predictions*, not the summaries. 167 offline tests, no downloads:
`python3 -m pytest tests/ -q`.

---

## Honest ledger

- **Zero customers.** No real-workload win exists. The blind-holdout offer is open precisely because that is the missing evidence.
- **One tenant at 0.53** in the serving mode the economics depend on, and its mechanism is **OPEN** — an earlier explanation (document length) was tested against the banked data, was not supported, and was withdrawn on the record rather than quietly revised.
- **Everything above is 0.5B.** Whether the margin holds at 1.5B–2B, where in-context learning is stronger, is measured next and publishes either way.
- **The ARC Prize track this harness grew out of scored 1.67% public** across three submissions and is formally deprioritized. It is disclosed, not featured.
- **Preregistration ordering:** gates from row 3 onward are SHA-256 Bitcoin-anchored with byte-exact snapshots shipped. The row-1 freeze (2026-08-12) predates the public repository and rests on private git history — stated plainly rather than implied.
