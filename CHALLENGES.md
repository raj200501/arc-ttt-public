# Challenges — the blind-holdout scoreboard

## The offer, in three lines

**Send 50 of your own labeled documents. Keep the gold labels — I never
see them. I return one submission, you score it once with a scorer pinned
to a commit named before I start, and the result publishes here whether it
makes me look good or not.**

Your side is roughly two hours of work: export the documents, run one
command from the kit, keep the gold file. Mine is 72 hours. Full terms,
the kit, the legal skeleton and an honest 2–4 week calendar are below —
but you do not need to read any of it to say yes, and **row 1 is open.**

If your prompted baseline is already good, this will tell you that, and
that answer is worth having before anyone sells you anything. There is a
script for exactly that case: `python3 scripts/try_your_documents.py
--docs mydocs.jsonl` prints, in words, when there is nothing here for you
to buy.

**Before you spend two hours on this, read the result that says you
probably should not.** On 2026-08-22 we put the cheapest hosted API tier
on our own 30 held-out waybills: it scored **0.9708–1.0000 across four runs**
(mean 0.9865) against our adapted 0.5B's 0.8833, and our arm won at most
one of the thirty in any run. **If your documents can be sent to a hosted API, send
them there instead of taking this slot.** The only reason to run this
offer is that yours cannot leave your building. We would rather tell you
that up front than take two hours of your time to arrive at it.
`experiments/waybill_market_baseline_gemini-3.5-flash-lite_2026-08-22.json`.

---

## The terms, and one thing that is not byte-identical

The standing offer is in
[`docs/research/BLIND_HOLDOUT_PROTOCOL.md`](docs/research/BLIND_HOLDOUT_PROTOCOL.md):
you split your own documents on your own machine, you keep the gold
labels, I adapt a small model blind and return one submission, you score
it once with a scorer pinned to a commit named in advance, and the
result publishes here either way. The terms were frozen and
OpenTimestamps-anchored before any challenger existed
(`docs/research/snapshots_BLIND_HOLDOUT_PROTOCOL_2026-08-20T2030Z.md` +
`.ots`). **The living file is NOT byte-identical to that snapshot** — on
2026-08-21 a dated note was appended to it recording that the binding
definition of "receipt" moved to `CHALLENGE_TERMS.md` §6, which SHORTENS
our own clock. The snapshot governs on any conflict, the appended text
is against our interest, and this sentence used to claim byte-identity
until an outside reader ran `sha256sum` on both files and found it
false. Run it yourself; a repo that says "check me" cannot carry a
checkable claim that is wrong.

This page is the ledger of who has taken it. One row has run. The next
row is open.

---

## The scoreboard

| # | Challenger | Terms countersigned | Corpus | Result | Published at |
|---|---|---|---|---|---|
| 0 | **AI agent, adversarial — operated by me, in the same working session and on the same host.** A dress rehearsal, not a third party. | 2026-08-20, self-issued: same operator on both sides. Kit-generated TERMS pinned the scorer to commit `617449ee1185…`. | 50 **agent-authored** freight waybills — realistic post-OCR text with deliberate non-verbatim and structural traps, from outside this repo's generator distribution; **not a real company's scans**. Split 20 train / 30 holdout, seed 41. Gold withheld and hash-committed before my submission (`gold_holdout.jsonl` sha256 `13e9cc7f9955…`). | **0.8792** mean per-document micro-F1 over 30 holdout docs, 0 scored 0 as invalid/missing/duplicate (30/30 valid JSON). By the challenger's own difficulty tiers: easy 0.950 / medium 0.931 / mixed 0.944 / **hard 0.679**. Adapted blind on the 20 pairs in 53.6 s on CPU; adapter sha256 `64baf52c0000…`. | [`experiments/blind_rehearsal_2026-08-20.json`](experiments/blind_rehearsal_2026-08-20.json) (per-doc scores, trap taxonomy, gaps and cures), summarized in [`EVIDENCE.md`](EVIDENCE.md) and [`VERDICT.md`](VERDICT.md) |
| 1 | **OPEN** | — | — | — | within a day of scoring, pass or fail |

**How row 0 is labeled, and it is the same label everywhere it appears:**
the challenger was an AI agent, the corpus was agent-authored, and I
operated it — same session, same host. Gold was withheld in process and
hash-committed before my submission, but **that is procedural blindness,
not third-party custody.** It is a protocol-integrity and transfer
signal, not the real event, and it caps what the number proves. The
rehearsal's recorded verdict, verbatim: *"it survived contact, it did not
win it — 0.88 on friendly-adversarial synthetic is a real signal and an
honest ceiling, and the number publishes as-is."*

**Row 0's missing baseline arm has now been run, and it FAILED its bar
(2026-08-21).** 0.8792 was an adapted score with nothing to subtract
from it, so the missing arm was run against a rule frozen and published
first. Matched greedy decode on the same 30 waybills: prompted
**0.7836**, adapted **0.8833**, paired delta **+9.97** — twice the +5
bar, CI excluding zero — but the sign test disagreed (**8W/5L/17T,
p=0.29**), and the preregistered two-statistics rule requires both. So
the verdict is **FAIL**, and it publishes as FAIL.

The rule caught exactly what it is for. **17 of 30 documents tie** (12
of them both-perfect), and **63% of the total delta comes from the 3
documents where the prompted arm emitted invalid JSON and scored 0.**
Where both arms produce valid JSON it is close to a wash. Four of the five
adapted losses are a single field (−0.125); the fifth, `h-3303`, is −0.0278. The honest reading: on this realistic
corpus, adaptation's measured benefit is **output-format reliability,
not extraction accuracy**.

**And we then measured the rival explanation for that, which beats us
here.** If the benefit is output format, a JSON-grammar-constrained
decoder buys it for free — no adaptation, no per-tenant weights, no
training. That explanation had never been named anywhere in this
repository. `python3 scripts/format_counterfactual.py` bounds it from
the raw arms published above, stdlib-only, in five readings:

| reading | what it grants the rival explanation | delta | sign test |
|---|---|---|---|
| as measured | nothing (control; reproduces the published row) | **+9.97** | 8W/5L/17T |
| schema-key pruned | the prompted arm's 12 off-schema keys deleted — a schema-constrained decoder cannot emit them | **+9.26** | 8W/5L/17T |
| + unparseable imputed | each of the 3 unparseable prompted outputs becomes valid JSON of that arm's *typical* quality | **+0.47** | 5W/8L/17T |
| + unparseable perfect | those 3 become PERFECT extractions (impossible; a hard ceiling) | **−0.74** | 5W/8L/17T |
| format-neutral, key-pruned | the 27 documents both arms parsed, but still key-pruned — so still a grant | +3.34 | 5W/5L/17T |
| **format-neutral, nothing granted** | **nothing at all** — the arms exactly as decoded, on the 27 documents both parsed | **+4.14** | **5W/5L/17T** |

The last row is the one that assumes nothing, and it is the one to
quote: on documents where format reliability *cannot* be the
explanation, and with nothing else granted, the paired delta is **+4.14
— under this project's own +5 bar, with a sign test that is a coin
flip.** **The rival explanation is not refuted, and neither is ours.**
At n=27 the interval is **[−3.5, +11.8]** and the sign test is
**5W/5L, p=0.62** — this corpus cannot distinguish adaptation from a
free decoder in either direction. Reading C, the one that models a
decoder that actually exists, sits at **+0.47 with the sign test
pointing the wrong way**. That is worse for us than a clean loss would
be, and it is what the numbers say.

*(This paragraph read "on this corpus it is the better-supported one"
until an outside reader pointed out that a 5W/5L sign test at p=0.62
supports nothing. It was an over-claim made against our own interest,
which is still an over-claim — the direction it points does not make it
true.)*

*(Until 2026-08-22 the key-pruned row above was the one labeled "assumes
nothing," at +3.34. It does grant something — the key-pruning — and an
outside auditor caught the mislabel by noticing that `VERDICT.md`
already carried +4.14 for the same subset. The reading that really
grants nothing was added rather than the label quietly changed.)*

Two things that does not mean. It does not touch gates 1, 4 and 5: the
prompted baseline there is 0.4333–0.6208, the arms are not separated by
format failures, and pruning has nothing to prune. And it is **post-hoc
analysis of banked data, not a gate** — no bar was frozen for it before
the numbers existed, which the artifact says in its own `status` field.

What it does mean is on the record and costs us something: the honest
v1 ships constrained decoding, and the adaptation layer has to earn its
keep **on top of** it. That paired comparison is now a PENDING row in
`VERDICT.md` with its bar and all three readings frozen before the arms
exist — including reading (c), which says that if the delta is at or
below zero then constrained decoding is the product on this corpus and
per-tenant adaptation is not justified by it. That reading publishes
too.

### Does that objection reach the headline gates? Measured: no.

The same question was pushed onto gates 1, 4 and 5 rather than left where
it landed. `PYTHONPATH=src python3
scripts/schema_conformance_decomposition.py` runs it in under a second,
regenerating gold from the deterministic generator rather than reading any
stored copy.

**Part 1 — format.** Invalid JSON across BOTH arms: **0 of 158** (gate 1),
**0 of 360** (gate 5), **1 of 158** (gate 4, on our arm) — and the denominator is post-exclusion: gate 1's 22 designed receipts that exceeded the frozen token budget produced no completion at all in either arm, which is the most complete output failure there is. The restriction is silent on those 22 and speaks only for the 158 scored. Restricting to
documents both arms parsed is a no-op:

| gate | as measured | format-neutral |
|---|---|---|
| 1 | +46.5 | **+46.5** |
| 4 | +24.0 | **+24.5** |
| 5 | +40.4 | **+40.4** |

**Part 2 — schema conformance.** Valid JSON is not the same as the right
keys. On gate 5 — the only gate whose baseline arm stores its predictions
— that baseline was rebuilt under gold's key paths keeping its own values
(exactly what a schema-constrained decoder guarantees), then rebuilt again
forgiving nesting mistakes (more than any real decoder gives you):

| baseline, as decoded | path-repaired | name-repaired | adapted |
|---|---|---|---|
| 0.5726 | 0.5726 | 0.5726 | **0.9761** |

**0.0% of the gap closed.** The reason is measured, not inferred: across
all 360 documents the baseline's key-path precision and recall are both
**1.00**, its key set is exactly gold's on **60/60 documents per tenant**,
and it puts the wrong content in **43% of the fields it names correctly** — a six-tenant mean over a **27%–51%** spread, and arithmetically 1 − the baseline's 0.5726 rather than an independent measurement.
There is nothing for a decoder to repair, because the baseline already
emits the tenant's schema perfectly — it just fills it with the wrong
values.

So the two mechanisms are opposite, and together they are the honest
scoping of this project: **on the headline gates adaptation buys
value-level extraction accuracy that no decoder can supply; on the
realistic waybill corpus, where the baseline is already close on content,
what is left is mostly format and a decoder plausibly takes it.**

Both analyses are post-hoc reads of banked data, labeled so in their own
artifacts. The repair function is pinned by a test that fails if it is
ever a no-op — "0.0% closed" is only evidence if the repair demonstrably
works, and you should not take our word for that either.

So read 0.8792 as "a small adapted model scored this on that corpus" —
not as evidence that adapting beat prompting on it, because the paired
test says it did not, by our own rule. Artifact:
[`experiments/blind_rehearsal_baseline_2026-08-21.json`](experiments/blind_rehearsal_baseline_2026-08-21.json).

This is also why §5a of the terms now *requires* a challenger to declare
a baseline or record that they declined. A blind number without one is
ambiguous by construction — and row 0 is the proof, since its headline
survived a week before the arm that qualifies it existed.

What the rehearsal's challenger recorded as generalizing: non-verbatim
date normalization (including scheduled-vs-actual selection),
governing-figure selection (net over tare/gross, reweigh, scale-over-quote,
multi-handling-unit totals), non-verbatim charges, distractor rejection
(bill-to third parties, order dates, POs), light OCR digit repair.
What it recorded as failing: **no unit arithmetic** ("11.5 short tons" →
`1150`, not 23000), **role-order inversion** (a positional heuristic, not
role understanding, so shipper/consignee swap when the layout inverts),
**fluent confabulation of names and cities under heavy OCR**
("Ta11ahassee" → "Tampa") — the dangerous production failure mode — and
name repair that is asymmetric to digit repair.

It also found two gaps in the deliverables paperwork, both of which
change what a real run does: the manifest cited a workspace commit the
challenger could not fetch (the artifact records that `src`+`scripts` at
public `89acea6` are byte-identical to the cited commit, sync-gate
verified; the cure is that real runs execute from a clone of the *public*
repo at the terms-pinned commit), and the base model was pinned by
mutable name only (the runner now records the resolved revision and
checkpoint hashes and **refuses** to emit an unpinned manifest without
`--allow-unpinned`, which is not valid for a real challenge).

---

## How to take the slot

> **Send no documents before the terms are countersigned.**
> Confidentiality, data handling, and the deletion deadline for your
> `train.jsonl`, `holdout.jsonl`, and the trained adapter live *in the
> terms*. Countersign first; ship files second. The kit is built so you
> can generate the package and read the terms without sending anything.

**1. Run the kit on your machine.** `split` is stdlib-only — no torch, no
install, no network:

```bash
python3 scripts/make_challenge.py split \
    --docs your_documents.jsonl \
    --train-k 20 --seed 41 \
    --out-dir challenge \
    --name "your-org" \
    --commit <public-repo commit to pin the scorer to>
```

Input is one JSON object per line: `{"id": ..., "text": ..., "gold": {...}}`.
Real documents with your own held-out gold are preferred — OCR noise,
non-verbatim values, and ambiguity are welcome and no fairness-invariant
guarantees are requested. An invented schema is accepted when your data
cannot leave your building.

**What you send me:** `train.jsonl` (the labeled adaptation pairs),
`holdout.jsonl` (text only, gold stripped), and the countersigned
`TERMS.md`.

**What you keep, and I never see:** `gold_holdout.jsonl`. The kit prints
its sha256 and the command to anchor it before anything ships:

```bash
ots stamp challenge/gold_holdout.jsonl   # pip install opentimestamps-client
```

**2. Fill in the terms before countersigning.** The generated `TERMS.md`
is a skeleton with two sections you own: the target schema (field names,
types, and **every normalization convention your gold relies on** —
non-verbatim gold is only fair if its conventions are declared) and the
pinned scorer commit. The terms you countersign also carry riders the
skeleton does not print: the base-checkpoint pin (name + immutable
revision + checkpoint sha256), the prediction format, when the clock
starts, confidentiality and adapter deletion, party roles if gold custody
or OCR sits with someone else, a decision bar if the result is meant to
decide something, and the calendar below. **The full terms document is
published: [`docs/research/CHALLENGE_TERMS.md`](docs/research/CHALLENGE_TERMS.md)**
— read it before you countersign. It is public specifically so you can
hand it to counsel without asking me for it first. (It was not published
until 2026-08-21: this page told you to countersign terms it did not let
you read, and an outside reader correctly called that the thing blocking
a real challenge. The anchored 42-line protocol contains no
data-handling language at all, which is why the gap was invisible from
the protocol alone.)

**3. The pinned scorer.** The default is `score_text_output` at the
public-repo commit named in your terms (`--commit`), aggregated as mean
per-document micro-F1 with invalid JSON scored 0, every holdout id
counted exactly once. Pin a commit you can `git clone` and fetch — the
rehearsal's pin was a commit that did not exist in the public repo, which
is exactly the gap it caught. You may declare any other scorer up front
instead. Semantics of the default, so your labels are written against the
real thing: it matches (field-path, normalized-value) pairs; string values
are whitespace-collapsed and **casefolded**; numeric strings compare
canonically (`"12,000"` == `"12000"` == `12000.0`); booleans and null by
JSON spelling. The code at the pinned commit governs on any conflict with
that summary. Sanity-check it against your own train gold before shipping
— scoring your gold as its own predictions should return 1.0:

```bash
python3 scripts/make_challenge.py score --pred <your-gold-as-predictions> --gold <your-gold>
```

(`score` needs `pip install torch` and `pip install -e .`; `split` does
not.)

**4. What I run, and what comes back.** From a fresh clone of the public
repo at the terms-pinned commit:

```bash
python3 scripts/run_challenge.py --train train.jsonl \
    --holdout holdout.jsonl --out-dir run/ --seed 1
```

`run_challenge.py` reads only the `text` fields of `holdout.jsonl` and is
never given a gold file at all. Back come `predictions.jsonl` (one line
per holdout id, exactly once — missing or duplicate ids score 0 for that
document), `adapter.pt`, and `manifest.json` carrying the adapter sha256,
the repo commit, the base-model pin, the exact command, and the seed — so
the predictions are regenerable, not merely hashed. Constraints, from the
protocol: base model at or under 2B parameters, adapted offline on only
the pairs you supply, **no external model or API calls anywhere in the
adaptation or prediction path**, one submission only. You may hash the
submission on receipt.

**5. The honest calendar.** "72 hours to a published number" is true and
misleading, because 72 hours is only my leg:

| Leg | Elapsed | Whose clock |
|---|---|---|
| Countersigned terms from your yes | ~2 days *(offered estimate, not a measured number)* | both |
| Documents delivered | typically 1–3 weeks *(offered estimate)* — driven by your legal and your data export, not by me | **yours** |
| Predictions, single submission | **72 hours, starting at the EARLIER of (a) my written receipt confirmation or (b) 24 hours after your delivery timestamp.** Corrected 2026-08-21: this row used to say the clock starts only at my confirmation, which the anchored protocol does not say and which would have let me delay indefinitely by never confirming. The validator refinement can shorten my clock and can never extend it. If the files fail the format validator I report that within 2 hours with the validator's exact output, and (b) pauses only for the time you take to resend. Binding text: [`CHALLENGE_TERMS.md` §6](docs/research/CHALLENGE_TERMS.md). | **mine** |
| Your scoring | 48 hours from submission, one pass, pinned scorer | **yours** |
| Reveal and publication | simultaneous reveal, published within a day either way | both |

Realistically **2–4 weeks from yes to a public number, of which 72 hours
is mine.** The two estimates are marked as estimates because they are:
nothing in this repo measures how long a countersignature or a data
export takes. The 72-hour and 48-hour legs are terms, not estimates.

Amendments to the protocol after a challenge begins bind only future
challenges. Both parties may publish the result, pass or fail, with the
terms attached, under your name or an agreed anonymization.

---

## What a result would prove, and what it would not

**Would.** It would be the first number this project has on documents it
did not generate, whose labels it cannot see. Every gate in
[`EVIDENCE.md`](EVIDENCE.md) runs on a synthetic corpus; the one real
public dataset tried (CORD receipts) **failed** its preregistered gates at
all three scales, −7.3 / −11.5 / −4.5 F1. That gap is precisely why this
offer is open, and closing it is worth more than another synthetic rung.
It would also put the rehearsal's named failure modes — unit arithmetic,
role-order inversion, OCR-name confabulation — against real noise instead
of an agent's imitation of it, and it would test the deliverables chain
(fetchable commit, pinned base checkpoint, regenerable predictions) under
someone else's audit.

**Would not.** One corpus, one schema, roughly 30 held-out documents, one
submission: that is a data point, not a benchmark, and a single run
carries no confidence interval. It would not establish quality
differentiation against the market — on this project's own synthetic
corpus **a frontier model and the cheapest API tier both score 1.00 by
plain prompting**, so a strong blind number means little unless you run
your own baseline on the same holdout, which the protocol explicitly
invites and publishes beside the result. It would not say anything about
cost; today the cheap API tier wins on cost on this corpus even with cold
caches (~$1.03/1k documents versus ~$5.13/1k for the voted CPU decode
that passed the gate). Everything measured here is at 0.5B. And a good
mean can hide a bad tier — the rehearsal's mean was 0.8792 while its hard
tier was 0.679, which is why the per-tier breakdown publishes with the
headline on both sides.

It would also not, on its own, retire the objection row 0 exists to name.
The rehearsal had procedural blindness without third-party custody; a real
challenge fixes exactly that, and nothing else, unless it also brings real
documents. A failure publishes on this page with the same prominence as a
pass — that is the whole point of pinning the scorer before the data
exists.
