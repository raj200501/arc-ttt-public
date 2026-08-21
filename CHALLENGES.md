# Challenges — the blind-holdout scoreboard

The standing offer is in
[`docs/research/BLIND_HOLDOUT_PROTOCOL.md`](docs/research/BLIND_HOLDOUT_PROTOCOL.md):
you split your own documents on your own machine, you keep the gold
labels, I adapt a small model blind and return one submission, you score
it once with a scorer pinned to a commit named in advance, and the
result publishes here either way. The terms were frozen and
OpenTimestamps-anchored before any challenger existed
(`docs/research/snapshots_BLIND_HOLDOUT_PROTOCOL_2026-08-20T2030Z.md` +
`.ots`); the living file above is byte-identical to that snapshot, and if
they ever differ the snapshot governs.

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
| Predictions, single submission | **72 hours from my written receipt confirmation** — the clock starts when I confirm in writing that both files arrived, passed the format validator, and are readable; not at your send timestamp. If they fail the validator I report that within 2 hours with the validator's exact output and the clock does not start. | **mine** |
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
