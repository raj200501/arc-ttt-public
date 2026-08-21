# Challenge terms — the full document, published before any challenger

This is the document `CHALLENGES.md` tells you to read before you
countersign. It used to say the confidentiality, data-handling and
deletion commitments "live in the terms" without publishing the terms —
so the only data-handling commitment a challenger could actually read was
a sentence asserting that commitments existed somewhere else. An outside
reader named that as the single thing stopping them from routing this to
a portfolio company's counsel, and they were right. It is published here,
in full, before any challenge has run.

**Governing order.** The OpenTimestamps-anchored snapshot of
`BLIND_HOLDOUT_PROTOCOL.md` governs. This document adds detail the
protocol does not carry; where the two ever conflict, **the anchored
snapshot wins and this document yields** — including where yielding is
worse for me. Section 6 records the one place they currently differ and
resolves it against my own interest.

**Not legal advice, and not drafted by a lawyer.** I am a solo founder,
not counsel. This is written to be read and redlined by your counsel, and
I expect changes. Nothing here asks you to waive anything or to accept my
wording as final. If your legal team prefers to paper this on your own
paper entirely, that is fine and faster — the technical protocol is what
matters to me, not whose template it rides on.

---

## 1. Parties and roles

| Role | Who | What they hold |
|---|---|---|
| **Challenger** | you | the source documents, the gold labels, the split |
| **Adapter** | Raj Kashikar (arc-ttt) | the training pairs and the un-labelled holdout text, only |
| **Gold custodian** | you, unless you name someone else | the gold file; never transmitted to the Adapter |
| **OCR provider** | yours if the documents are scans | named here if a third party: ______________ |

If gold custody or OCR sits with a third party (a vendor, a portfolio
company, a client of yours), name them above before countersigning.
Preferring **your** OCR over mine is not a courtesy — it keeps me blinder,
which is better evidence for both of us.

## 2. What I receive, and what I never receive

I receive exactly two files:

- `train.jsonl` — your designated training pairs, **with** their gold
- `holdout.jsonl` — the held-out documents, **text and id only**

I never receive, and do not ask for: the holdout gold, your scoring
script's outputs, your internal labels, or any document outside these two
files. The kit's `split` command produces both files on **your** machine
and never transmits anything; the gold file it writes stays with you.

## 3. Confidentiality

- Your documents and gold are **your confidential information**. I treat
  both as confidential from receipt, with no time limit, and the
  obligation survives the challenge ending, being abandoned, or a
  negative result.
- I will not disclose your documents, your schema, your field names, or
  any content of either file to any third party, and will not quote or
  reproduce document content in any published result, deck, or repository
  without your prior written approval of the exact text.
- I will not use your documents or gold to train, evaluate, or improve
  anything other than this one challenge. They do not enter a training
  corpus, a benchmark, a fine-tune, or a future eval.
- **No external model or API calls anywhere in the adaptation or
  prediction path** (protocol §6). Your documents are never sent to a
  third-party model provider — not OpenAI, not Anthropic, not Google, not
  any hosted inference endpoint. The run is local, offline, and the
  manifest records the exact command so you can check that claim against
  the code at the pinned commit.
- Standard carve-outs apply and are stated rather than assumed:
  information already public, already known to me without obligation,
  independently developed without reference to yours, or compelled by law
  — and in the last case I tell you before disclosing, if I am lawfully
  able to.

## 4. Data handling and deletion

- **Storage.** Both files live on a single machine under my control, in
  one working directory, for the duration of the challenge. No cloud
  sync, no shared drive, no third-party backup service. I do not copy
  them into this repository or any other git history.
- **Deletion deadline.** Your `train.jsonl`, your `holdout.jsonl`, the
  trained adapter, and every derived intermediate (tokenized caches,
  prompt dumps, raw completion logs containing your document text) are
  **permanently deleted within 7 days of the reveal**, or within 7 days
  of either party abandoning the challenge, whichever comes first. On
  request I delete immediately instead, at any point, including
  mid-run — that aborts the challenge and I publish nothing.
- **Confirmation.** I confirm deletion in writing, naming what was
  deleted and when. If you want a stronger form of proof than my word,
  say so before countersigning and we will agree one; I would rather
  agree it in advance than be asked for it afterwards.
- **What survives deletion**, and it is the only thing that does: the
  numeric scores, the failure taxonomy in your words, and the manifest
  hashes. No document text, no gold values, no schema content unless you
  approve it in writing under §3.
- **Breach.** If I ever discover that any of the above was violated, I
  tell you within 24 hours of discovering it, in writing, with what
  happened — and it goes on `CORRECTIONS.md` with the same prominence as
  a wrong number. That page is the reason this pitch exists; it would be
  worthless if it exempted the embarrassing category.

## 5. The technical terms

Fill these in before countersigning. A blank here is a term nobody
agreed, and the runner refuses to emit a submission with an unpinned base
model.

| Term | Value |
|---|---|
| Base checkpoint | name + **immutable revision** + sha256: ______________ |
| Parameter ceiling | ≤ 2B (protocol §6) |
| Scorer | `score_text_output` at public-repo commit ______________ (or your declared scorer) |
| Aggregation | mean per-document micro-F1, invalid JSON scored 0, every holdout id counted exactly once |
| Prediction format | `predictions.jsonl`, one line per holdout id: `{"id", "prediction"}` |
| Submissions | **one**, hashable by you on receipt |
| Adaptation input | only the pairs in your `train.jsonl` |
| External calls | none, anywhere in adaptation or prediction |
| Target schema | yours, declared by you, **including every normalization convention your gold relies on** |

**Normalization conventions are a term, not a detail.** Non-verbatim gold
is only fair if its conventions are declared in advance — if your gold
writes dates as `2026-08-21`, strips currency symbols, or expects a total
computed rather than copied, that belongs in the terms before I see a
document. The rehearsal's single largest failure class was unit
arithmetic; had the convention been declared, it would still have failed,
but it would have failed as evidence rather than as a misunderstanding.

**Baseline parity (§5a).** A strong blind number means little on its own.
The challenger is expected to run, and the terms record, **a baseline of
their choosing on the same holdout with the same scorer** — a frontier
model prompted, a cheap API tier, an incumbent parser, or nothing at all
if you decline. This used to sit in prose on `CHALLENGES.md` as an
invitation, which meant the default challenge produced exactly the
ambiguous artifact that page warns against. It is a term now. If you
decline a baseline, that is recorded too, and the published result says
the number stands without one.

Baseline declared: ______________________  (or: declined)

## 6. The clock — and the one conflict, resolved against me

The anchored protocol (§6) says **"turnaround within 72 hours of
receipt."** `CHALLENGES.md` says the clock starts at my written
confirmation that both files arrived and passed the format validator.
Those are not the same term, and the difference favours me: on the second
reading I could delay indefinitely by simply not confirming.

Resolved, and this is the binding definition:

> The 72-hour clock starts at the **earlier** of (a) my written receipt
> confirmation, or (b) **24 hours after your delivery timestamp** —
> whichever comes first. The validator refinement can shorten my clock
> and can never extend it beyond the anchored term.

If the files fail the format validator I report that within 2 hours with
the validator's exact output, and (b) pauses only for the time you take
to resend. The kit's generated `TERMS.md` carried the older
"72h from document receipt" wording; it is superseded by this section,
and the skeleton has been updated to point here.

| Leg | Elapsed | Whose clock |
|---|---|---|
| Countersigned terms from your yes | ~2 days *(estimate, not a measured number)* | both |
| Documents delivered | typically 1–3 weeks *(estimate)* — your legal and your data export | **yours** |
| Predictions, single submission | **72 hours**, started per the definition above | **mine** |
| Your scoring | 48 hours from submission, one pass, pinned scorer | **yours** |
| Reveal and publication | simultaneous reveal, published within a day either way | both |

Realistically **2–4 weeks from yes to a public number, of which 72 hours
is mine.** The estimates are marked as estimates because nothing in this
repo measures how long a countersignature or a data export takes. The
72-hour and 48-hour legs are terms.

## 7. Publication

- The result publishes in this repository **either way**, as an absolute
  number with no pass/fail spin, beside your baseline if you ran one.
- Under your name, your organization's name, or an agreed anonymization —
  your choice, recorded here: ______________
- You may publish it too, independently, with these terms attached. I do
  not get approval rights over your description of your own result.
- I get no veto on a bad number. That is the entire point: a standing
  offer that only publishes wins is advertising, not evidence.
- **A decision bar, if this is meant to decide something.** If you are
  running this to inform an investment or a purchase, write the bar here
  before the run, so the result is read against a number you set rather
  than one either of us picks afterwards: ______________

## 8. Scope of what a result proves

Stated in the terms rather than left to the write-up, because a
challenger should know in advance what they are and are not buying:

- One corpus, one schema, one submission is **a data point, not a
  benchmark.** It does not establish that the approach generalizes to
  your other document types, or to anyone else's.
- A good mean can hide a bad tier. The per-tier or per-difficulty
  breakdown publishes with the headline **on both sides**, whenever the
  challenger provides tiers. (The rehearsal's mean was 0.8792 while its
  hard tier was 0.679.)
- A win here says nothing about cost. On our own published measurements a
  cheap cached API tier beats our serving on cost today; see
  `VERDICT.md`.
- A loss is a real loss and publishes as one.

## 9. Amendments

Amendments to the protocol or to these terms after a challenge begins
bind only **future** challenges. Your countersigned copy governs your
challenge for its duration, whatever this file later says.

---

**Countersigned**

| | Name | Organization | Date |
|---|---|---|---|
| Challenger | | | |
| Adapter | Raj Kashikar | arc-ttt (entity in formation) | |

*Entity status, stated plainly rather than implied: the company is in
formation. If a countersignature from an incorporated entity is a
blocker, say so — there is a faster path that needs no data agreement at
all, which is to run the first challenge on public records (federal and
state contract award notices are genuinely filthy OCR and are already
public).*
