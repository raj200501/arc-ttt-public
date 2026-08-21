# Corrections

Every number or claim withdrawn or revised here, with the date and the
cause. Self-correction is only evidence if it is **countable**, so this
page collects what is otherwise scattered across the spec's errata
block, `VERDICT.md` rows, and commit messages.

**If you find something that belongs on this page and is not here, open
an issue — it goes up within 24 hours, with your name on it.**

Two things this page deliberately does *not* do: it does not rewrite
frozen preregistration text (corrections land as dated errata beside it,
per the spec's own rule), and it does not quietly improve a number. A
correction that moves a headline says so.

---

## Claims withdrawn

| Date | Claim as published | What happened | Where |
|---|---|---|---|
| 2026-08-20 | Seed 2's document-only gap (0.5282) was explained by document length/complexity | **Withdrawn.** The conjecture was tested against the banked data: documents, gold and schemas are statistically identical across seeds, and within-seed short/long halves score the same. It was not supported. The mechanism is now recorded as **OPEN** rather than replaced with a new story. | spec erratum P11; `experiments/novel_f_length_analysis_2026-08-20.json` |
| 2026-08-20 | "Every k=30 demo prompt runs at 98–100% of the token budget" (stated as a property of the corpus) | **Rescoped.** True of the fixed-geometry corpus; the E-r2 tenants measure 6,619–7,447 decode tokens (81–91%). The ceiling binds where the shape is fixed, not universally. | `VERDICT.md` context-ceiling row |
| 2026-08-19 | Quality "0.95–0.99 at 20–58x smaller payloads" | **Withdrawn as a conflation.** Those numbers come from two different serving modes: demo-context serving holds high quality but ships the examples in every request (no payload win); document-only serving earns the payload advantage at 0.53–0.94 by seed. The split is now stated wherever either number appears. | `VERDICT.md` mode rows |
| 2026-08-21 | The rehearsal's **0.8792**, published for a week as the transfer signal | **Qualified, and against us.** The paired baseline it lacked has now been run against a bar frozen first: prompted 0.7836, adapted 0.8833, delta +9.97 — but the sign test disagreed (8W/5L/17T, p=0.29), so the two-statistics rule returns **FAIL**. 17 of 30 documents tie, and 63% of the delta comes from 3 documents where the prompted arm emitted invalid JSON. On this corpus the measured benefit is **output-format reliability, not extraction accuracy**. 0.8792 stands as an adapted score; it is no longer offered as evidence that adaptation beat prompting there | `VERDICT.md` rehearsal rows; `experiments/blind_rehearsal_baseline_2026-08-21.json` |
| 2026-08-19 | An early cost comparison favouring self-hosted serving | **Withdrawn.** Measured: cheap cached API tiers beat our CPU serving on this corpus today, cold caches included. The row is published against our own interest. | `VERDICT.md` serving-cost + cache-state rows |

## Numbers corrected

| Date | Was | Is | Cause |
|---|---|---|---|
| 2026-08-21 | Addendum E cluster CI `[+32.75, +47.95]` | **`[+31.0, +49.7]`** | **Found by an outside reader who recomputed our published interval from our own artifacts, and reported it against our interest.** Our reader's `t95()` was a lookup table starting at df=39, with a fallback that returned the invented constant `2.09` for anything smaller. Addendum E clusters over six seeds — df=5, where the true quantile is 2.5706 — so the published interval was computed at roughly t(df≈19) and came out **~19% too narrow, in the direction that flatters us**. The verdict is unchanged: the corrected lower bound `+31.0` still clears the `+5` bar six-fold. Erratum P14(a). The table is gone rather than extended — `t95()` now computes the exact Student-t quantile (bisected CDF over a continued-fraction incomplete beta, stdlib only), because a wider table fixes the instance and leaves the class. |
| 2026-08-21 | Addendum E cluster CI `[+32.8, +48.0]` | `[+32.75, +47.95]` | The upper bound was rounded up. Superseded the same day by the row above — both bounds were wrong for a deeper reason than rounding. |
| 2026-08-21 | k=30 gate receipt-level CI `[42.8, 49.4]` | **`[42.9, 49.4]`** | **Third correction of one number, in one day, for three different reasons — recorded in full because the pattern matters more than the digit.** (1) `verify_verdict.py` used the normal quantile 1.96 where the authorized reader used t; the 0.033 gap sat inside the 1e-3 cross-check tolerance, so nothing failed, and `VERDICT.md` quoted a number its own "check it" command contradicted. (2) The repair transcribed the constant `1.980` instead of asking the reader for its quantile — closer, still not t at df=157 (1.9752), and it is why the value moved to 42.8 at all. (3) The reader's own quantile was the broken table above. Every one of the three was **a copied constant**, so nothing is copied any more: both readers call one computed `t95()`, and `tests/test_readers_agree.py` now fails on any hardcoded quantile in either file rather than on a value. Erratum P14(b) supersedes P12(a); P12(b)'s record of an unlogged in-place edit of this same figure in frozen text on 2026-08-19 stands. |
| 2026-08-21 | Demo-context k=30 quality `0.977–0.993` | `0.936–0.993` | Addendum E banked six more k=30 tenants; the old range was the pre-E three. |
| 2026-08-21 | Outbound copy stated that a competent team clones this stack in **"a quarter"** | **"two to four weeks"** — the figure our standing answer to that question has always given | Two different answers to the same question, and the invented one was in the copy that goes out, written while drafting a paragraph about not inventing numbers. Caught within the hour in-house. Outbound copy is the one artifact that cannot be corrected after it ships, so a test now fails if a rendered message states any figure that appears in no evidence page. |
| 2026-08-21 | Offline test count (variously 83, 128, 137, 144, 151 across documents) | `172` | Re-synced to the measured collection count — for the **second** time in one day. The 08-21 re-sync to `167` was stale within hours, because the commit that performed it also added tests; two outside readers found it independently by running `pytest -q`, which is the one command a claim like ours invites. A hand-sync was never a fix, since a hand-sync is what had already failed. `tests/test_doc_counts_agree.py` now reads the collected count and fails any document that disagrees. |
| 2026-08-20 | Addendum E sign test quoted as "340 wins / 5 losses across 360" | `340W / 5L / 15T` | Dropping the ties left 15 documents unaccounted for. |
| 2026-08-19 | Cheap-API cost advantage stated as "5–20x" | parity-to-~23x, by decode | The stated range did not match the stated figures. |

## Overstatements corrected

| Date | Was | Is | Cause |
|---|---|---|---|
| 2026-08-21 | "Both deliverables gaps are now hard-gated in code" | One hard-gated, one documented in the runner | Only the base-model pin is enforced by code; the commit-regenerability cure is documentation. |
| 2026-08-21 | The blind rehearsal described so that "its gold never left its machine" | The challenger agent was **operated by us, in the same working session and on the same host**; gold was withheld in process and hash-committed before submission — procedural blindness, not third-party custody. (An interim wording said "a separate session"; the artifact's own recorded command path shows one session, so that phrasing was removed too.) | The original phrasing implied an independent machine and an independent party. Neither was true. |
| 2026-08-21 | "Six fresh tenant schemas, each a different shape" | Six fresh tenants, shapes varying 6–7 fields across 2–3 groups | The original 6–12 field range was compacted after it proved unmeasurable at k=30; the wider phrasing described the abandoned design, not the run. |
| 2026-08-21 | Addendum E's "zero documents excluded", stated plainly | Zero excluded **because those seeds were screened for token-budget fit before any arm ran** (token-count-only, outcome-blind, banked) | Zero attrition is a construction of the screen, not a property the corpus happened to have. |
| 2026-08-21 | "All gates from D onward are chain-anchored pre-data" | E's +5 bar and decision rule are chain-anchored; E-r2's measurability amendment was git-committed pre-data and anchored the next day | The blanket claim overstated what the chain attests for the amendment. |
| 2026-08-21 | Addendum E "**retires** the shared-geometry objection" (printed by the verdict script itself) | **Narrows** it. Same word everywhere now: script output, `VERDICT.md`, `EVIDENCE.md`, outbound copy | One result was described at three different strengths in three places — "retires" in the script, "NARROWS" in `VERDICT.md`, "answered" in the outbound email — and the copy a stranger reads first had picked the strongest. E varies field count and group count *inside one generator family*; it does not touch the generator-family objection, and its shapes are smaller than the gate corpus, not larger. |
| 2026-08-21 | The 3-line DM said "my hardest test failed, the fix passed its own frozen bar 12.5h later" | The hardest test is CORD — the only real public dataset — which failed at all three scales and is **still unfixed**. The repaired one is a different gate | In three lines with no room for the CORD negative, the only failure disclosed was the one we repaired. The repo pairs them scrupulously; the shortest outbound artifact did not, and it is the one read first. |
| 2026-08-21 | The same DM quoted the rehearsal as "0.88, failure taxonomy published" | 0.88 mean **and 0.68 on its hard tier** | `CHALLENGES.md` states the rule that a good mean can hide a bad tier and that the per-tier breakdown publishes with the headline on both sides. The DM broke a published claim rule of ours, in the one artifact a reader sees before the rule. |
| 2026-08-21 | The outbound copy attributed the 22 gate-1 exclusions to "the length tail" | Budget-edge tokenization jitter at 8,184–8,195 tokens against an 8,192 budget — **not** long documents | Erratum P11 retired the length explanation in the repo, and the outbound copy kept carrying it. Exactly the failure this page named a day earlier: "a disclosure whose honest half was added while the misleading half stayed in the outbound copy." |

## Corrections to this page

A page about corrections has to correct itself in public or it is
decoration. The receipt-CI row above was published with the wrong
explanation and is annotated in place rather than rewritten. The lesson
that produced it is worth stating plainly, because an outside reviewer
reached it before we did: **on 2026-08-21 we shipped roughly forty
corrections in a few hours, and the correction layer itself did not
audit clean** — a fix that contradicted its own verification command, a
disclosure whose honest half was added while the misleading half stayed
in the outbound copy, and two live-conversation scripts still quoting
claims this page had already retired. Editing faster than you can verify
is the same failure the preregistration machinery exists to prevent,
just at a smaller scale. The remedies are here rather than in a promise:
`tests/test_readers_agree.py`, `tests/test_evidence_card.py`, and
`tests/test_verify_without_torch.py` now fail rather than drift.

**Update, later the same day, and it goes the wrong way.** Outside
readers ran the obvious commands and found two more: the test count had
drifted *again* within hours of the re-sync above, and — the serious one
— a shipped PASS carried an interval that was too narrow in our own
favour, because our quantile function invented a constant for any
cluster smaller than forty units. Neither was caught by us. So the
honest tally is that a correction layer written to catch exactly this
class of error has now shipped **three successive wrong versions of one
number** and one wrong interval in a live result, and every instance had
the same cause: a constant copied instead of computed, or a number typed
instead of measured.

That is the lesson worth more than any individual fix, and it is why the
remedies in this round are all of the same shape — the constant is
deleted and computed, the count is measured by a test rather than typed
by a person, and the tests assert the *absence of the cause* rather than
the presence of a value. A number a human has to remember to update is a
number that will be wrong. The next reader to find one should still
expect to find one; the standing offer below is not rhetorical.

**And one more turn of the same screw, worth recording because it is
the most instructive thing here.** Three separate fixes shipped that day
failed *in the same way as the bug they were fixing*: the check written
to stop the count drifting passed clean over seven stale claims, because
it only knew the phrasings already fixed; the fix for a duplicated
paragraph deleted the signature instead, because it worked by position
rather than by naming what it removed; and a paragraph arguing against
inventing numbers invented one. Each was caught within the hour, none by
anyone outside. The rule that fell out, and that the remedies now
follow: **a guard written from the instances it just fixed reopens the
same hole.** Fix the class, put the rule in one place, and let the thing
that fixes and the thing that checks be the same code — otherwise the
disagreement between them will always favour the stale number, because a
fixer's blind spot is invisible until its checker fails.

## Defects found in our own verification path

| Date | Defect | Fix |
|---|---|---|
| 2026-08-21 | The test-count check added that morning **passed clean over seven stale live claims** in our own presentation and talking-point documents — the numbers said out loud in a room. It only knew the phrasings that had already been fixed ("N offline tests"), not the hyphenated "167-test suite" | The match rule moved into `scripts/sync_test_counts.py`, which does the fixing, is imported by the test to do the checking, and is run by the export inside the exported tree — one rule, three users, no copies. Running it found **19** stale claims in documents that had already been "fixed" twice the same day. Its test carries a regression list of every phrasing that once slipped past |
| 2026-08-21 | `t95()` in the authorized reader was a lookup table whose smallest entry was df=39, with a fallback returning the invented constant `2.09`. Any interval clustered over fewer than 40 units silently borrowed a quantile from the wrong distribution — which is every cluster-level CI this project publishes. It reached a shipped PASS (Addendum E, df=5) and made the interval ~19% too narrow in our favour | The table is deleted, not extended: `t95()` computes the exact quantile by bisecting the Student-t CDF (regularized incomplete beta, continued fraction, stdlib only). `verify_verdict.py` no longer carries its own constants and imports the same function, so the repo has one estimator. `tests/test_readers_agree.py` pins the computed values against published tables and **fails on the presence of any hardcoded quantile** in either reader |
| 2026-08-21 | `CHALLENGES.md` told challengers that confidentiality, data handling and the deletion deadline "live in the terms" and to ask for the terms before countersigning — and the terms document did not exist publicly. The anchored 42-line protocol contains no data-handling language at all, so the only commitment a challenger could read was a sentence asserting that commitments existed somewhere they could not see | `docs/research/CHALLENGE_TERMS.md` published in full, before any challenger: party roles, confidentiality with its carve-outs stated, storage and a 7-day deletion deadline with written confirmation, breach disclosure inside 24 hours, publication rights with no veto on a bad number, and a baseline-parity clause promoted from prose to a term. The clock conflict is resolved **against us**: three documents defined the 72-hour start three different ways, and the binding definition is now the EARLIER of our written confirmation or 24h after delivery, so the refinement can shorten our clock and never extend it |
| 2026-08-21 | `src/arcttt/serve.py` — the file a reader opens looking for the product — documented an ARC grid-task endpoint while the pitch describes document→schema serving | Scope stated at the top of the file: this endpoint is ARC-shaped, it is not the product surface, and the document path is named. Said plainly rather than fixed with a rename — closing the gap is product work |
| 2026-08-21 | The one-command demo (`demo/run_endpoint_demo.sh`) defaulted to CORD — the corpus our own Addendum A says the recipe FAILS on at all three scales. The invitation to "touch it rather than audit it" ran the documented loss, and the surrounding copy mentioned neither the ~950MB download nor the 6-10 minute runtime | Default switched to the novel-schema demo corpus, the regime the gates actually cover; CORD kept one env var away, because the negative is part of the evidence; cost and runtime stated in the script header |
| 2026-08-21 | `scripts/verify_from_primary.py` — the command we ask readers to run — imported the scorer from a module that imports torch at load time, so it failed with `ModuleNotFoundError` on any machine without PyTorch, while the README said the scoring scripts needed none | Scoring extracted to a torch-free `arcttt.scoring` (re-exported, so existing callers are unchanged); `tests/test_verify_without_torch.py` runs the whole verification path against a stub that raises exactly like a missing module |
| 2026-08-20 | The blind-holdout protocol's OpenTimestamps proof attested a **draft** of the file, not the published one — `ots verify` failed against the living document | The broken proof was deleted, the protocol amended before any challenger existed, and a byte-exact snapshot re-stamped |
| 2026-08-20 | A challenge submission could be emitted with the base model pinned by mutable name only | The runner now refuses to emit an unpinned manifest |
| 2026-08-19 | Geometry resolution in `verify_from_primary.py` guessed from filenames, producing 60 false mismatches on the first Addendum E artifact | Geometry is now resolved by matching the artifact's stored schema against the regenerated schema; a failure to match under any geometry is reported as an integrity failure |

## Provenance errata

The spec carries eleven dated errata (**P1–P11**) from a line-by-line
audit of its own text against git history, OTS anchors, and artifacts —
including stamp-provenance drift (P1), post-freeze wording edits to a
frozen addendum (P3), a test claim that overstated coverage (P5), and
sign-test tie conventions (P9). They are in
`docs/research/ENTERPRISE_EVAL_SPEC.md` under *Errata & provenance
amendments*, and they are not rewritten in place — the frozen text
stands and the correction sits beside it, dated.
