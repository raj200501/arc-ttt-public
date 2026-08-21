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
| 2026-08-19 | An early cost comparison favouring self-hosted serving | **Withdrawn.** Measured: cheap cached API tiers beat our CPU serving on this corpus today, cold caches included. The row is published against our own interest. | `VERDICT.md` serving-cost + cache-state rows |

## Numbers corrected

| Date | Was | Is | Cause |
|---|---|---|---|
| 2026-08-21 | Addendum E cluster CI `[+32.8, +48.0]` | `[+32.75, +47.95]` | The upper bound was rounded up. Now stated at the artifact's own precision. |
| 2026-08-21 | k=30 gate receipt-level CI `[42.9, 49.4]` | `[42.8, 49.4]` | **This entry was itself wrong when first written, and is corrected here rather than edited away.** The original reason given — "the artifact says 0.4284" — was incomplete: the real cause was that `verify_verdict.py` used the normal quantile for the receipt interval while the authorized reader used t, so the two disagreed by 0.033 F1 points, inside the 1e-3 cross-check tolerance. For a few hours `VERDICT.md` therefore quoted a number its own "check it" command contradicted. The estimator is now t in both readers, `[42.8, 49.4]` is correct, and `tests/test_readers_agree.py` fails if they diverge again. See spec erratum P12, which also records an unrelated in-place edit of the same figure in frozen text on 2026-08-19 that was never logged. |
| 2026-08-21 | Demo-context k=30 quality `0.977–0.993` | `0.936–0.993` | Addendum E banked six more k=30 tenants; the old range was the pre-E three. |
| 2026-08-21 | Offline test count (variously 83, 128, 137, 144, 151 across documents) | `167` | Re-synced to the measured collection count. |
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

## Defects found in our own verification path

| Date | Defect | Fix |
|---|---|---|
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
