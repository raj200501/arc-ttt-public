# Blind-holdout protocol (frozen 2026-08-20, before any challenger)

Standing offer to any challenger (fund, researcher, or company):

1. You send 50 documents in any schema. REAL documents with your own
   held-out gold labels are preferred (OCR noise, non-verbatim values,
   ambiguity welcome — no fairness-invariant guarantees requested);
   an invented schema is accepted when your data cannot leave your
   building.
2. You keep the gold labels. I never see them.
3. I adapt a small model on whatever training pairs you provide (or
   on a split you designate) and return predictions for the held-out
   set, with the adapter hash and the exact commit of this repository
   used.
4. You score ONCE, by whatever scorer you declare up front (this
   repo's `score_text_output` offered as a default).
5. The result publishes in this repository either way, beside this
   protocol, with your name or an agreed anonymization.
6. Amendments to this protocol after a challenge begins bind only
   future challenges.

This document's hash is OpenTimestamps-anchored on publication so the
offer's terms cannot be quietly rewritten after a challenge.
