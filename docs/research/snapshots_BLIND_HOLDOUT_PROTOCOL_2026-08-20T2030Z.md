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
6. Adaptation constraints: base model at or under 2B parameters,
   adapted offline on only the pairs the challenger supplies; NO
   external model or API calls anywhere in the adaptation or
   prediction path; turnaround within 72 hours of receipt.
7. Deliverables with the predictions: the adapter weights and their
   sha256, the exact repository commit, the exact adaptation command
   and seed — so the predictions are regenerable, not just hashed.
8. One prediction submission only; the challenger may hash the
   submission on receipt. Challengers are encouraged to OTS-stamp
   their held-out gold file before sending documents, protecting
   both sides.
9. Scorer pinning: the default scorer is `score_text_output` at the
   repository commit named in the challenge terms; aggregation is
   mean per-document micro-F1 with invalid JSON scored 0; results
   publish as absolute numbers (no pass/fail spin), beside any
   baseline the challenger runs on the same holdout.
10. Amendments to this protocol after a challenge begins bind only
    future challenges.

A byte-exact dated snapshot of this document is OpenTimestamps-
anchored (snapshots_BLIND_HOLDOUT_PROTOCOL_<date>.md + .ots beside
this file) so the offer's terms cannot be quietly rewritten after a
challenge. If this living file ever differs from the newest anchored
snapshot, the snapshot governs.
