# VERDICT.md — every headline number, reconciled

Rule: a number appears here only if it is banked in `experiments/` with
its artifact named. Pending experiments appear as PENDING with their
frozen bars, never with projected numbers. If any number here disagrees
with its artifact, that is a bug we pay attention to first — open an
issue.

| Claim | Number | Artifact | Check it |
|---|---|---|---|
| Novel-schema k=30 gate (Addendum B) | GO: +46.5 mean F1 over seeds {1,2,3} (+36.0/+49.0/+54.4) vs +5 bar | `experiments/novel_schema_summary_2026-08-12.json` | `python3 scripts/verify_verdict.py` |
| Sign test | 156W / 0L / 2T over 158 scored paired receipts of 180 designed | same | same |
| CIs | receipt-level [42.8, 49.4]; cluster-level (3 seeds) [23.1, 69.9] — both clear the bar | same | same |
| Baseline validity (not a floor) | kshot arms 0.4333–0.6208, inside preregistered [0.15, 0.95] window | per-arm `novel_schema_0.5b_k30_seed*_kshot_2026-08-12.json` | same |
| Scoping of +46.5 | adaptation ADDED ON TOP of the same model's 30-shot prompt, at 0.5B | spec B.9.1 | read the spec |
| CORD negative (the scoping result) | FAILED gates at all three scales: −7.3 / −11.5 / −4.5 F1 | `experiments/ge2_*_result_2026-08-11.json` | artifacts |
| k=10 replication | 10/10 tenants, +41.5 pooled, 569W/1L | `experiments/novel_schema_summary` k=10 section + per-arm files | artifacts |
| Frontier context (comparability, self-run) | frontier k-shot 1.00; adapted 0.5B 0.954–0.985 (k=10 arms) | `experiments/novel_frontier_baseline_2026-08-16.json` | artifact |
| Payload asymmetry (document-only endpoint, this corpus) | 20.0x–58.0x | `experiments/novel_payload_asymmetry_2026-08-15.json` | artifact |
| Document-only serving, demo-trained adapter (Addendum D, partial) | seed-3: 0.0000, 0/60 valid JSON (prose collapse; mechanism reproduced and logged) | `experiments/novel_schema_d_0.5b_k30_seed3_doconly_2026-08-18.json`, spec D.6 | artifact |
| Document-only, no adapter (D.5 comparability) | seeds 1,2: 0.0000, 0/60 valid JSON | `experiments/novel_schema_d_0.5b_k30_seed{1,2}_doczero_2026-08-18.json` | artifacts |
| Addendum D verdict | PENDING (seeds 1,2 doconly in flight; bars frozen in D.2) | — | spec |
| Doc-only-TRAINED adapters (Addendum F) | PENDING (bars frozen F.1, 2026-08-19T03:30Z, before any run) | — | spec + OTS |
| Diverse-geometry gate (Addendum E) | PENDING (6 seeds, bars frozen E.2) | — | spec + OTS |
| ARC Prize track | 1.67 public ×3 (v8/v9/v10); climb formally deprioritized | `experiments/kaggle_v{8,9,10}_scored_*.json` | artifacts |
| Preregistration ordering | spec SHA-256 Bitcoin-anchored (OpenTimestamps), byte-exact snapshots shipped | `docs/research/snapshots_*` + `.ots` | `ots verify` |

Primary-evidence path (artifacts from Addendum E/F onward store raw
predictions): `python3 scripts/verify_from_primary.py <artifact>` —
re-scores every stored prediction against gold labels regenerated from
the deterministic corpus generator.
