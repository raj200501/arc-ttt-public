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
| CIs | receipt-level [42.9, 49.4]; cluster-level (3 seeds) [23.1, 69.9] — both clear the bar | same | same |
| Baseline validity (not a floor) | kshot arms 0.4333–0.6208, inside preregistered [0.15, 0.95] window | per-arm `novel_schema_0.5b_k30_seed*_kshot_2026-08-12.json` | same |
| Scoping of +46.5 | adaptation ADDED ON TOP of the same model's 30-shot prompt, at 0.5B | spec B.9.1 | read the spec |
| CORD negative (the scoping result) | FAILED gates at all three scales: −7.3 / −11.5 / −4.5 F1 | `experiments/ge2_*_result_2026-08-11.json` | artifacts |
| k=10 replication | 10/10 tenants, +41.5 pooled, 569W/1L | `experiments/novel_schema_summary` k=10 section + per-arm files | artifacts |
| Frontier context (comparability, self-run) | frontier k-shot 1.00; adapted 0.5B 0.954–0.985 (k=10 arms) | `experiments/novel_frontier_baseline_2026-08-16.json` | artifact |
| Payload asymmetry (document-only endpoint, this corpus) | 20.0x–58.0x | `experiments/novel_payload_asymmetry_2026-08-15.json` | artifact |
| Document-only serving, demo-trained adapter (Addendum D, partial) | seeds 2,3: both 0.0000, 0/60 valid JSON (prose collapse; mechanism reproduced — spec D.6; seed-2 fresh-adapter run rules out checkpoint staleness) | `experiments/novel_schema_d_0.5b_k30_seed{2,3}_doconly_2026-08-18.json` | artifacts |
| Document-only, no adapter (D.5 comparability) | seeds 1,2: 0.0000, 0/60 valid JSON (dz-s3 in flight) | `experiments/novel_schema_d_0.5b_k30_seed{1,2}_doczero_2026-08-18.json` | artifacts |
| **Addendum D verdict** | **FAIL, both reads** (retention −98.4 vs −5 bar; unified −51.9, 0W/155L; all seeds 0.0000; adapter contribution +0.0 over no-adapter) — published per the pre-written D.3 branch; corrective is Addendum F | `experiments/novel_schema_d_*` + spec D.7 | `read_addendum_d.py` |
| **Doc-only-TRAINED adapters (Addendum F)** | **PASS**: +24.0 seed-mean vs prompted baseline (+32.0/+5.5/+34.7; bar +5), CI low +22.3, sign 126W/19L; quality cost vs demo-context arm −22.4 (seed-dependent −4.1..−43.5); adapt ~272s/tenant; all three arms PRIMARY-VERIFIED | `experiments/novel_schema_f_0.5b_k30_seed{1,2,3}_docadapted_2026-08-19.json` + spec F.5 | `verify_from_primary.py` |
| Diverse-geometry gate (Addendum E) | PENDING (6 seeds, bars frozen E.2) | — | spec + OTS |
| ARC Prize track | 1.67 public ×3 (v8/v9/v10); climb formally deprioritized | `experiments/kaggle_v{8,9,10}_scored_*.json` | artifacts |
| Preregistration ordering | spec SHA-256 Bitcoin-anchored (OpenTimestamps), byte-exact snapshots shipped | `docs/research/snapshots_*` + `.ots` | `ots verify` |

Primary-evidence path (artifacts from Addendum E/F onward store raw
predictions): `python3 scripts/verify_from_primary.py <artifact>` —
re-scores every stored prediction against gold labels regenerated from
the deterministic corpus generator.
