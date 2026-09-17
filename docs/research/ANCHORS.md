# Anchors — what the `.ots` files beside the protocols prove, and what they do not

An OpenTimestamps file (`<file>.<UTC stamp>.ots`) is a Bitcoin-anchored
proof that the exact bytes of `<file>` existed at or before the block the
attestation lands in. Verify one with `ots verify <file>.<stamp>.ots`
(the attestation upgrades from pending to complete once the calendar's
transaction confirms; `ots upgrade` fetches it).

## What an anchor proves

That the file's bytes existed **by** the anchored time. Nothing more.

## What it does not prove, stated so it is not mistaken for more

- **It does not prove the protocol was frozen before its data.** The
  anchors made on 2026-09-17 cover protocols whose arms ran between
  2026-09-03 and 2026-09-17, so every one of them was anchored **after**
  its run. The freeze order for those addenda rests on commit order in
  the source history, which is private, and on the dated text of the
  protocols themselves — a simulated technical due-diligence reviewer
  (review round 8, 2026-09-17) was right that a stranger cannot check it
  from the public tree, and that is why this page exists rather than a
  sentence claiming otherwise.
- **It does not prove the run happened.** Artifacts are anchored too, so
  a reader can check the reading existed by the same time; whether the
  numbers are right is the reader's own recomputation (`--read`).

## Policy from Addendum Z onward

A protocol is anchored **before its first arm runs**, and the anchor's
stamp is quoted in the protocol's freeze line. An anchor made after the
run is filed here with the run date beside it and never described as a
pre-registration proof.

## The anchors

| file | anchored | first arm ran | proves freeze-before-run? |
|---|---|---|---|
| `ENTERPRISE_EVAL_SPEC.md` | 2026-08-19, 2026-08-20 (three stamps) | — | yes for the gates it froze |
| `snapshots_BLIND_HOLDOUT_PROTOCOL` | 2026-08-20T2030Z | after | yes |
| `ADDENDUM_S_PROTOCOL.md` | 2026-09-17T0750Z | 2026-09-03 | **no** — after the run |
| `ADDENDUM_T_PROTOCOL.md` | 2026-09-17T0750Z | 2026-09-03 | **no** |
| `ADDENDUM_U_PROTOCOL.md`, `ADDENDUM_U_EXT_PROTOCOL.md` | 2026-09-17T0750Z | 2026-09-04 | **no** |
| `ADDENDUM_V_PROTOCOL.md` | 2026-09-17T0750Z | 2026-09-08 | **no** |
| `ADDENDUM_W_PROTOCOL.md` | 2026-09-17T0750Z | 2026-09-08 | **no** |
| `ADDENDUM_X_PROTOCOL.md` | 2026-09-17T0750Z | 2026-09-16 | **no** |
| `ADDENDUM_Y_PROTOCOL.md` | 2026-09-17T0750Z | 2026-09-17 (earlier the same day) | **no** |
| `experiments/cord_decoder_isolation_2026-09-16.json` | 2026-09-17T0750Z | — | the reading existed by then |
| `experiments/phi3_instability_2026-09-17.json` | 2026-09-17T0750Z | — | the reading existed by then |
| `experiments/results_against_thesis_2026-09-17.json` | 2026-09-17T0750Z | — | the ledger existed by then |

The protocols carry dated errata appended after their freeze lines; the
anchored bytes are the files as they stood on 2026-09-17, errata
included, so a later edit to any of them is detectable against the
anchor.
