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
stamp is quoted in the protocol's freeze line. Because the proof commits
to the file's exact bytes, the order is fixed: choose the stamp string,
write it into the freeze line and into this page, commit, and only then
`ots stamp` those bytes — a stamp made before the string was written in
would attest a draft. An anchor made after the run is filed here with
the run date beside it and never described as a pre-registration proof.

## The anchors

The stamp in a file's name is when it was submitted to the calendars.
What the proof attests is the **Bitcoin block** it was folded into —
the time a stranger can check, and the only one that matters. Every
proof below was upgraded to a complete Bitcoin attestation on
2026-09-23 (`ots upgrade`; until then ten of them were calendar
promises only, which a simulated reviewer pointed out). Block times
are the block header timestamps (blockstream.info), earliest
attestation per proof.

| file | stamp | attested in (block, time UTC) | first arm ran | proves freeze-before-run? |
|---|---|---|---|---|
| `ENTERPRISE_EVAL_SPEC.md` | 2026-08-19T0119Z, 2026-08-19T0330Z, 2026-08-20T1330Z | 963117 (08-19 02:22), 963127 (08-19 04:23), 963319 (08-20 14:25) | — | yes for the gates it froze |
| `snapshots_BLIND_HOLDOUT_PROTOCOL` | 2026-08-20T2030Z | 963325 (08-20 15:39) | after | yes |
| `ADDENDUM_S_PROTOCOL.md` | 2026-09-17T0750Z | 967383 (09-17 08:02) | 2026-09-03 | **no** — after the run |
| `ADDENDUM_T_PROTOCOL.md` | 2026-09-17T0750Z | 967383 (09-17 08:02) | 2026-09-03 | **no** |
| `ADDENDUM_U_PROTOCOL.md`, `ADDENDUM_U_EXT_PROTOCOL.md` | 2026-09-17T0750Z | 967383 (09-17 08:02) | 2026-09-04 | **no** |
| `ADDENDUM_V_PROTOCOL.md` | 2026-09-17T0750Z | 967383 (09-17 08:02) | 2026-09-08 | **no** |
| `ADDENDUM_W_PROTOCOL.md` | 2026-09-17T0750Z | 967383 (09-17 08:02) | 2026-09-08 | **no** |
| `ADDENDUM_X_PROTOCOL.md` | 2026-09-17T0750Z | 967383 (09-17 08:02) | 2026-09-16 | **no** |
| `ADDENDUM_Y_PROTOCOL.md` | 2026-09-17T0750Z | 967383 (09-17 08:02) | 2026-09-17 (earlier the same day) | **no** |
| `ADDENDUM_Z_PROTOCOL.md` | 2026-09-23T0219Z | **968216 (2026-09-23 02:36:37)** | first process 02:19:54 (`started_utc`), killed by a container restart ~03:04 before saving anything; restarted 03:07; **no document had been decoded at the attested time** | **yes** — the attested time precedes every decoded output of the run; the runner's start clock is ours to report, the block time is not |
| `experiments/cord_decoder_isolation_2026-09-16.json` | 2026-09-17T0750Z | 967383 (09-17 08:02) | — | the reading existed by then |
| `experiments/phi3_instability_2026-09-17.json` | 2026-09-17T0750Z | 967383 (09-17 08:02) | — | the reading existed by then |
| `experiments/results_against_thesis_2026-09-17.json` | 2026-09-17T0750Z | 967383 (09-17 08:02) | — | the ledger existed by then |

Note on the 2026-08-20 blind-holdout row: the snapshot is *stamped*
20:30Z in its name but *attested* at 15:39Z, earlier than the name
says. Why the name says 20:30Z is not recorded. The block time bounds
the bytes from above; the earlier bound is the stronger claim, and it
is the block's, not ours.

The protocols carry dated errata appended after their freeze lines; the
anchored bytes are the files as they stood at their stamp, errata to
that date included, so a later edit to any of them is detectable against
the anchor.
