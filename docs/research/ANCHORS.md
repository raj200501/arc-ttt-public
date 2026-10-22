# Anchors — what the `.ots` files beside the protocols prove, and what they do not

An OpenTimestamps file (`<file>.<UTC stamp>.ots`) is a Bitcoin-anchored
proof that the exact bytes of `<file>` existed at or before the block the
attestation lands in. Verify one with `ots verify <file>.<stamp>.ots`
(the attestation upgrades from pending to complete once the calendar's
transaction confirms; `ots upgrade` fetches it). `ots verify` needs a
local Bitcoin node and exits 1 without one. `ots --no-bitcoin verify
<file>.<stamp>.ots` prints each block height and the merkle root to
compare by hand on any block explorer.

*(Corrected 2026-10-22.)* `ots` looks for the stamped file under the
proof's name minus `.ots`, so with the `<file>.<stamp>.ots` naming used
here it must be told which bytes the proof covers. For a protocol that
has a snapshot, the anchored bytes are the snapshot's:
`ots --no-bitcoin verify -f snapshots_<STEM>_<stamp>.md <file>.<stamp>.ots`,
run in `docs/research/`; for one without, `-f <file>`.

## What an anchor proves

That the file's bytes existed **by** the anchored time. Nothing more.

## What it does not prove, stated so it is not mistaken for more

- **It does not prove the protocol was frozen before its data.** The
  anchors made on 2026-09-17 cover protocols whose arms ran between
  2026-09-01 and 2026-09-17, so every one of them was anchored **after**
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

The stamp in a file's name is the operator's label, not a checked
time. Three proofs were committed before the time their names give
(0330Z at 03:18:59Z, 1330Z at 13:16:06Z, 2030Z at 15:19:37Z). What the
proof attests is the **Bitcoin block** it was folded into — the time a
stranger can check, and the only one that matters. Fourteen of the
sixteen proofs were upgraded to a complete Bitcoin attestation on
2026-09-23 (`ots upgrade`). Until then they were calendar promises
only, which a simulated reviewer pointed out. The two 2026-08-19 spec
proofs were already complete (upgraded 2026-08-19, commit `265a51b`).
Block times are the block header timestamps (blockstream.info),
earliest attestation per proof.

| file | stamp | attested in (block, time UTC) | first arm ran | proves freeze-before-run? |
|---|---|---|---|---|
| `ENTERPRISE_EVAL_SPEC.md` | 2026-08-19T0119Z, 2026-08-19T0330Z, 2026-08-20T1330Z | 963117 (08-19 02:22), 963127 (08-19 04:23), 963319 (08-20 14:25) | — | yes for the gates it froze |
| `snapshots_BLIND_HOLDOUT_PROTOCOL` | 2026-08-20T2030Z | 963325 (08-20 15:39) | after | yes |
| `ADDENDUM_S_PROTOCOL.md` | 2026-09-17T0750Z | 967383 (09-17 08:02) | 2026-09-01 | **no** — after the run |
| `ADDENDUM_T_PROTOCOL.md` | 2026-09-17T0750Z | 967383 (09-17 08:02) | 2026-09-03 | **no** |
| `ADDENDUM_U_PROTOCOL.md`, `ADDENDUM_U_EXT_PROTOCOL.md` | 2026-09-17T0750Z | 967383 (09-17 08:02) | 2026-09-04 | **no** |
| `ADDENDUM_V_PROTOCOL.md` | 2026-09-17T0750Z | 967383 (09-17 08:02) | 2026-09-08 | **no** |
| `ADDENDUM_W_PROTOCOL.md` | 2026-09-17T0750Z | 967383 (09-17 08:02) | 2026-09-08 | **no** |
| `ADDENDUM_X_PROTOCOL.md` | 2026-09-17T0750Z | 967383 (09-17 08:02) | 2026-09-16 | **no** |
| `ADDENDUM_Y_PROTOCOL.md` | 2026-09-17T0750Z | 967383 (09-17 08:02) | 2026-09-17 (earlier the same day) | **no** |
| `ADDENDUM_Z_PROTOCOL.md` | 2026-09-23T0219Z | **968216 (2026-09-23 02:36:37)** | first process 02:19:54 (`started_utc`), killed by a container restart ~03:04 (a time given only in the `534e7fd` commit message) before saving anything; restarted once by 04:18Z (run log: restarts 1); **no document of the run had been decoded at the attested time**. One document (seed 1, index 0, k-shot, greedy plus one sample) was decoded on this box before the freeze to price the run, and its score was seen before the predictions were written (protocol, cost paragraph; prose only, not banked) | **yes** — the attested time precedes every decoded output of the run; the runner's start clock is ours to report, the block time is not |
| `ADDENDUM_AA_PROTOCOL.md` | 2026-10-22T0114Z | **972551 (2026-10-22 01:33:25)**; the other calendars' paths end in 972552 and 972566 | the reader runs only after the Bitcoin attestation; it first ran at 2026-10-22T05:57:19Z, computed the readings in memory and refused, writing and printing nothing (protocol erratum 6), then banked its reading once at 05:58:20Z, 4 h 25 min after block 972551; the reading's start clock, the proof's sha256 and the block heights are banked in `experiments/cord_decoder_isolation_read_2026-10-22.json` | **yes** — before the reader ran on any cell. Not before the data existed or before any of it was seen: the cells existed since 2026-09-16, and X1 was computed by hand during X's run. The record is silent on X2 and X3, so all three count as possibly seen, and every choice in AA was made after that (the protocol says so) |
| `ADDENDUM_AD_PROTOCOL.md` | 2026-10-22T1655Z | **972636 (2026-10-22 17:19:14)** | every stage (fetch, score, read) runs only after the committed proof parses with a Bitcoin attestation and block height, and after `ots verify` against a node is committed; fetch, score and read ran once each on 2026-10-22, the read at 19:21:31Z, 2 h 2 min after block 972636; each banks its start clock, the anchor check, the block heights and the git HEAD in its artifact | **yes**: before any stored `tablejoin` answer was fetched or scored and before any published `tablejoin` value was read; one model's fence tags, every table's model names and the scorer code were seen first, and the table bytes and that model's answers sat on disk in a scratch directory before the freeze (the protocol says so and lists them in `experiments/ad_preexposure_2026-10-22.json`) |
| `experiments/cord_decoder_isolation_2026-09-16.json` | 2026-09-17T0750Z | 967383 (09-17 08:02) | — | the reading existed by then |
| `experiments/phi3_instability_2026-09-17.json` | 2026-09-17T0750Z | 967383 (09-17 08:02) | — | the reading existed by then |
| `experiments/results_against_thesis_2026-09-17.json` | 2026-09-17T0750Z | 967383 (09-17 08:02) | — | the ledger existed by then |
| `experiments/cord_decoder_isolation_read_2026-10-22.json` | none in the name; the proof was committed with the reading at 2026-10-22T05:59:04Z (`068e9f2`) and upgraded in `475fa4d` | 972585 (10-22 07:32:36); the other calendars' paths end in 972587, 972602 and 972607 | — | the reading existed by then |

Note on the 2026-08-20 blind-holdout row: the snapshot is *stamped*
20:30Z in its name but *attested* at 15:39Z, earlier than the name
says. Why the name says 20:30Z is not recorded. The block time bounds
the bytes from above; the earlier bound is the stronger claim, and it
is the block's, not ours. The commit that added the proof is dated
2026-08-20T15:19:37+00:00, so the stamp was sent to the calendars before
then; the name is not a UTC submission time.

The protocols carry dated errata appended after their freeze lines; the
anchored bytes are the files as they stood at their stamp, errata to
that date included, so a later edit to any of them is detectable against
the anchor.
