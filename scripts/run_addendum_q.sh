#!/usr/bin/env bash
# Addendum Q: does adaptation still add anything at 3B?
#
# Waits for the impact-table chain to finish first. A 3B training run and a
# 3B inference run cannot share 15 GB, and a timing-sensitive measurement
# must never share cores -- both rules are in AGENTS.md and both have cost
# this project a correction already.
#
# The dtype contingency is the one frozen in VERDICT.md before any number
# existed: float32 first; if it will not train, bfloat16 with gradient
# checkpointing PLUS a bfloat16 prompted-3B control, so a bf16 adapted arm
# is never read against an fp32 bar.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
LOG=work/addendum_q.log
mkdir -p work
: > "$LOG"

say () { echo "=== $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }

# ---- wait for the cores and the memory to be free --------------------
say "waiting for the impact-table chain"
while pgrep -f scale_rung_arm >/dev/null 2>&1; do sleep 30; done
say "cores free"

RAW=experiments/blind_rehearsal_2026-08-20_raw
# work/, not /tmp: the environment reclaims processes AND /tmp between
# agent turns, and one completed 20-minute bf16 adaptation was lost that
# way -- the log said "Q1 bfloat16 adaptation OK" while its output
# directory no longer existed. A completed adaptation now leaves a
# sentinel recording its dtype and is reused instead of redone; the
# float32-first contingency order was exercised and its OOM recorded
# (twice, in this log's history), so a relaunch that finds a completed
# bf16 adaptation does not re-attempt float32 -- the contingency is a
# rule about method, not a ritual to re-perform per process.
ADAPT_DIR=work/arunQ

adapt () {  # adapt <dtype> <outdir>
  local dtype="$1" out="$2"
  if [ -f "$out/.complete" ] && [ "$(cat "$out/.complete")" = "$dtype" ]; then
    say "adaptation already complete in $dtype; reusing $out"
    return 0
  fi
  rm -rf "$out"
  python3 scripts/run_challenge.py \
      --train "$RAW/train.jsonl" --holdout "$RAW/holdout.jsonl" \
      --out-dir "$out" --model Qwen/Qwen2.5-3B-Instruct \
      --dtype "$dtype" --grad-checkpointing \
      --samples 1 --max-new-tokens 512 --allow-unpinned >>"$LOG" 2>&1 \
    && echo "$dtype" > "$out/.complete"
}

if [ -f "$ADAPT_DIR/.complete" ] && [ "$(cat "$ADAPT_DIR/.complete")" = "bfloat16" ]; then
  say "Q1 found a completed bfloat16 adaptation; skipping float32 re-attempt"
  DTYPE=bfloat16
  say "CONTROL prompted 3B schema in bfloat16"
  python3 scripts/scale_rung_arm.py \
      --model Qwen/Qwen2.5-3B-Instruct --mode schema --dtype bfloat16 \
      --out experiments/waybill_scale_rung_3b_schema_bf16_control_2026-08-25.json \
      >>"$LOG" 2>&1 && say "CONTROL OK" || say "CONTROL FAILED"
elif say "Q1 adapting the 3B in float32" && adapt float32 "$ADAPT_DIR"; then
  say "Q1 float32 adaptation OK"
  DTYPE=float32
else
  say "float32 adaptation did not run; taking the FROZEN contingency: bfloat16"
  if adapt bfloat16 "$ADAPT_DIR"; then
    say "Q1 bfloat16 adaptation OK"
    DTYPE=bfloat16
    # The frozen contingency requires a within-dtype control before any
    # bf16 arm is read against a bar measured in fp32.
    say "CONTROL prompted 3B schema in bfloat16"
    python3 scripts/scale_rung_arm.py \
        --model Qwen/Qwen2.5-3B-Instruct --mode schema --dtype bfloat16 \
        --out experiments/waybill_scale_rung_3b_schema_bf16_control_2026-08-25.json \
        >>"$LOG" 2>&1 && say "CONTROL OK" || say "CONTROL FAILED"
  else
    say "Q1 FAILED in both dtypes -- reporting the failure, not a number"
    exit 1
  fi
fi

say "Q1 scoring the adapted 3B on the 30 held-out documents (dtype=$DTYPE)"
python3 scripts/score_adapted_arm.py \
    --model Qwen/Qwen2.5-3B-Instruct --adapter "$ADAPT_DIR/adapter.pt" \
    --dtype "$DTYPE" \
    --out experiments/waybill_adapted_3b_2026-08-25.json >>"$LOG" 2>&1 \
  && say "Q1 SCORED OK" || say "Q1 SCORING FAILED"

say "ALL DONE"
