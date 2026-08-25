#!/usr/bin/env bash
# Addendum R: is it the demonstrations, or the length, or the turn count?
#
# Three new cells at 0.5B. Waits for every other arm to clear first --
# these are short runs and there is no reason to contend for cores.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
LOG=work/addendum_r.log
mkdir -p work
: > "$LOG"
say () { echo "=== $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }

say "waiting for every other arm"
while pgrep -f "scale_rung_arm|score_adapted_arm|run_challenge" >/dev/null 2>&1; do
  sleep 30
done
say "cores free"

cell () {  # cell <label> <out> <args...>
  local label="$1" out="$2"; shift 2
  if [ -f "$out" ]; then say "$label SKIP (banked)"; return 0; fi
  say "$label"
  python3 scripts/scale_rung_arm.py --model Qwen/Qwen2.5-0.5B-Instruct \
      --dtype float32 "$@" --out "$out" >>"$LOG" 2>&1 \
    && say "$label OK" || say "$label FAILED"
}

cell "R1 k=1"            experiments/waybill_fence_dose_k1_2026-08-25.json \
     --mode kshot --k 1
cell "R2 k=3"            experiments/waybill_fence_dose_k3_2026-08-25.json \
     --mode kshot --k 3
cell "R3 k=1 + schema"   experiments/waybill_fence_dose_k1schema_2026-08-25.json \
     --mode schema_kshot --k 1
say "ALL DONE"
