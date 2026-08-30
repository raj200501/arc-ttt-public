#!/usr/bin/env bash
# Everything still owed, cheapest-first so a restart loses the least.
# R attacks our own mechanism claim and costs the least per bit, so it
# runs first. The 3B k-shot arm OOM-killed at 21/30 in float32 (12.4 GB
# of weights plus 4,142-token prompts on a 15 GB box, no traceback --
# the kernel, not Python). It returns in bfloat16 WITH a same-family
# bfloat16 control, so the dtype is measured rather than assumed.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
LOG=work/pending_arms.log
mkdir -p work; : > "$LOG"
say () { echo "=== $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }

cell () {
  local label="$1" out="$2"; shift 2
  if [ -f "$out" ]; then say "$label SKIP (banked)"; return 0; fi
  say "$label"
  python3 scripts/scale_rung_arm.py "$@" --out "$out" >>"$LOG" 2>&1 \
    && say "$label OK" || say "$label FAILED"
}

# --- Addendum R: is it the demonstrations, or the length, or the turns?
cell "R1 k=1"          experiments/waybill_fence_dose_k1_2026-08-25.json \
     --model Qwen/Qwen2.5-0.5B-Instruct --mode kshot --k 1 --dtype float32
cell "R2 k=3"          experiments/waybill_fence_dose_k3_2026-08-25.json \
     --model Qwen/Qwen2.5-0.5B-Instruct --mode kshot --k 3 --dtype float32
cell "R3 k=1+schema"   experiments/waybill_fence_dose_k1schema_2026-08-25.json \
     --model Qwen/Qwen2.5-0.5B-Instruct --mode schema_kshot --k 1 --dtype float32

# --- the impact grid's missing cell, plus its dtype control
cell "CTRL 1.5B kshot bf16" \
     experiments/waybill_scale_rung_1.5b_kshot_bf16_2026-08-25.json \
     --model Qwen/Qwen2.5-1.5B-Instruct --mode kshot --dtype bfloat16
cell "3B kshot bf16" \
     experiments/waybill_scale_rung_3b_kshot_bf16_2026-08-25.json \
     --model Qwen/Qwen2.5-3B-Instruct --mode kshot --dtype bfloat16

# --- Addendum Q last: heaviest, and its own driver handles the dtype
say "handing off to Addendum Q"
bash scripts/run_addendum_q.sh >>"$LOG" 2>&1 && say "Q OK" || say "Q FAILED"
say "ALL DONE"
