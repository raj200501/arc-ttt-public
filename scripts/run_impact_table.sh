#!/usr/bin/env bash
# The impact half of the fence finding: how much does one fence cost, per
# checkpoint and per prompt regime, on the same 30 documents?
#
# Cheapest first so a container restart loses the least. Every arm stores
# RAW model text, which is what makes the fence question askable at all --
# the arms that store parsed objects cannot be re-scored for it.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
LOG=work/impact_table.log
mkdir -p work
: > "$LOG"

run () {
  local label="$1"; shift
  local out="$1"; shift
  if [ -f "$out" ]; then echo "=== ${label} SKIP (banked)" | tee -a "$LOG"; return 0; fi
  echo "=== ${label} $(date -u +%FT%TZ)" | tee -a "$LOG"
  if python3 scripts/scale_rung_arm.py "$@" --out "$out" >>"$LOG" 2>&1; then
    echo "=== ${label} OK" | tee -a "$LOG"
  else
    echo "=== ${label} FAILED" | tee -a "$LOG"
  fi
}

run "0.5B schema" experiments/waybill_scale_rung_0.5b_schema_2026-08-25.json \
    --model Qwen/Qwen2.5-0.5B-Instruct --mode schema --dtype float32
run "0.5B kshot" experiments/waybill_scale_rung_0.5b_kshot_2026-08-25.json \
    --model Qwen/Qwen2.5-0.5B-Instruct --mode kshot --dtype float32
run "1.5B kshot" experiments/waybill_scale_rung_1.5b_kshot_RAW_2026-08-25.json \
    --model Qwen/Qwen2.5-1.5B-Instruct --mode kshot --dtype float32
run "3B kshot" experiments/waybill_scale_rung_3b_kshot_2026-08-25.json \
    --model Qwen/Qwen2.5-3B-Instruct --mode kshot --dtype float32
echo "=== ALL DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
