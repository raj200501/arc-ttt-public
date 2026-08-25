#!/usr/bin/env bash
# Addendum O, in the order that finds out fastest whether anything survives.
#
# O1 (the 3B given only a field list) is the decisive cell and its prompts
# are ~190 tokens, so it is also the cheapest. It runs first. The
# reproduction check on the banked 1.5B arms runs second -- it is a result
# in its own right and must not be allowed to delay the decisive one. O2
# (the 3B with 20 demonstrations) is ~4,300-token prompts and runs last.
#
# The dtype contingency is the one frozen in VERDICT.md before any number
# existed: float32 first, bfloat16 only if float32 will not load, and a
# bfloat16 1.5B control banked before any bfloat16 result is read against
# a float32 bar.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
LOG=work/addendum_o.log
mkdir -p work
: > "$LOG"

run () {  # run <label> <outfile> <args...>
  local label="$1"; shift
  local out="$1"; shift
  echo "=== ${label} $(date -u +%FT%TZ)" | tee -a "$LOG"
  if python3 scripts/scale_rung_arm.py "$@" --out "$out" >>"$LOG" 2>&1; then
    echo "=== ${label} OK" | tee -a "$LOG"
    return 0
  fi
  echo "=== ${label} FAILED (exit $?)" | tee -a "$LOG"
  return 1
}

# ---- O1: the decisive cell -------------------------------------------
if ! run "O1 3B schema fp32" experiments/waybill_scale_rung_3b_schema_2026-08-25.json \
      --model Qwen/Qwen2.5-3B-Instruct --mode schema --dtype float32; then
  echo "float32 3B did not run; taking the FROZEN contingency: bfloat16 + control" \
    | tee -a "$LOG"
  run "O0 1.5B schema bf16 CONTROL" \
      experiments/waybill_scale_rung_1.5b_schema_bf16_control_2026-08-25.json \
      --model Qwen/Qwen2.5-1.5B-Instruct --mode schema --dtype bfloat16
  run "O1 3B schema bf16" \
      experiments/waybill_scale_rung_3b_schema_bf16_2026-08-25.json \
      --model Qwen/Qwen2.5-3B-Instruct --mode schema --dtype bfloat16
fi

# ---- reproduction check on the banked 1.5B arms ----------------------
run "REPRO 1.5B schema fp32 (vs Addendum N)" \
    experiments/waybill_scale_rung_1.5b_schema_REPRO_2026-08-25.json \
    --model Qwen/Qwen2.5-1.5B-Instruct --mode schema --dtype float32

# ---- O2: the 3B with demonstrations ----------------------------------
if ! run "O2 3B kshot fp32" experiments/waybill_scale_rung_3b_kshot_2026-08-25.json \
      --model Qwen/Qwen2.5-3B-Instruct --mode kshot --dtype float32; then
  run "O2 3B kshot bf16" \
      experiments/waybill_scale_rung_3b_kshot_bf16_2026-08-25.json \
      --model Qwen/Qwen2.5-3B-Instruct --mode kshot --dtype bfloat16
fi

# ---- Priority 2: the unfair comparison we currently hold -------------
# Addendum M priced the 1.5B at batch 1 and us at batch 16. Batch the
# 1.5B the same way we batched ourselves and publish the fair ratio.
run "FAIR 1.5B kshot batch4" \
    experiments/waybill_scale_rung_1.5b_kshot_batch4_2026-08-25.json \
    --model Qwen/Qwen2.5-1.5B-Instruct --mode kshot --dtype float32 --batch 4

echo "=== ALL DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
