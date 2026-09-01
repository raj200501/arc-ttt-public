#!/usr/bin/env bash
# The adaptation engineering ladder, rungs E2/E3/E5. Preregistration:
# docs/research/ADAPTATION_ENGINEERING_LADDER.md. Every cell leaves a
# sentinel and is skipped on relaunch, because this environment reclaims
# processes between agent turns and re-running finished training is how
# an afternoon disappears.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
RAW=experiments/blind_rehearsal_2026-08-20_raw
LOG=work/ladder.log
mkdir -p work
say () { echo "=== $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }

cell () {  # cell <name> <outdir> <extra run_challenge args...>
  local name="$1" out="$2"; shift 2
  if [ -f "$out/.complete" ]; then say "$name SKIP (complete)"; return 0; fi
  say "$name"
  if python3 scripts/run_challenge.py \
      --train "$RAW/train.jsonl" --holdout "$RAW/holdout.jsonl" \
      --out-dir "$out" --seed 1 --max-new-tokens 512 --allow-unpinned \
      "$@" >>"$LOG" 2>&1; then
    touch "$out/.complete"; say "$name OK"
  else
    say "$name FAILED"
  fi
}

# E2: the training dial nobody turned. Same 0.5B, same float32 as the
# banked samples=1 arm, only --samples moves.
cell "E2 samples=2" work/e2_s2 --samples 2
cell "E2 samples=4" work/e2_s4 --samples 4

# E3: adapters AND demonstrations, 0.5B. samples=1 to match the banked
# adapted arm; only the serving changes.
cell "E3 0.5B adapted+k20" work/e3_05b --samples 1 --serve-demos

# E5: the same stack at 3B, reusing Addendum Q's durable adapter.
if [ ! -f experiments/ladder_e5_3b_adapted_kshot_2026-08-31.json ]; then
  say "E5 3B adapted+k20 (Q's adapter, bfloat16)"
  python3 scripts/score_adapted_arm.py \
      --model Qwen/Qwen2.5-3B-Instruct --adapter work/arunQ/adapter.pt \
      --dtype bfloat16 --serve-demos --ladder \
      --out experiments/ladder_e5_3b_adapted_kshot_2026-08-31.json \
      >>"$LOG" 2>&1 && say "E5 OK" || say "E5 FAILED"
else
  say "E5 SKIP (banked)"
fi

say "LADDER RUNS DONE — reader next"
python3 scripts/ladder_reader.py >>"$LOG" 2>&1 && say "READER OK" || say "READER FAILED"
say "ALL DONE"
