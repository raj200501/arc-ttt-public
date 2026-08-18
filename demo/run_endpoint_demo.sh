#!/usr/bin/env bash
# One-command demo: start the adaptation endpoint, run the before/after
# CORD-receipt narrative against it, shut the endpoint down.
#
#   bash demo/run_endpoint_demo.sh
#
# Capture a transcript with:
#   bash demo/run_endpoint_demo.sh 2>&1 | tee demo/endpoint_demo_transcript.txt
#
# Overrides via environment: PYTHON, PORT, MODEL, DATA, K, EVAL_N, SEED.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
PORT="${PORT:-8341}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DATA="${DATA:-$ROOT/demo/cord_validation.jsonl}"
K="${K:-10}"
EVAL_N="${EVAL_N:-2}"
SEED="${SEED:-0}"

if [ ! -f "$DATA" ]; then
    echo "missing $DATA — fetch it with:" >&2
    echo "  $PYTHON $ROOT/scripts/fetch_cord.py --split validation --limit 100 --out $DATA" >&2
    exit 1
fi

"$PYTHON" "$ROOT/demo/endpoint_demo.py" serve --model "$MODEL" --port "$PORT" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

"$PYTHON" "$ROOT/demo/endpoint_demo.py" demo \
    --data "$DATA" --port "$PORT" --k "$K" --eval-n "$EVAL_N" --seed "$SEED"

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
echo "[demo] endpoint stopped."
