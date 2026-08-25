#!/usr/bin/env bash
# One-command demo: start the adaptation endpoint, run the before/after
# narrative against it, shut the endpoint down.
#
#   bash demo/run_endpoint_demo.sh
#
# The default corpus is the NOVEL-SCHEMA demo (demo/novel_demo.jsonl) —
# the regime the preregistered gates say adaptation works in. It used to
# default to CORD, which is the regime our own Addendum A says the recipe
# FAILS in at all three scales: the one-command experience showcased the
# documented loss. CORD is still one env var away, deliberately, because
# the negative is part of the evidence:
#
#   DATA=demo/cord_validation.jsonl bash demo/run_endpoint_demo.sh
#
# Cost of this command, stated up front: it downloads ~950MB of weights on
# first run and takes roughly 6-10 minutes on CPU. The verification scripts
# in scripts/ need none of that and no PyTorch.
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
DATA="${DATA:-$ROOT/demo/novel_demo.jsonl}"
K="${K:-10}"
EVAL_N="${EVAL_N:-2}"
SEED="${SEED:-0}"

# Fail fast on the declared floor rather than deep inside transformers.
# A cold clone on a below-floor interpreter used to get as far as loading
# the tokenizer and then die with an opaque
# "TypeError: __init__() got an unexpected keyword argument 'dtype'",
# while the client half kept polling a server that was already dead. An
# outside simulation of this repo hit exactly that and could not tell a
# stale environment from a broken product. pyproject.toml is the single
# source of the floors below.
"$PYTHON" - <<'PREFLIGHT' || exit 1
import sys

problems = []
if sys.version_info < (3, 11):
    problems.append(
        f"Python {sys.version_info.major}.{sys.version_info.minor} — "
        "pyproject.toml requires >=3.11")
try:
    import transformers
    major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    if (major, minor) < (4, 54):
        problems.append(
            f"transformers {transformers.__version__} — pyproject.toml "
            "requires >=4.54,<5.18")
except ImportError:
    problems.append("transformers is not installed")
try:
    import torch  # noqa: F401
except ImportError:
    problems.append(
        "torch is not installed (the demo needs it; the verification "
        "scripts in scripts/ do NOT)")

if problems:
    print("\n[demo] REFUSING TO START — this environment is below the "
          "floors this repo declares:", file=sys.stderr)
    for problem in problems:
        print(f"  * {problem}", file=sys.stderr)
    print("\n[demo] Fix with:\n"
          "    python3.11 -m venv .venv && . .venv/bin/activate\n"
          "    pip install -e '.[demo]'\n"
          "    bash demo/run_endpoint_demo.sh\n\n"
          "[demo] This is an environment problem, not a failing product — "
          "and nothing here is needed to CHECK the evidence: "
          "scripts/verify_verdict.py is stdlib-only.", file=sys.stderr)
    raise SystemExit(1)
PREFLIGHT

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
