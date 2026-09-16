#!/bin/bash
# Addendum X driver -- the decoder isolated. Idempotent per cell: every
# stage skips if its artifact exists, and every cell checkpoints per
# document, so a reboot resumes instead of restarting.
#
#   bash scripts/run_addendum_x.sh
#
# Order matters and is the protocol's, not a convenience:
#   1. the scope record, banked BEFORE any arm, so which arms run on which
#      family is decided by the checkpoints and not by the results;
#   2. the determinism gate on the arms that are REUSED (constrained and
#      the generate comparator) -- cheap, and if it fails nothing else is
#      worth running;
#   3. arm C, re-run only for a family whose chat template reads the clock
#      and whose banked cells therefore carry a prompt we cannot reproduce;
#   4. arm P, smallest family first so a broken flag shows up in minutes;
#   5. arm G, only where the banked scope says it runs;
#   6. the reading, which withholds until all of the above exist.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
RUN="python3 scripts/cord_decoder_isolation.py"

# smallest/fastest first: a broken flag surfaces in 25 minutes, not 5 hours
FAMILIES="qwen2.5-0.5b smollm2-1.7b falcon3-1b granite-2b phi3-mini"

scope_says () {   # scope_says <family> <key>
  python3 - "$1" "$2" <<'PY'
import json, pathlib, sys
cfg = pathlib.Path("experiments/generation_configs_2026-09-16.json")
sys.exit(0 if json.loads(cfg.read_text())["families"][sys.argv[1]][sys.argv[2]] else 1)
PY
}

$RUN --bank-configs || exit 1

for f in $FAMILIES; do
  echo "=== determinism gate: $f"
  $RUN --determinism "$f" || echo "!! gate stage failed for $f (continuing; the reader withholds)"
done

for f in $FAMILIES; do
  if ! scope_says "$f" arm_C_reused_from_V; then
    echo "=== arm C (constrained, re-run under the pinned prompt): $f"
    $RUN --constrained "$f" || echo "!! arm C failed for $f"
  fi
done

for f in $FAMILIES; do
  echo "=== arm P (plain, constraint off): $f"
  $RUN --cell "$f" || echo "!! arm P failed for $f (continuing; the reader withholds)"
done

for f in $FAMILIES; do
  if scope_says "$f" arm_G_runs; then
    echo "=== arm G (generate, defaults neutralised): $f"
    $RUN --generate-neutral "$f" || echo "!! arm G failed for $f"
  else
    echo "=== arm G skipped for $f: no generation modifier (G == G0 by construction)"
  fi
done

echo "=== reading"
$RUN --read
