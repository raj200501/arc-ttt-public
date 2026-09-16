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
#   4. arm P;
#   5. arm G, only where the banked scope says it runs;
#   6. the reading, which withholds until all of the above exist.
# Stages 2-5 run family by family rather than stage by stage, and the family
# order is chosen below for what it leaves behind if the box is reclaimed.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
RUN="python3 scripts/cord_decoder_isolation.py"

# Order is scheduling, not protocol: the readings, bars and cells are fixed
# in the frozen protocol and none of them depends on the order arms run in.
# It is chosen so the most informative cells exist earliest, because this box
# is reclaimed on idle and a run that is interrupted for good should leave
# the decisive data behind rather than the cheapest data:
#   qwen2.5-0.5b  the only family with generation modifiers -> the whole of X3,
#                 the contrast that can narrow Addendum V's own sentence
#   falcon3-1b    the only family where the validator ever fired -> the only
#                 place X1 can read anything but INERT
#   granite-2b    the clock-dependent prompt -> its arm C must be re-run
#   smollm2-1.7b  control: V saw byte-identical arms
#   phi3-mini     control, and the slowest by 2x: V saw 32 of 50 bodies differ
FAMILIES="qwen2.5-0.5b falcon3-1b granite-2b smollm2-1.7b phi3-mini"

scope_says () {   # scope_says <family> <key>
  python3 - "$1" "$2" <<'PY'
import json, pathlib, sys
cfg = pathlib.Path("experiments/generation_configs_2026-09-16.json")
sys.exit(0 if json.loads(cfg.read_text())["families"][sys.argv[1]][sys.argv[2]] else 1)
PY
}

$RUN --bank-configs || exit 1

# One family at a time, all of its arms, so a family's contrasts become
# readable together instead of five half-finished families.
for f in $FAMILIES; do
  echo "=== determinism gate: $f"
  $RUN --determinism "$f" || echo "!! gate stage failed for $f (continuing; the reader withholds)"

  if ! scope_says "$f" arm_C_reused_from_V; then
    echo "=== arm C (constrained, re-run under the pinned prompt): $f"
    $RUN --constrained "$f" || echo "!! arm C failed for $f"
  fi

  echo "=== arm P (plain, constraint off): $f"
  $RUN --cell "$f" || echo "!! arm P failed for $f (continuing; the reader withholds)"

  if scope_says "$f" arm_G_runs; then
    echo "=== arm G (generate, defaults neutralised): $f"
    $RUN --generate-neutral "$f" || echo "!! arm G failed for $f"
  else
    echo "=== arm G skipped for $f: no generation modifier (G == G0 by construction)"
  fi
done

echo "=== reading"
$RUN --read
