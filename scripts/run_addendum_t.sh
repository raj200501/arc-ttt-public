#!/usr/bin/env bash
# Addendum T driver: eight cells (four families x two regimes) then the
# reader. Idempotent: banked cells return immediately; the reader
# withholds until all eight exist.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
for fam in smollm2-1.7b granite-2b phi3-mini falcon3-1b; do
  for regime in schema kshot; do
    echo "=== T cell $fam:$regime $(date -u +%FT%TZ)"
    python3 scripts/cord_fence_tax.py --addendum T --cell "$fam:$regime" || echo "=== $fam:$regime FAILED"
  done
done
echo "=== T reader $(date -u +%FT%TZ)"
python3 scripts/cord_fence_tax.py --addendum T --read
echo "=== T DONE $(date -u +%FT%TZ)"
