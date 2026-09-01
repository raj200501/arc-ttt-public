#!/usr/bin/env bash
# Addendum S driver: all four cells in sequence, then the reader.
# Each cell is per-document checkpointed and banks on completion, so
# this driver is idempotent — banked cells return immediately.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
for cell in 0.5b:schema 0.5b:kshot 1.5b:schema 1.5b:kshot; do
  echo "=== S cell $cell $(date -u +%FT%TZ)"
  python3 scripts/cord_fence_tax.py --cell "$cell" || echo "=== $cell FAILED"
done
echo "=== S reader $(date -u +%FT%TZ)"
python3 scripts/cord_fence_tax.py --read
echo "=== S DONE $(date -u +%FT%TZ)"
