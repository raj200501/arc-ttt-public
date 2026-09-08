#!/usr/bin/env bash
# Addendum V driver: five constrained schema-only cells then the reader.
# Idempotent: banked cells return immediately; the reader withholds until
# all five exist. Order puts the already-cached checkpoints first so the
# SmolLM2 / Granite re-downloads can finish in the background.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
for fam in qwen2.5-0.5b falcon3-1b phi3-mini smollm2-1.7b granite-2b; do
  echo "=== V cell $fam $(date -u +%FT%TZ)"
  python3 scripts/cord_constrained_families.py --cell "$fam" || echo "=== $fam FAILED"
done
echo "=== V reader $(date -u +%FT%TZ)"
python3 scripts/cord_constrained_families.py --read
echo "=== V DONE $(date -u +%FT%TZ)"
