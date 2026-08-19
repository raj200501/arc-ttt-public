#!/bin/bash
# Pre-push gate for the public repo working copy. Run before EVERY push
# to arc-ttt-public. Exists because the 08-19 round-3 cold-read audit
# found the public tree shipping a stale src/arcttt/novel_schema.py —
# which made verify_from_primary.py (the email's second command) crash
# with TypeError for anyone who ran it. Incremental public commits must
# never drift from the source tree on the paths investors execute.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
PUB="${1:?usage: check_public_sync.sh <public-working-copy>}"

fail=0

# 1) library + scripts byte-identical (export_public.sh is private-only)
for d in src scripts tests; do
  if ! diff -rq "$SRC/$d" "$PUB/$d" \
      --exclude=__pycache__ --exclude=export_public.sh \
      --exclude=check_public_sync.sh; then
    fail=1
  fi
done

# 2) no experiment artifact present privately but missing publicly.
#    *.journal.jsonl are resumable run-state, not artifacts — their
#    rows are embedded in the finished artifact's per_doc field.
missing=$(comm -23 <(ls "$SRC/experiments" | grep -v '\.journal\.jsonl$') \
                   <(ls "$PUB/experiments"))
if [ -n "$missing" ]; then
  echo "MISSING public artifacts:" >&2
  echo "$missing" >&2
  fail=1
fi

# 3) no tracked public file absent from the source tree — stale
#    leftovers drift silently (this caught a stale root
#    ENTERPRISE_EVAL_SPEC.md missing E-r1/E-r2, and an undated .ots
#    proof that fails ots verify against the current spec)
stray=$(cd "$PUB" && git ls-files | while read -r f; do
  [ -e "$SRC/$f" ] || echo "$f"
done)
if [ -n "$stray" ]; then
  echo "STALE public-only files (not in source tree):" >&2
  echo "$stray" >&2
  fail=1
fi

# 4) the three public verify paths must actually run clean in the
#    public tree — exactly what a cold reader executes
( cd "$PUB" \
  && python3 scripts/verify_verdict.py >/dev/null \
  && python3 scripts/read_addendum_d.py >/dev/null \
  && PYTHONPATH=src python3 scripts/verify_from_primary.py \
       experiments/novel_schema_f_*.json >/dev/null ) || fail=1

if [ "$fail" -ne 0 ]; then
  echo "PUBLIC SYNC GATE FAILED — do not push." >&2
  exit 1
fi
echo "public sync gate: CLEAN — safe to push"
