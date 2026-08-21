#!/bin/bash
# Pre-push gate for the public repo working copy. Run before EVERY push
# to arc-ttt-public. Exists because the 08-19 round-3 cold-read audit
# found the public tree shipping a stale src/arcttt/novel_schema.py —
# which made verify_from_primary.py (the email's second command) crash
# with TypeError for anyone who ran it. Incremental public commits must
# never drift from the source tree on the paths a cold reader executes.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
PUB="${1:?usage: check_public_sync.sh <public-working-copy>}"

fail=0

# 1) library + scripts byte-identical.
#    The exclusions must stay in lockstep with the same list in
#    export_public.sh — test_render_send.py exercises the outbound-email
#    renderer and necessarily names real targets, so the leak gate
#    rejects it and it is deliberately private. Anything private-only is
#    named here so a DELIBERATE omission cannot be confused with drift,
#    and so drift cannot hide behind a deliberate omission.
for d in src scripts tests; do
  if ! diff -rq "$SRC/$d" "$PUB/$d" \
      --exclude=__pycache__ --exclude=export_public.sh --exclude=fill_gate_slots.py \
      --exclude=check_public_sync.sh --exclude=test_render_send.py; then
    fail=1
  fi
done

# 1b) the private-only list is a hole in gate 1, so verify each entry is
#     private-only ON PURPOSE: still present in the source tree, and
#     actually excluded by export_public.sh. An entry that silently
#     disappeared from the source would otherwise never be noticed.
for private_only in test_render_send.py export_public.sh fill_gate_slots.py; do
  if ! find "$SRC/src" "$SRC/scripts" "$SRC/tests" -name "$private_only" \
      | grep -q .; then
    echo "EXCLUDED-BUT-MISSING from the source tree: $private_only" >&2
    fail=1
  fi
  # Match the stem as a FIXED string: export_public.sh writes the names
  # as regexes with escaped dots ("fill_gate_slots\.py"), so a pattern
  # built from the plain filename silently misses them.
  if ! grep -qF "${private_only%.py}" "$SRC/scripts/export_public.sh"; then
    echo "excluded here but NOT by export_public.sh: $private_only" >&2
    fail=1
  fi
done

# 1c) the documented test count must match the tree it is documented in.
#     The public tree has fewer tests (see 1); its docs are rewritten by
#     export_public.sh, and this re-checks that the rewrite happened —
#     a stale count is what two outside readers caught by running
#     `pytest -q`, which is the one command this pitch invites.
pub_tests=$(cd "$PUB" && python3 -m pytest tests/ -q --collect-only \
  -p no:cacheprovider 2>/dev/null | grep -oE '[0-9]+ tests? collected' \
  | grep -oE '^[0-9]+' || true)
if [ -n "$pub_tests" ]; then
  for doc in README.md EVIDENCE.md ROADMAP.md paper/DRAFT.md; do
    [ -f "$PUB/$doc" ] || continue
    if grep -oE '[0-9]+ offline tests' "$PUB/$doc" \
        | grep -qvE "^${pub_tests} offline tests$"; then
      echo "PUBLIC $doc claims a test count this tree does not have" >&2
      echo "  (tree collects ${pub_tests})" >&2
      fail=1
    fi
  done
else
  echo "WARNING: could not collect the public test count" >&2
fi

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
