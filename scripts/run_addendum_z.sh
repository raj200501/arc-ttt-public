#!/bin/bash
# Addendum Z launcher: start (or re-attach to) the six-arm re-run, detached.
#
#   bash scripts/run_addendum_z.sh          # start if not running; no-op if it is
#   bash scripts/run_addendum_z.sh status   # one line: running / idle, arms banked
#
# The run outlasts any one session: the runner journals every document and
# saves the adapter after training (scripts/novel_schema_rerun.py), so a
# restart resumes. This script refuses to start a second instance — two
# processes on one journal would interleave rows — and never kills one.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$ROOT/work/z"
PIDFILE="$WORK/runner.pid"
LOG="$WORK/run.log"
mkdir -p "$WORK"

running() {
  # a pid alone is not enough: the container reboots when reclaimed and
  # pids restart, so the recorded pid can belong to anything by then
  [ -f "$PIDFILE" ] || return 1
  local pid; pid="$(cat "$PIDFILE")"
  kill -0 "$pid" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "novel_schema_rerun.py --run"
}

banked() {
  ls "$ROOT"/experiments/novel_schema_0.5b_k30_seed*_2026-09-23.json 2>/dev/null | wc -l
}

if [ "${1:-}" = "status" ]; then
  if running; then state="running pid $(cat "$PIDFILE")"; else state="idle"; fi
  echo "addendum z: $state | $(banked)/6 arms banked | $(tail -n 1 "$LOG" 2>/dev/null || echo 'no log')"
  exit 0
fi

if running; then
  echo "already running (pid $(cat "$PIDFILE")); not starting a second instance"
  exit 0
fi
if [ "$(banked)" -eq 6 ]; then
  echo "all six arms banked; nothing to run"
  exit 0
fi

cd "$ROOT"
PYTHONPATH=src nohup setsid python3 scripts/novel_schema_rerun.py --run >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "started pid $(cat "$PIDFILE"), log $LOG"
