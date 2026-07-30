#!/usr/bin/env bash
# Supervisor for long benchmark runs. A Mac sleep (lid close / battery) kills the
# harness's sockets: the process survives but stops making progress forever, and
# a silent stall looks exactly like a slow run. This watches the results jsonl
# and restarts the harness whenever it goes quiet, until every question is done.
#
# Safe to restart at any moment: both harnesses resume per-answer (completed
# answers are skipped, errored ones re-run) AND per-episode (a partial graph is
# topped up, not accepted as complete).
#
# Usage (env is passed through to the harness):
#   PER_TYPE=200 CHUNK_TURNS=2 ... ./evals/run_supervised.sh longmemeval 500
#   N_CONV=10 QA_PER_CONV=999 ... ./evals/run_supervised.sh locomo 1986
set -u
HARNESS="${1:?usage: run_supervised.sh <longmemeval|locomo> <target_answers>}"
TARGET="${2:?missing target answer count}"
STALL_SECS="${STALL_SECS:-900}"    # no new answer for this long => restart
POLL_SECS="${POLL_SECS:-60}"
EVALS="$(cd "$(dirname "$0")" && pwd)"
JSONL="$EVALS/${HARNESS}_results.jsonl"
LOG="$EVALS/${HARNESS}_supervised.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

count_done() {  # unique non-error answers; jsonl is append-only so ids repeat
  python3 - "$JSONL" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
by = {}
if p.exists():
    for line in p.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = r.get('question_id') or (r.get('gid'), r.get('q'))
        by[k] = r          # last write wins: a re-run supersedes an earlier error
print(sum(1 for r in by.values() if not r.get('error')))
PY
}

while true; do
  done_n=$(count_done)
  if [ "$done_n" -ge "$TARGET" ]; then say "COMPLETE: $done_n/$TARGET answered"; exit 0; fi
  say "launching $HARNESS harness ($done_n/$TARGET done)"
  caffeinate -i uv run python "$EVALS/${HARNESS}_harness.py" >>"$EVALS/${HARNESS}_run.log" 2>&1 &
  pid=$!
  last_n=$(count_done); last_change=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_SECS"
    now_n=$(count_done)
    if [ "$now_n" -ne "$last_n" ]; then last_n=$now_n; last_change=$(date +%s); continue; fi
    if [ $(( $(date +%s) - last_change )) -ge "$STALL_SECS" ]; then
      say "STALLED at $now_n answers for ${STALL_SECS}s (likely slept) -- restarting"
      kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
      break
    fi
  done
  wait "$pid" 2>/dev/null
  say "harness exited; $(count_done)/$TARGET answered"
  sleep 5
done
