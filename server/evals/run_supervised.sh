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
set -m   # job control: each background job leads its OWN process group, so the
         # stall kill below can take down the whole `caffeinate -> uv -> python`
         # tree. Killing just the launched pid orphans the real python worker
         # (it reparents to init and keeps running) -- that once left TWO
         # harnesses racing on the same graphs, double-spending and writing
         # duplicate episodes.
HARNESS="${1:?usage: run_supervised.sh <longmemeval|locomo> <target_answers>}"
TARGET="${2:?missing target answer count}"
# No new answer for this long => assume hung and restart. Must exceed the
# harness's own worst case: with_retry backs off up to 75s x 12 tries (~8-10min
# of legitimate silence) during a TPM-throttled stretch, and a big instance's
# ingest adds to that. 900s produced false restarts; 30min is the safe floor.
STALL_SECS="${STALL_SECS:-1800}"
POLL_SECS="${POLL_SECS:-60}"
EVALS="$(cd "$(dirname "$0")" && pwd)"
JSONL="$EVALS/${HARNESS}_results.jsonl"
HEARTBEAT="$EVALS/${HARNESS}_heartbeat"
LOG="$EVALS/${HARNESS}_supervised.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

kill_tree() {  # kill the job's whole process group, then verify nothing survived
  local pid="$1"
  kill -9 -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  for _ in 1 2 3 4 5; do
    pgrep -f "${HARNESS}_harness.py" >/dev/null || return 0
    sleep 1
  done
  # Last resort: a survivor here would race the next launch on the same graphs.
  say "WARNING: ${HARNESS}_harness.py survived group kill -- pkill'ing stragglers"
  pkill -9 -f "${HARNESS}_harness.py" 2>/dev/null
  sleep 2
}

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
    # Heartbeat = the harness did ANY unit of work (an episode ingested, a
    # retrieval, an answer). Answers alone are too coarse: they arrive in bursts
    # and a healthy run goes quiet for 30+ min during ingest or rate-limit
    # backoff. Only when BOTH are frozen is the process actually hung.
    if [ -f "$HEARTBEAT" ]; then
      hb=$(stat -f %m "$HEARTBEAT" 2>/dev/null || stat -c %Y "$HEARTBEAT" 2>/dev/null || echo 0)
      [ "$hb" -gt "$last_change" ] && last_change=$hb && continue
    fi
    if [ $(( $(date +%s) - last_change )) -ge "$STALL_SECS" ]; then
      say "STALLED at $now_n answers for ${STALL_SECS}s -- restarting"
      kill_tree "$pid"
      break
    fi
  done
  wait "$pid" 2>/dev/null
  say "harness exited; $(count_done)/$TARGET answered"
  sleep 5
done
