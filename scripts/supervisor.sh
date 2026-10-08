#!/usr/bin/env bash
# Idempotent supervisor, meant to be run every few minutes from cron.
#   - outside the GPU window (see gpu_window.py): makes sure nothing is running
#   - inside it: (re)starts scripts/timed_pipeline.sh if it is not running, the GPU is
#     free and enough of the window is left; the pipeline gets GPU_DEADLINE (window end
#     minus a margin) and stops cleanly at it, resuming from its checkpoints next time
#   - survives reboots and killed sessions: cron re-runs it, all state lives in files
# Usage: bash scripts/supervisor.sh <supervisor.env>   (written by launch_supervised.sh)
set -uo pipefail
cfg="${1:?usage: supervisor.sh <supervisor.env>}"
mkdir -p "$(dirname "$cfg")/state" && date -Is > "$(dirname "$cfg")/state/heartbeat"  # proves cron is calling us
[ -f "$cfg" ] || exit 0   # installed but nothing launched yet: stay idle
# shellcheck disable=SC1090
source "$cfg"
cd "$REPO" || exit 1
: "${MARGIN_MIN:=10}" "${MIN_USEFUL_MIN:=30}" "${GPU_BUSY_MB:=2000}" "${MAX_FAILURES:=8}" "${RETRY_DELAY_MIN:=15}"
state="$OUTPUT_ROOT/state"; mkdir -p "$state"
log="$OUTPUT_ROOT/supervisor.log"
say() { echo "$(date -Is) $*" >> "$log"; }
say_once() { [ "$(cat "$state/last_msg" 2>/dev/null)" = "$*" ] || { say "$*"; echo "$*" > "$state/last_msg"; }; }

exec 9> "$state/supervisor.lock"
flock -n 9 || exit 0   # another tick is still working

all_done=1
for st in $STAGES; do [ -f "$state/$st.done" ] || all_done=0; done
if [ "$all_done" -eq 1 ]; then say_once "all stages done: nothing to do (remove the cron entry with scripts/launch_supervised.sh --uninstall)"; exit 0; fi

pidfile="$state/pipeline.pid"
pid=$(cat "$pidfile" 2>/dev/null || true)
running=0
if [ -n "$pid" ] && grep -q "timed_pipeline" "/proc/$pid/cmdline" 2>/dev/null; then running=1; fi

read -r wstate _ _ < <(python3 scripts/gpu_window.py status)   # real window, no margin
if [ "$wstate" = "closed" ]; then
  if [ "$running" -eq 1 ]; then
    say "window closed while the pipeline (pgid $pid) is running: TERM, then KILL"
    kill -TERM -- "-$pid" 2>/dev/null
    for _ in $(seq 1 12); do sleep 5; kill -0 "$pid" 2>/dev/null || break; done
    kill -KILL -- "-$pid" 2>/dev/null
  fi
  exit 0
fi
[ "$running" -eq 1 ] && exit 0

# window is open and nothing is running
read -r mstate deadline left < <(python3 scripts/gpu_window.py status --margin "$MARGIN_MIN")
if [ "$mstate" != "open" ] || [ "$left" -lt $((MIN_USEFUL_MIN * 60)) ]; then
  say_once "window open but less than ${MIN_USEFUL_MIN} min usable: waiting for the next one"; exit 0
fi
failures=$(cat "$state/failures" 2>/dev/null || echo 0)
if [ "$failures" -ge "$MAX_FAILURES" ]; then
  say_once "giving up after $failures consecutive failures: see $OUTPUT_ROOT/*.log, then delete $state/failures"; exit 0
fi
last_fail=$(cat "$state/last_failure_ts" 2>/dev/null || echo 0)
if [ $(( $(date +%s) - last_fail )) -lt $((RETRY_DELAY_MIN * 60)) ]; then exit 0; fi
busy=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
if [ "$busy" -gt "$GPU_BUSY_MB" ]; then say_once "GPU busy (${busy} MiB used by other compute processes): waiting"; exit 0; fi

say "launching pipeline; deadline $(date -d "@$deadline" -Is), stages: $STAGES, previous consecutive failures: $failures"
echo "$(date -Is)" > "$state/last_msg"
(
  export GPU_DEADLINE="$deadline"
  setsid nohup bash -c '
    echo $$ > "'"$pidfile"'"
    source "'"$REPO"'/.venv/bin/activate"
    bash scripts/timed_pipeline.sh
    rc=$?
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 75 ]; then rm -f "'"$state"'/failures"
    else echo $(( $(cat "'"$state"'/failures" 2>/dev/null || echo 0) + 1 )) > "'"$state"'/failures"; date +%s > "'"$state"'/last_failure_ts"; fi
    echo "$(date -Is) pipeline exited with code $rc" >> "'"$log"'"
  ' >> "$OUTPUT_ROOT/pipeline_run.log" 2>&1 9>&- &
)
exit 0
