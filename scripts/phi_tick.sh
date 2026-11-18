#!/usr/bin/env bash
# Called over SSH by the laptop monitor (scripts/phi_watch/PhiWatch.ps1):
#   ssh <machine> bash ~/EA/EA_data_taggants/scripts/phi_tick.sh
# The machine's jobs are listed in ~/.config/data-taggants/phi_run.<host>, one per line, run in order:
#   <seed> <run folder> [preset]     preset: labelfree (default) | select | hard
# The first job that is not finished is the active one. If its queue should be running but is not (GPU window open with
# >= 30 min left, GPU free, fewer than 8 consecutive failures), relaunches scripts/queue_phi_seed.sh detached with the
# preset's settings; then prints the state as ONE JSON line (scripts/phi_status.py --chain).
# This relaunches the queue after the nightly 00:33 cleanup: the user has taken responsibility for that (see
# CLAUDE.md, "Nightly cleanup"). `bash scripts/phi_launch.sh --stop` unregisters every job of this machine.
set -uo pipefail
exec </dev/null
repo="$(cd "$(dirname "$0")/.." && pwd)"
pointer="$HOME/.config/data-taggants/phi_run.$(hostname -s)"
[ -f "$pointer" ] || exec /usr/bin/python3 "$repo/scripts/phi_status.py" --none

marker() {  # marker <run folder> <preset>: file whose existence means the job is finished
  case $2 in select) echo "$1/state/select.done" ;; *) echo "$1/state/all.done" ;; esac
}
SEED="" R="" PRESET=""
while read -r s r p; do
  [ -n "${s:-}" ] || continue
  SEED=$s R=$r PRESET=${p:-labelfree}
  [ -f "$(marker "$R" "$PRESET")" ] || break   # first unfinished job (else: the last one, finished)
done < "$pointer"
[ -n "$R" ] || exec /usr/bin/python3 "$repo/scripts/phi_status.py" --none
mkdir -p "$R/state" && date -Is > "$R/state/tick_heartbeat"

relaunched=""
running=0
pid=$(cat "$R/state/queue.pid" 2>/dev/null) && tr "\\0" " " 2>/dev/null < "/proc/$pid/cmdline" | grep -q queue_phi_seed && running=1
if [ $running = 0 ] && [ ! -f "$(marker "$R" "$PRESET")" ]; then
  read -r wstate _ wleft <<< "$(/usr/bin/python3 "$repo/scripts/gpu_window.py" status)"
  fails=$(cat "$R/state/failures" 2>/dev/null || echo 0)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ "$wstate" = open ] && [ "$wleft" -ge 1800 ] && [ "$fails" -lt 8 ] && [ "${mem:-99999}" -lt 2000 ]; then
    case $PRESET in
      labelfree) job=seed set="" ;;
      hard) job=seed set="n_keys=20 per_key=25 eps=8 pick=target" ;;
      select) job=select set="" ;;
      *) echo "error: unknown preset $PRESET in $pointer" >&2; exit 1 ;;
    esac
    # whole subshell redirected: nothing may keep the caller's stdout (ssh channel / pipe) open
    ( cd "$repo" && SEED=$SEED R=$R JOB=$job SET=$set exec setsid nohup bash scripts/queue_phi_seed.sh ) \
      >> "$R/queue.log" 2>&1 < /dev/null &
    echo "$(date -Is) relaunch by phi_tick ($PRESET, queue was not running)" >> "$R/state/events.log"
    relaunched="--relaunched"
    sleep 2
  fi
fi
exec /usr/bin/python3 "$repo/scripts/phi_status.py" --chain "$pointer" $relaunched
