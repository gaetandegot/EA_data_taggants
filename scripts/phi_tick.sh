#!/usr/bin/env bash
# Called over SSH by the laptop monitor (scripts/phi_watch/PhiWatch.ps1):
#   ssh <machine> bash ~/EA/EA_data_taggants/scripts/phi_tick.sh
# The machine's jobs are listed in ~/.config/data-taggants/phi_run.<host>, one per line, run in order:
#   <seed> <run folder> [preset]     preset: labelfree (default) | select | select2 | hard | hard_deit | deit_auto | deit_auto_decide | export
#                                  | robust | robust_deit | robust_spread | robust_same | robust_lf | budget | budget_e16 | budget_e16_<b>_<b> | budget100 | <any>_retry
# The first job that is not finished is the active one. If its queue should be running but is not (GPU window open with
# >= 30 min left, GPU unused or >= 13 GB free, fewer than 8 consecutive failures), relaunches scripts/queue_phi_seed.sh detached with the
# preset's settings; then prints the state as ONE JSON line (scripts/phi_status.py --chain).
# This relaunches the queue after the nightly 00:33 cleanup: the user has taken responsibility for that (see
# CLAUDE.md, "Nightly cleanup"). `bash scripts/phi_launch.sh --stop` unregisters every job of this machine.
set -uo pipefail
exec </dev/null
repo="$(cd "$(dirname "$0")/.." && pwd)"
pointer="$HOME/.config/data-taggants/phi_run.$(hostname -s)"
[ -f "$pointer" ] || exec /usr/bin/python3 "$repo/scripts/phi_status.py" --none

marker() {  # marker <run folder> <preset>: file whose existence means the job is finished
  case $2 in select|select2|export) echo "$1/state/$2.done" ;; *_retry) echo "$1/state/retry.done" ;; *) echo "$1/state/all.done" ;; esac
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
  read -r mem free <<< "$(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ',')"
  # GPU usable: (almost) unused, or shared with another job but >= 13 GB free (our peak was ~10-12 GB);
  # per-machine override in MiB: ~/.config/data-taggants/share_min_mib.<host>
  share_min=$(cat "$HOME/.config/data-taggants/share_min_mib.$(hostname -s)" 2>/dev/null || echo 13312)
  gpu_ok=0; { [ "${mem:-99999}" -lt 2000 ] || [ "${free:-0}" -ge "$share_min" ]; } && gpu_ok=1
  if [ "$wstate" = open ] && [ "$wleft" -ge 1800 ] && [ "$fails" -lt 8 ] && [ $gpu_ok = 1 ]; then
    base=${PRESET%_retry} done_tag=all   # <preset>_retry: the same job again on its finished run (state/retry.done)
    [ "$base" != "$PRESET" ] && done_tag=retry
    case $base in
      labelfree) job=seed set="" ;;
      hard) job=seed set="n_keys=20 per_key=25 eps=8 pick=target" ;;
      hard_deit) job=seed set="n_keys=20 per_key=25 eps=8 pick=target arch=deit_tiny_cifar" ;;
      select) job=select set="" ;;
      select2) job=select2 set="" ;;
      deit_auto) job=deit_auto set="" ;;
      deit_auto_decide) job=deit_auto set="" decide=1 ;;
      export) job=export set="" ;;
      robust) job=robust set="" src="$repo/outputs/phi_poisons/phi20_e8_same_s$SEED" ;;
      robust_deit) job=robust set="" src="$repo/outputs/phi_poisons/phi20_e8_same_deit_auto_s$SEED" ;;
      robust_spread) job=robust set="" src="$repo/outputs/phi_poisons/phi20_e8_same_s$SEED" variants=shuffle_spread ;;
      robust_same) job=robust set="" src="/Data/gaetan.degot/checkpoints/phi10_silu_s$SEED" variants=shuffle,shuffle_spread \
        poisons=ablation/poisons_same.pt ;;
      robust_lf) job=robust set="" src="$repo/outputs/phi_poisons/phi10_silu_s$SEED" variants=shuffle ;;
      budget) job=budget set="" ;;
      budget_e16) job=budget set="" eps=16 budgets=2,5,10,25,50 ladder_budgets="2 5 10 25 50" ;;
      budget_e16_*) bl=${base#budget_e16_}   # e.g. budget_e16_15_20: GM + ladder at 15 and 20 only (same config as budget_e16)
        job=budget set="" eps=16 budgets=${bl//_/,} ladder_budgets="${bl//_/ }" ;;
      budget100) job=budget set="" per_key=100 budgets=50,75,100 ladder_budgets=none ;;
      *) echo "error: unknown preset $PRESET in $pointer" >&2; exit 1 ;;
    esac
    # whole subshell redirected: nothing may keep the caller's stdout (ssh channel / pipe) open
    ( cd "$repo" && SEED=$SEED R=$R JOB=$job SET=$set DECIDE=${decide:-0} DONE_TAG=$done_tag SRC=${src:-} VARIANTS=${variants:-} POISONS=${poisons:-} PER_KEY=${per_key:-} EPS=${eps:-} BUDGETS=${budgets:-2,5,10,25,50} \
        LADDER_BUDGETS=${ladder_budgets:-2 5 10} \
        exec setsid nohup bash scripts/queue_phi_seed.sh ) \
      >> "$R/queue.log" 2>&1 < /dev/null &
    echo "$(date -Is) relaunch by phi_tick ($PRESET, queue was not running)" >> "$R/state/events.log"
    relaunched="--relaunched"
    sleep 2
  fi
fi
exec /usr/bin/python3 "$repo/scripts/phi_status.py" --chain "$pointer" $relaunched
