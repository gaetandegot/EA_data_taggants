#!/usr/bin/env bash
# Register one seed of the multi-seed phi experiment on THIS machine and start it (one seed per machine):
#   bash scripts/phi_launch.sh <seed>|auto     auto: tarse 1, frontal 2, temporal 3
#   bash scripts/phi_launch.sh chain <preset> [seed]   append a job (labelfree | select | select2 | hard | hard_deit) after the current ones
#   bash scripts/phi_launch.sh --stop          unregister every job (phi_tick becomes a no-op) and stop the queue
# Run folder: /Data/gaetan.degot/checkpoints/phi10_silu_s<seed> (local disk). After that, the laptop watchdog's
# phi_tick relaunches the queue whenever it should be running; relaunching by hand is also just this command.
set -uo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
host=$(hostname -s)
pointer="$HOME/.config/data-taggants/phi_run.$host"
mkdir -p "$(dirname "$pointer")"

if [ "${1:-}" = --stop ]; then  # stop and unregister every job of this machine
  if [ -f "$pointer" ]; then
    while read -r SEED R _; do
      [ -n "${R:-}" ] || continue
      pid=$(cat "$R/state/queue.pid" 2>/dev/null) && kill -- -"$(ps -o pgid= "$pid" | tr -d ' ')" 2>/dev/null \
        && echo "$(date -Is) stopped by phi_launch --stop" >> "$R/state/events.log"
    done < "$pointer"
  fi
  rm -f "$pointer"; echo "unregistered every phi job on $host"; exit 0
fi

if [ "${1:-}" = chain ]; then  # chain <preset> [seed]: append a job run after the current ones (labelfree|select|hard)
  preset=${2:?usage: phi_launch.sh chain <labelfree|select|hard> [seed]}
  seed=${3:-$( [ -f "$pointer" ] && awk 'NF {print $1; exit}' "$pointer")}
  [ -n "$seed" ] || { echo "error: no seed given and none registered on $host" >&2; exit 1; }
  case $preset in
    labelfree|select|select2) run=/Data/gaetan.degot/checkpoints/phi10_silu_s$seed ;;
    hard) run=/Data/gaetan.degot/checkpoints/phi20_e8_same_s$seed ;;
    hard_deit) run=/Data/gaetan.degot/checkpoints/phi20_e8_same_deit_s$seed ;;
    *) echo "error: unknown preset $preset" >&2; exit 1 ;;
  esac
  grep -q "^$seed $run $preset\$" "$pointer" 2>/dev/null && { echo "already registered: $seed $run $preset"; exit 0; }
  echo "$seed $run $preset" >> "$pointer"
  echo "jobs on $host:"; sed 's/^/  /' "$pointer"; exit 0
fi

SEED=${1:?usage: phi_launch.sh <seed>|auto | --stop}
if [ "$SEED" = auto ] && [ -f "$pointer" ]; then  # keep the seed already registered on this machine
  read -r SEED _ < "$pointer"
fi
if [ "$SEED" = auto ]; then  # default machine -> seed assignment of the robustness experiment
  case $host in tarse) SEED=1 ;; frontal) SEED=2 ;; temporal) SEED=3 ;;
    *) echo "error: no seed assigned to $host (use phi_launch.sh <seed>)" >&2; exit 1 ;; esac
fi
R=/Data/gaetan.degot/checkpoints/phi10_silu_s$SEED
if [ -f "$pointer" ] && read -r s0 r0 < "$pointer" && [ "$s0" != "$SEED" ]; then
  echo "error: $host already runs seed $s0 ($r0); use --stop first" >&2; exit 1
fi
[ -f /Data/gaetan.degot/cifar100/train.parquet ] || [ -f /Data/gaetan.degot/cifar100/train.npz ] \
  || { echo "error: CIFAR-100 missing in /Data/gaetan.degot/cifar100 on $host" >&2; exit 1; }
if crontab -l 2>/dev/null | grep -qF "# data-taggants-supervisor" || pgrep -u "$USER" -f timed_pipeline.sh >/dev/null; then
  echo "warning: the ImageNet supervisor is installed or running on $host and will compete for the GPU." >&2
  echo "         stop it with: bash scripts/launch_supervised.sh --uninstall   (then re-run this command)" >&2
  [ "${FORCE:-0}" = 1 ] || exit 1
fi
if [ -f "$pointer" ]; then  # already registered: never rewrite the job list (it may hold several chained jobs)
  echo "already registered on $host: $(awk 'NF {printf "%s%s", sep, $0; sep=" | "}' "$pointer")"
else
  mkdir -p "$R/state"
  echo "$SEED $R" > "$pointer"
  echo "registered seed $SEED on $host -> $R"
fi
bash "$repo/scripts/phi_tick.sh" | /usr/bin/python3 -c "
import json, sys
s = json.loads(sys.stdin.read())
print('relaunched' if s.get('relaunched') else 'not started', '| queue running:', s['health']['pipeline_running'],
      '| window:', s['health']['gpu_window'].get('state'), '| stage:', s.get('current_stage'))"
