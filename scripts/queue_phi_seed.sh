#!/usr/bin/env bash
# One job of the multi-seed late-window experiments, resumable at every level. JOB=seed (default):
#   alice -> gm (2 restarts) -> ablation (label-free runs only: same-label GM, 3 arms of Bob at every k0)
#   -> re-centred ladder from GM at k0 = 32 28 24, in rounds of 2 iterations (all k0 progress), 4 iterations each.
#   SET = config overrides of keysel/run_phi_seed.py at creation, e.g. "n_keys=20 per_key=25 eps=8 pick=target".
# JOB=select: adjoint-based image selection on a finished label-free run (keysel/phi_select.py, k0=32);
# JOB=select2: the same plus the gradient-norm normalised selections (phi_select --extra).
# Only runs inside the GPU window: each command is hard-killed MARGIN_MIN before the window ends and the queue then
# waits for the next window. Exits non-zero on a crash (state/failures is incremented; the phi tick relaunches).
# When the job is finished, starts the next job registered for this machine (scripts/phi_tick.sh).
# Usage (normally started by scripts/phi_tick.sh / phi_launch.sh):
#   SEED=1 nohup bash scripts/queue_phi_seed.sh >> <run>/queue.log 2>&1 &
# Env: SEED (required), R (run folder), JOB, SET, KS="32 28 24", ITERS=4 (per k0), MARGIN_MIN=10.
set -u
SEED=${SEED:?SEED is required}
R=${R:-/Data/gaetan.degot/checkpoints/phi10_silu_s$SEED}
KS=${KS:-32 28 24}
ITERS=${ITERS:-4}
MARGIN_MIN=${MARGIN_MIN:-10}
JOB=${JOB:-seed}
SET=${SET:-}
cd "$(dirname "$0")/.."
PY=${PY:-$PWD/.venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$R/state"
echo $$ > "$R/state/queue.pid"
ev() { echo "$(date -Is) $*" >> "$R/state/events.log"; echo "$(date -Is) $*"; }

left() {  # usable seconds left in the current window (0 if closed)
  set -- $(python3 scripts/gpu_window.py status)
  if [ "$1" = open ]; then echo $(( $3 - MARGIN_MIN * 60 )); else echo 0; fi
}

wait_window() {
  local l; l=$(left)
  if [ "$l" -le 300 ]; then echo idle > "$R/state/current"; ev "waiting for the GPU window"; fi
  while [ "$(left)" -le 300 ]; do sleep 60; done
}

run() {  # run <stage name> <log file> <command...>: one attempt inside the window
  local name=$1 log=$2; shift 2
  wait_window
  local l; l=$(left)
  echo "$name" > "$R/state/current"
  export GPU_DEADLINE=$(( $(date +%s) + l ))
  ev "start $name"
  timeout --signal=KILL "$l" "$@" >> "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    rm -f "$R/state/failures"; ev "end $name"
    return 0
  fi
  if [ $rc -eq 75 ] || [ $rc -eq 137 ] || [ "$(left)" -le 300 ]; then ev "stopped $name at the end of the window (rc $rc)"; return 75; fi
  echo $(( $(cat "$R/state/failures" 2>/dev/null || echo 0) + 1 )) > "$R/state/failures"
  ev "FAILED $name (rc $rc, see $log)"
  exit 1
}

stage() {  # stage <name> [marker]: run_phi_seed stage until its done marker exists
  local name=$1 marker=${2:-$1}
  while [ ! -f "$R/state/$marker.done" ]; do
    run "$name" "$R/$name.log" "$PY" -m keysel.run_phi_seed --out "$R" --seed "$SEED" --stage "$name" --set $SET \
      && touch "$R/state/$marker.done"
  done
}

ladder_state() {  # prints "<accepted iterations> <stopped 0/1>" for k0=$1
  python3 - "$R/e2/k$1/recenter_gm/history.json" <<'EOF'
import json, sys
try:
    h = json.load(open(sys.argv[1]))
except OSError:
    print(0, 0); sys.exit()
print(len(h.get("accepted", [])), int(bool(h.get("stopped"))))
EOF
}

next_job() {  # the job is finished: start the next job registered for this machine right away
  rm -f "$R/state/queue.pid"  # this queue no longer counts as running (the next job may use the same folder)
  ( exec bash scripts/phi_tick.sh ) > /dev/null 2>&1 < /dev/null &
}

if [ "$JOB" = select ] || [ "$JOB" = select2 ]; then  # select2: + gradient-norm normalised selections (--extra)
  extra=""; [ "$JOB" = select2 ] && extra="--extra"
  while [ ! -f "$R/state/$JOB.done" ]; do
    run "$JOB" "$R/$JOB.log" "$PY" -m keysel.phi_select --run "$R" --k0 32 $extra && touch "$R/state/$JOB.done"
  done
  echo done > "$R/state/current"
  ev "$JOB done"
  next_job
  exit 0
fi

stage alice
stage gm
stage ablation
stage ablation ablation3   # fills the gm_same_last arm in runs whose ablation predates it (no-op otherwise)
for target in $(seq 2 2 "$ITERS"); do
  for k in $KS; do
    while true; do
      read -r n stopped <<< "$(ladder_state "$k")"
      [ "$stopped" = 1 ] || [ "$n" -ge "$target" ] && break
      run "ladder_k$k" "$R/ladder_k$k.log" "$PY" -m keysel.phi_recenter --run "$R" --k0 "$k" --init gm \
        --iters $(( target - n )) --steps 120 --restarts 1 --ladder 0.5,0.25,0.125 --adaptive --stop-gain 0.02 --cache
    done
  done
done
touch "$R/state/all.done"
echo done > "$R/state/current"
ev "queue done"
next_job
