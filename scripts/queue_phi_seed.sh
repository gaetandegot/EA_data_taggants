#!/usr/bin/env bash
# One job of the late-window experiments, resumable at every level. R = job folder (pid, events, completion marker);
# RUN = run folder the stages work in (= R except for deit_auto). Jobs (JOB=...):
#   seed      alice -> gm -> ablation (label-free runs only) -> re-centred ladder at k0 = 32 28 24 (rounds of 2
#             iterations, 4 each). SET = config overrides of keysel/run_phi_seed.py, e.g. "n_keys=20 eps=8 pick=target".
#   select / select2   adjoint image selection on a finished label-free run (keysel/phi_select.py; select2: --extra)
#   deit_auto hard same-label DeiT-tiny: Alice at each learning rate of LRS (folders R/lr<lr>), the shared choice
#             (keysel/phi_lr_choice.py; written here if DECIDE=1, else waited for), then the seed job in R/lr<chosen>
#   export    copy the finished run's poisons to the shared folder outputs/phi_poisons/<run name> (keysel/phi_robust.py)
#   robust    Bob retrained from scratch: shuffle / other architecture / both (keysel/phi_robust.py), poisons from SRC
#   budget    DeiT-tiny, 10 keys x PER_KEY (50), k0=28: GM at every budget of BUDGETS, 4-iteration ladder at
#             LADDER_BUDGETS ("none": GM only); more than 512 positions = one merged poison step (keysel.phi.poison_scale)
# Only runs inside the GPU window: each command is hard-killed MARGIN_MIN before the window ends and the queue then
# waits for the next window. Exits non-zero on a crash (state/failures is incremented; the phi tick relaunches).
# When the job is finished, starts the next job registered for this machine (scripts/phi_tick.sh).
# Env: SEED (required), R, JOB, SET, KS="32 28 24", ITERS=4, MARGIN_MIN=10, LRS, DECIDE, SRC, VARIANTS (robust),
#      BUDGETS, LADDER_BUDGETS.
set -u
SEED=${SEED:?SEED is required}
R=${R:-/Data/gaetan.degot/checkpoints/phi10_silu_s$SEED}
RUN=$R
KS=${KS:-32 28 24}
ITERS=${ITERS:-4}
MARGIN_MIN=${MARGIN_MIN:-10}
JOB=${JOB:-seed}
SET=${SET:-}
LRS=${LRS:-0.1 0.25 0.5}
DECIDE=${DECIDE:-0}
BUDGETS=${BUDGETS:-2,5,10,25,50}
LADDER_BUDGETS=${LADDER_BUDGETS:-2 5 10}
cd "$(dirname "$0")/.."
REPO=$PWD
PY=${PY:-$PWD/.venv/bin/python}
LRFILE=${LRFILE:-$REPO/outputs/deit_lr_choice.json}  # overridable for tests
DEIT_SET="n_keys=20 per_key=25 eps=8 pick=target arch=deit_tiny_cifar"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$R/state"
echo $$ > "$R/state/queue.pid"
ev() { echo "$(date -Is) $*" >> "$R/state/events.log"; echo "$(date -Is) $*"; }

left() {  # usable seconds left in the current window (0 if closed)
  set -- $(${WINDOW_STATUS:-python3 scripts/gpu_window.py status})  # WINDOW_STATUS: test hook
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

stage() {  # stage <name> [marker]: run_phi_seed stage in $RUN until its done marker exists
  local name=$1 marker=${2:-$1}
  mkdir -p "$RUN/state"
  while [ ! -f "$RUN/state/$marker.done" ]; do
    run "$name" "$RUN/$name.log" "$PY" -m keysel.run_phi_seed --out "$RUN" --seed "$SEED" --stage "$name" --set $SET \
      && touch "$RUN/state/$marker.done"
  done
}

ladder_state() {  # ladder_state <history.json>: prints "<accepted iterations> <stopped 0/1>"
  python3 - "$1" <<'EOF'
import json, sys
try:
    h = json.load(open(sys.argv[1]))
except OSError:
    print(0, 0); sys.exit()
# "no radius improved" without the smaller radii: phi_recenter --fallback reopens it, so not final
print(len(h.get("accepted", [])), int(bool(h.get("stopped")) and h.get("stopped") != "no radius improved"))
EOF
}

ladders() {  # the re-centred ladder from GM at every k0 of KS, rounds of 2 iterations, in $RUN
  for target in $(seq 2 2 "$ITERS"); do
    for k in $KS; do
      while true; do
        read -r n stopped <<< "$(ladder_state "$RUN/e2/k$k/recenter_gm/history.json")"
        [ "$stopped" = 1 ] || [ "$n" -ge "$target" ] && break
        run "ladder_k$k" "$RUN/ladder_k$k.log" "$PY" -m keysel.phi_recenter --run "$RUN" --k0 "$k" --init gm \
          --iters $(( target - n )) --steps 120 --restarts 1 --ladder 0.5,0.25,0.125 --adaptive --fallback 2 --cache
      done
    done
  done
}

wait_file() {  # wait_file <path> <label>: wait (sleeping, no GPU) until a file written by another job exists
  [ -f "$1" ] && return 0
  echo "waiting_$2" > "$R/state/current"; ev "waiting for $2 ($1)"
  while [ ! -f "$1" ]; do sleep 60; done
  ev "found $2"
}

lr_chosen() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['lr'])" "$1"; }

finish() {
  touch "$R/state/${1:-all}.done"
  echo done > "$R/state/current"
  ev "$JOB done"
  rm -f "$R/state/queue.pid"  # this queue no longer counts as running (the next job may use the same folder)
  ( exec bash scripts/phi_tick.sh ) > /dev/null 2>&1 < /dev/null &
  exit 0
}

case $JOB in
  select|select2)  # select2: + gradient-norm normalised selections (--extra)
    extra=""; [ "$JOB" = select2 ] && extra="--extra"
    while [ ! -f "$R/state/$JOB.done" ]; do
      run "$JOB" "$R/$JOB.log" "$PY" -m keysel.phi_select --run "$R" --k0 32 $extra && touch "$R/state/$JOB.done"
    done
    finish "$JOB" ;;

  export)  # from the run folder (R, or R/chosen for a deit_auto job), named after the job folder
    dst=$REPO/outputs/phi_poisons/$(basename "$R")
    src=$R; [ -e "$R/chosen" ] && src=$(readlink -f "$R/chosen")
    "$PY" -m keysel.phi_robust --export --run "$src" --to "$dst" >> "$R/export.log" 2>&1 || { ev "FAILED export"; exit 1; }
    finish export ;;

  robust)
    SRC=${SRC:?SRC (exported poisons folder) is required}
    [ -n "${POISONS:-}" ] || wait_file "$SRC/done" "exported poisons"   # POISONS: a file of the run folder SRC
    wait_file "$LRFILE" "DeiT learning rate"
    while [ ! -f "$R/state/robust.done" ]; do
      run robust "$R/robust.log" "$PY" -m keysel.phi_robust --src "$SRC" --out "$R" --lr-file "$LRFILE" \
        --variants "${VARIANTS:-shuffle,arch,both}" ${POISONS:+--poisons "$POISONS"} \
        && touch "$R/state/robust.done"
    done
    finish ;;

  deit_auto)
    SET0="$DEIT_SET"
    for lr in $LRS; do
      RUN=$R/lr$lr
      SET="$SET0 lr=$lr"
      stage alice
    done
    if [ "$DECIDE" = 1 ]; then
      [ -f "$LRFILE" ] || "$PY" -m keysel.phi_lr_choice --runs $(for lr in $LRS; do echo "$R/lr$lr"; done) \
        --out "$LRFILE" >> "$R/lr_choice.log" 2>&1 || { ev "FAILED lr choice"; exit 1; }
    fi
    choice=$LRFILE
    if [ ! -f "$LRFILE" ]; then  # wait for the deciding machine, at most 2 h, then decide on this seed's runs
      echo waiting_lr > "$R/state/current"; ev "waiting for the DeiT learning rate"
      for _ in $(seq 120); do [ -f "$LRFILE" ] && break; sleep 60; done
      if [ ! -f "$LRFILE" ]; then
        choice=$R/lr_choice_local.json
        "$PY" -m keysel.phi_lr_choice --runs $(for lr in $LRS; do echo "$R/lr$lr"; done) --out "$choice" \
          >> "$R/lr_choice.log" 2>&1 || { ev "FAILED local lr choice"; exit 1; }
        ev "no shared choice after 2 h: decided locally"
      fi
    fi
    L=$(lr_chosen "$choice")
    ln -sfn "lr$L" "$R/chosen"
    ev "DeiT learning rate $L"
    RUN=$R/lr$L
    SET="$SET0 lr=$L"
    stage gm
    ladders
    finish ;;

  budget)
    wait_file "$LRFILE" "DeiT learning rate"
    L=$(lr_chosen "$LRFILE")
    SET="n_keys=10 per_key=${PER_KEY:-50} eps=${EPS:-8} pick=target arch=deit_tiny_cifar lr=$L k0s=28"
    [ "$LADDER_BUDGETS" = none ] && LADDER_BUDGETS=""   # GM only
    stage alice
    stage gm
    while [ ! -f "$R/state/budget_gm.done" ]; do
      run budget_gm "$R/budget_gm.log" "$PY" -m keysel.phi_budget --run "$R" --k0 28 --budgets "$BUDGETS" \
        && touch "$R/state/budget_gm.done"
    done
    for target in $(seq 2 2 "$ITERS"); do
      for b in $LADDER_BUDGETS; do
        while true; do
          read -r n stopped <<< "$(ladder_state "$R/e2/k28/budget_b$b/history.json")"
          [ "$stopped" = 1 ] || [ "$n" -ge "$target" ] && break
          run "budget_ladder_b$b" "$R/budget_ladder_b$b.log" "$PY" -m keysel.phi_recenter --run "$R" --k0 28 \
            --init-file "$R/budget/poisons_b$b.pt" --budget "$b" --subdir "budget_b$b" --iters $(( target - n )) \
            --steps 120 --restarts 1 --ladder 0.5,0.25,0.125 --adaptive --fallback 2 --cache
        done
      done
    done
    finish "${DONE_TAG:-all}" ;;   # DONE_TAG=retry: a *_retry job of phi_tick.sh on a finished run

  seed)
    stage alice
    stage gm
    stage ablation
    stage ablation ablation3   # fills the gm_same_last arm in runs whose ablation predates it (no-op otherwise)
    ladders
    case " $SET " in *" keys=backdoor "*) stage beval ;; esac   # backdoor: success on the triggered test images
    finish ;;

  *) ev "unknown JOB $JOB"; exit 1 ;;
esac
