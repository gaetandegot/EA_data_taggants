#!/usr/bin/env bash
# SiLU run: GM with 8 restarts (once, serves every k0), then the re-centred radius ladder from it at k0=32,28,24,
# in rounds of 2 iterations per k0 (so every k0 progresses), up to 8 iterations each; a ladder stops by itself when
# the kept real gain is < 0.02 (marker "stopped" in history.json) and is then skipped. Resumable: relaunch the same way.
# Usage: nohup bash scripts/queue_phi_gm8_ladder.sh > <run>/queue_gm8.log 2>&1 &
R=${R:-/Data/gaetan.degot/checkpoints/phi_r18gn_silu_s0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
PY=${PY:-$PWD/.venv/bin/python}
until [ "$(python3 scripts/gpu_window.py status | cut -d' ' -f1)" = open ]; do sleep 60; done
[ -f "$R/gm8/poisons.pt" ] || "$PY" -m keysel.phi_gm8 --run "$R" --restarts 8 >> "$R/run_gm8.log" 2>&1
for round in 1 2 3 4; do
  for k in 32 28 24; do
    "$PY" -m keysel.phi_recenter --run "$R" --k0 $k --init gm8 --iters 2 --steps 120 --restarts 1 \
      --ladder 0.5,0.25,0.125 --adaptive --stop-gain 0.02 >> "$R/run_gm8_ladder_k$k.log" 2>&1
  done
done
echo "queue done $(date)"
