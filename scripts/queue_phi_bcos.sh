#!/usr/bin/env bash
# Queue for the B-cos experiments on the SiLU run (sequential, resumable: each step skipped if its result exists).
# Usage: nohup bash scripts/queue_phi_bcos.sh > <run>/queue_bcos.log 2>&1 &
R=${R:-/Data/gaetan.degot/checkpoints/phi_r18gn_silu_s0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
PY=${PY:-$PWD/.venv/bin/python}
until [ "$(python3 scripts/gpu_window.py status | cut -d' ' -f1)" = open ]; do sleep 60; done
bcos() { for s in group global; do
  [ -f "$R/e2/k$1/results_Bcos_$s.json" ] || "$PY" -m keysel.phi_bcos --run "$R" --k0 "$1" --scope $s > "$R/run_bcos_k$1_$s.log" 2>&1
done; }
e2()   { [ -f "$R/e2/k$1/results.json" ] || "$PY" -m keysel.phi_dyn --run "$R" --k0s "$1" --restarts 2 --steps 240 > "$R/run_e2_k$1.log" 2>&1; }
bcos 32
e2 28; bcos 28
e2 24; bcos 24
echo "queue done $(date)"
