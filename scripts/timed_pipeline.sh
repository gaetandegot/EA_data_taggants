#!/usr/bin/env bash
# Run all four stages sequentially and record wall-clock time per stage.
# Usage: DATA_PATH=... [EPOCHS=2 N_KEYS=2 STAGES="validate detect" ...] bash scripts/timed_pipeline.sh
set -uo pipefail
: "${OUTPUT_ROOT:=outputs/timing}"
: "${STAGES:=pretrain sign validate detect}"
export OUTPUT_ROOT
mkdir -p "$OUTPUT_ROOT"
timing="$OUTPUT_ROOT/timing.txt"

# Log every GPU compute process (with its owner) once per second, so memory
# contention from other jobs can be attributed after the fact.
(
  while true; do
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits |
      while IFS=, read -r pid mem; do
        pid=${pid// /}
        echo "$(date -Is) pid=$pid mem=${mem// /}MiB $(ps -p "$pid" -o user=,args= 2>/dev/null | cut -c1-120)"
      done
    sleep 1
  done
) >> "$OUTPUT_ROOT/gpu_monitor.log" 2>&1 &
monitor=$!
trap 'kill "$monitor" 2>/dev/null' EXIT

{ echo "start: $(date -Is)"; echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)";
  echo "MODEL=${MODEL:-} OPT=${OPT:-} EPOCHS=${EPOCHS:-} N_KEYS=${N_KEYS:-} WORKERS=${WORKERS:-} RESERVE_GPU_MEM=${RESERVE_GPU_MEM:-} STAGES=$STAGES"; } >> "$timing"
for stage in $STAGES; do
  t0=$(date +%s)
  bash scripts/reproduce.sh "$stage" 2>&1 | tee -a "$OUTPUT_ROOT/$stage.log"
  rc=${PIPESTATUS[0]}
  echo "$stage: $(( $(date +%s) - t0 ))s (exit $rc)" >> "$timing"
  [ "$rc" -eq 0 ] || { echo "$stage failed" >> "$timing"; exit "$rc"; }
done
echo "end: $(date -Is)" >> "$timing"
