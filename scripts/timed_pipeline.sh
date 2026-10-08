#!/usr/bin/env bash
# Run the pipeline stages sequentially and record wall-clock time per stage attempt.
# Resumable: finished stages are marked in $OUTPUT_ROOT/state/<stage>.done and skipped;
# a stage stopped by the GPU deadline (exit 75) or a crash resumes from its checkpoint.
# Usage: DATA_PATH=... [EPOCHS=2 N_KEYS=2 STAGES="validate detect" GPU_DEADLINE=<unix ts> ...] \
#        bash scripts/timed_pipeline.sh
set -uo pipefail
: "${OUTPUT_ROOT:=outputs/timing}"
: "${STAGES:=pretrain sign validate detect}"
export OUTPUT_ROOT
mkdir -p "$OUTPUT_ROOT/state"
timing="$OUTPUT_ROOT/timing.txt"

# Log every GPU compute process (with its owner) every 5 s, so memory contention
# from other jobs can be attributed after the fact.
(
  while true; do
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits |
      while IFS=, read -r pid mem; do
        pid=${pid// /}
        echo "$(date -Is) pid=$pid mem=${mem// /}MiB $(ps -p "$pid" -o user=,args= 2>/dev/null | cut -c1-120)"
      done
    sleep 5
  done
) >> "$OUTPUT_ROOT/gpu_monitor.log" 2>&1 &
monitor=$!
trap 'kill "$monitor" 2>/dev/null' EXIT

{ echo "start: $(date -Is)"; echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)";
  echo "MODEL=${MODEL:-} OPT=${OPT:-} EPOCHS=${EPOCHS:-} N_KEYS=${N_KEYS:-} WORKERS=${WORKERS:-} PBATCH=${PBATCH:-} RESERVE_GPU_MEM=${RESERVE_GPU_MEM:-} STAGES=$STAGES GPU_DEADLINE=${GPU_DEADLINE:-none}"; } >> "$timing"
for stage in $STAGES; do
  [ -f "$OUTPUT_ROOT/state/$stage.done" ] && { echo "$stage: already done, skipped" >> "$timing"; continue; }
  t0=$(date +%s)
  bash scripts/reproduce.sh "$stage" 2>&1 | tee -a "$OUTPUT_ROOT/$stage.log"
  rc=${PIPESTATUS[0]}
  echo "$stage: $(( $(date +%s) - t0 ))s (exit $rc)" >> "$timing"
  if [ "$rc" -eq 0 ]; then
    touch "$OUTPUT_ROOT/state/$stage.done"
  else
    [ "$rc" -eq 75 ] && echo "$stage stopped at the GPU deadline" >> "$timing" || echo "$stage failed" >> "$timing"
    exit "$rc"
  fi
done
echo "end: $(date -Is)" >> "$timing"
