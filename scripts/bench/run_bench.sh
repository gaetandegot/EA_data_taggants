#!/usr/bin/env bash
# GPU benchmark of the ladder primitives (HVP, crafting gradient): eager vs torch.func vs torch.compile, profiler,
# determinism (compiled HVP run twice in fresh processes: compare the two sha). ~10-20 min. GPU window only.
# Usage: bash scripts/bench/run_bench.sh <out_dir>
OUT=${1:?out dir}; mkdir -p "$OUT"
cd "$(dirname "$0")/../.."
PY=${PY:-$PWD/.venv/bin/python}
export PYTHONPATH=$PWD
timeout 1200 "$PY" scripts/bench/hvp_profile.py > "$OUT/hvp_profile.txt" 2>&1
for i in 1 2; do timeout 1200 "$PY" scripts/bench/hvp_compile.py resnet18_gn_silu > "$OUT/hvp_compile_$i.txt" 2>&1; done
echo "bench done $(date)" >> "$OUT/hvp_profile.txt"
