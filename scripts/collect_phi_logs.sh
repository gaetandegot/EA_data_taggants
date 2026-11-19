#!/usr/bin/env bash
# Copy the small result files of this machine's phi runs (logs, JSON; no .pt) from the local /Data into the shared
# home, so they can be analysed from any machine:
#   ssh <machine> bash ~/EA/EA_data_taggants/scripts/collect_phi_logs.sh
# -> ~/EA/EA_data_taggants/outputs/phi_results/<host>/<run>/...   (outputs/ is gitignored; re-running updates it)
set -uo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
host=$(hostname -s)
out="$repo/outputs/phi_results/$host"
mkdir -p "$out"
cd /Data/gaetan.degot/checkpoints 2>/dev/null || { echo "$host: no /Data/gaetan.degot/checkpoints"; exit 0; }
n=0
for d in phi10_silu_s* phi20_e8_same_s* phi20_e8_same_deit_s*; do
  [ -d "$d" ] || continue
  rsync -a --prune-empty-dirs --include='*/' --include='*.log' --include='*.json' --include='*.jsonl' \
    --exclude='*' "$d" "$out/" && n=$((n + 1))
done
cp -f "$HOME/.config/data-taggants/phi_run.$host" "$out/job_list.txt" 2>/dev/null
echo "$host: $n run folder(s) -> $out ($(du -sh "$out" | cut -f1))"
