#!/usr/bin/env bash
# Start (or restart after a killed session) the timed pipeline in a detached tmux session.
# Every stage resumes from its own checkpoint, so relaunching with the same settings is safe:
#   sign: poisons.pth every 10 iterations | pretrain/validate: checkpoint.pth every epoch.
# The exact command is saved to $OUTPUT_ROOT/relaunch.sh; after a crash just run that file.
# Usage: DATA_PATH=... [MODEL=... EPOCHS=... N_KEYS=... PBATCH=... STAGES="sign validate detect"] bash scripts/launch_tmux.sh
set -euo pipefail
: "${DATA_PATH:?Set DATA_PATH}"
: "${OUTPUT_ROOT:=outputs/timing}"
: "${SESSION:=taggants}"
repo="$(cd "$(dirname "$0")/.." && pwd)"
vars=(DATA_PATH OUTPUT_ROOT MODEL OPT LR EPOCHS WORKERS N_KEYS BUDGET_PER_KEY PBATCH RESERVE_GPU_MEM STAGES)
env_str=""
for v in "${vars[@]}"; do [ -n "${!v+x}" ] && env_str+="$v=$(printf %q "${!v}") "; done
cmd="cd $(printf %q "$repo") && source .venv/bin/activate && ${env_str}bash scripts/timed_pipeline.sh; echo PIPELINE_EXIT; exec bash"
mkdir -p "$OUTPUT_ROOT"
printf '#!/usr/bin/env bash\n# Relaunch the same pipeline (stages resume from their checkpoints)\ntmux kill-session -t %q 2>/dev/null\ntmux new-session -d -s %q %q\n' "$SESSION" "$SESSION" "$cmd" > "$OUTPUT_ROOT/relaunch.sh"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "tmux session '$SESSION' already running" >&2; exit 1; fi
tmux new-session -d -s "$SESSION" "$cmd"
echo "Started tmux session '$SESSION'. Relaunch after a crash with: bash $OUTPUT_ROOT/relaunch.sh"
