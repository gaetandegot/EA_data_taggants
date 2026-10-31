#!/usr/bin/env bash
# I1 -> I3 after the CIFAR driver ends. Resumable. GPU_DEADLINE must be set by the caller.
set -u
cd "$(dirname "$0")/.."
R=/Data/gaetan.degot/keysel
PY=.venv/bin/python
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
while tmux has-session -t keysel_cifar 2>/dev/null; do sleep 30; done
for st in i1 i3; do
  echo "$(date +%T) start $st"
  $PY -m keysel.imagenet $st --key_minutes 30 > $R/$st.log 2>&1; rc=$?
  [ $rc -eq 0 ] || { echo "$(date +%T) $st exit $rc (75 = deadline)"; exit $rc; }
  echo "$(date +%T) $st done"
done
