#!/usr/bin/env bash
# C2 -> C1 -> C5 -> analysis, started after C0 ends. Resumable: rerun to continue after a kill.
# Uses .venv/bin/python directly (no uv lock). GPU_DEADLINE must be set by the caller.
set -u
cd "$(dirname "$0")/.."
R=/Data/gaetan.degot/keysel
PY=.venv/bin/python
while tmux has-session -t keysel_c0 2>/dev/null; do sleep 30; done
[ -f $R/c0/test_acc.json ] || { echo "C0 incomplete (no c0/test_acc.json)"; exit 1; }
for st in c2 c1 c5; do
  echo "$(date +%T) start $st"
  $PY -m keysel.run $st > $R/$st.log 2>&1; rc=$?
  [ $rc -eq 0 ] || { echo "$(date +%T) $st exit $rc (75 = deadline)"; exit $rc; }
  echo "$(date +%T) $st done"
done
$PY -m keysel.run analyze > $R/analyze.log 2>&1; echo "$(date +%T) analyze exit $?"
