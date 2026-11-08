#!/usr/bin/env bash
# E5 (sink vs random keys on ImageNet) -> analysis -> resume I3. Resumable. GPU_DEADLINE set by the caller.
set -u
cd "$(dirname "$0")/.."
R=/Data/gaetan.degot/keysel
PY=.venv/bin/python
echo "$(date +%T) start e5"
$PY -m keysel.imagenet e5 --key_minutes 15 > $R/e5.log 2>&1; rc=$?
echo "$(date +%T) e5 exit $rc"
[ $rc -eq 0 ] && $PY -m keysel.run e5_analysis > $R/e5_analysis.log 2>&1
[ $rc -eq 75 ] && exit 75
echo "$(date +%T) start i3 (resume)"
$PY -m keysel.imagenet i3 --key_minutes 30 >> $R/i3.log 2>&1; echo "$(date +%T) i3 exit $?"
