#!/usr/bin/env bash
# Called over SSH by the laptop watchdog (or any external scheduler):
#   ssh <machine> bash ~/EA/EA_data_taggants/scripts/remote_tick.sh
# Re-installs the cron entry if the nightly cleanup removed it, runs one supervisor tick
# (which itself respects the GPU window, the free-GPU check, the failure limit and the
# "all stages done" state), then prints the run state as ONE JSON line on stdout
# (scripts/run_status.py; human messages go to stderr).
# The active run of this host is the one recorded by launch_supervised.sh in
# ~/.config/data-taggants/active_run.<host> (the path of its supervisor.env), so
# `launch_supervised.sh --uninstall` turns these ticks into no-ops ({"active": false}).
set -uo pipefail
exec </dev/null   # the ssh channel must not stay attached to the detached pipeline
repo="$(cd "$(dirname "$0")/.." && pwd)"
pointer="$HOME/.config/data-taggants/active_run.$(hostname -s)"
tag="# data-taggants-supervisor"

read -r env_file 2>/dev/null < "$pointer" || { echo "idle: no active run on $(hostname -s)" >&2; exec /usr/bin/python3 "$repo/scripts/run_status.py" --none; }
[ -f "$env_file" ] || { echo "error: $env_file not found" >&2; exit 1; }
root="$(dirname "$env_file")"

rearmed=""
if ! crontab -l 2>/dev/null | grep -qF "$tag"; then
  bash "$repo/scripts/launch_supervised.sh" --install "$root" >/dev/null 2>&1
  echo "$(date -Is) cron entry was missing: re-installed by remote_tick" >> "$root/supervisor.log"
  rearmed="--rearmed"
fi
mkdir -p "$root/state" && date -Is > "$root/state/remote_heartbeat"
bash "$repo/scripts/supervisor.sh" "$env_file" >&2
exec /usr/bin/python3 "$repo/scripts/run_status.py" "$env_file" $rearmed
