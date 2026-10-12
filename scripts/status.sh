#!/usr/bin/env bash
# Show the state of a run driven by scripts/launch_supervised.sh.
# Usage: bash scripts/status.sh [OUTPUT_ROOT]
# Without argument, OUTPUT_ROOT is read from the cron entry (supervisor.env path).
# Read-only; run it on the machine where launch_supervised.sh installed the cron entry.
set -uo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
tag="# data-taggants-supervisor"

if [ -t 1 ]; then B=$'\e[1m'; G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; D=$'\e[2m'; N=$'\e[0m'; else B=; G=; Y=; R=; D=; N=; fi
hdr() { printf '\n%s== %s ==%s\n' "$B" "$1" "$N"; }
ok()  { printf '  %s✔%s %s\n' "$G" "$N" "$*"; }
warn(){ printf '  %s!%s %s\n' "$Y" "$N" "$*"; }
bad() { printf '  %s✘%s %s\n' "$R" "$N" "$*"; }
info(){ printf '  %s\n' "$*"; }
age() {  # seconds -> "3h12m"
  local s=$1; [ "$s" -lt 0 ] && s=0
  if [ "$s" -ge 3600 ]; then printf '%dh%02dm' $((s/3600)) $((s%3600/60))
  elif [ "$s" -ge 60 ]; then printf '%dm%02ds' $((s/60)) $((s%60)); else printf '%ds' "$s"; fi
}
# last tqdm progress line of a log (tqdm rewrites with \r)
last_bar() { tail -c 20000 "$1" 2>/dev/null | tr '\r' '\n' | grep -E '[0-9]+%\||[0-9]+/[0-9]+' | tail -1 | sed 's/^ *//' | cut -c1-160; }

# ---------------------------------------------------------------- discovery
hdr "Supervisor"
cronline=$(crontab -l 2>/dev/null | grep -F "$tag" || true)
env_file=""
if [ -n "$cronline" ]; then
  ok "cron entry installed: ${D}${cronline%% /bin/bash*}${N}"
  env_file=$(sed -E 's#.*supervisor\.sh ([^ ]+) .*#\1#' <<< "$cronline")
else
  warn "no cron entry on this machine ($(hostname -s)); launch_supervised.sh was run elsewhere or --uninstall was used"
fi
if [ -n "${1:-}" ]; then
  case "$1" in /*) OUTPUT_ROOT="$1" ;; *) OUTPUT_ROOT="$PWD/$1" ;; esac
elif [ -n "$env_file" ] && [ -f "$env_file" ]; then
  OUTPUT_ROOT=$( (source "$env_file" >/dev/null 2>&1; echo "$OUTPUT_ROOT") )
else
  [ -n "$env_file" ] && bad "cron points to $env_file which does not exist (installed with --install but never launched?)"
  echo "Pass the run folder explicitly: bash scripts/status.sh <OUTPUT_ROOT>"; exit 1
fi
[ -d "$OUTPUT_ROOT" ] || { bad "run folder not found: $OUTPUT_ROOT"; exit 1; }
state="$OUTPUT_ROOT/state"
now=$(date +%s)
info "run folder: $B$OUTPUT_ROOT$N"
[ -f "$OUTPUT_ROOT/run_config.txt" ] && info "config:     $(grep -o -- '--model [^ ]*' "$OUTPUT_ROOT/run_config.txt" | head -1), $(head -1 "$OUTPUT_ROOT/run_config.txt")"

# heartbeat: written by every cron tick (even if idle)
if [ -f "$state/heartbeat" ]; then
  hb=$(date -d "$(cat "$state/heartbeat")" +%s 2>/dev/null || echo 0)
  a=$((now - hb))
  if [ "$a" -le 600 ]; then ok "last cron tick $(age $a) ago"; else bad "last cron tick $(age $a) ago (cron not firing? crond down?)"; fi
else
  bad "no heartbeat file: the supervisor never ran"
fi
if [ -f "$state/remote_heartbeat" ]; then
  a=$((now - $(date -d "$(cat "$state/remote_heartbeat")" +%s 2>/dev/null || echo 0)))
  if [ "$a" -le 900 ]; then ok "last remote tick (watchdog VM) $(age $a) ago"; else warn "last remote tick (watchdog VM) $(age $a) ago"; fi
  [ -f "$HOME/.config/data-taggants/active_run.$(hostname -s)" ] || warn "no active-run pointer: remote ticks are no-ops (--uninstall was used)"
fi

# pipeline process
pid=$(cat "$state/pipeline.pid" 2>/dev/null || true)
running=0
if [ -n "$pid" ] && grep -q timed_pipeline "/proc/$pid/cmdline" 2>/dev/null; then
  running=1
  et=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
  ok "pipeline RUNNING (pgid $pid, up $(age "${et:-0}"))"
  procs=$(pgrep -af "(pretrain|sign|validate|detect)\.py" | grep -F -- "$OUTPUT_ROOT" | cut -c1-90 | head -3)
  [ -n "$procs" ] && sed 's/^/      /' <<< "$procs"
else
  warn "pipeline not running"
fi

# GPU window
read -r wstate wts wleft < <(python3 "$repo/scripts/gpu_window.py" status 2>/dev/null || echo "? 0 0")
if [ "$wstate" = "open" ]; then ok "GPU window OPEN, closes in $(age "${wleft%.*}") ($(date -d "@${wts%.*}" '+%a %H:%M'))"
else info "GPU window closed, opens in $(age "${wleft%.*}") ($(date -d "@${wts%.*}" '+%a %H:%M'))"; fi

# failures / waiting reason
fails=$(cat "$state/failures" 2>/dev/null || echo 0)
max=8; [ -n "$env_file" ] && [ -f "$env_file" ] && max=$( (source "$env_file" >/dev/null 2>&1; echo "${MAX_FAILURES:-8}") )
if [ "$fails" -ge "$max" ]; then bad "GAVE UP: $fails consecutive failures (fix, then rm $state/failures)"
elif [ "$fails" -gt 0 ]; then warn "consecutive failures: $fails / $max"
else ok "no consecutive failures"; fi
[ -f "$state/last_msg" ] && [ "$running" -eq 0 ] && info "last supervisor message: $(cut -c1-140 "$state/last_msg")"
if [ "$running" -eq 0 ]; then
  busy=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
  info "GPU memory used by compute processes now: ${busy} MiB"
fi

# ---------------------------------------------------------------- stages
hdr "Stages"
stages="pretrain sign validate detect"
[ -n "$env_file" ] && [ -f "$env_file" ] && [ -z "${1:-}" ] && stages=$( (source "$env_file" >/dev/null 2>&1; echo "${STAGES:-$stages}") )
n_keys=$(grep -o 'n_keys=[0-9]*' "$OUTPUT_ROOT/run_config.txt" 2>/dev/null | cut -d= -f2); n_keys=${n_keys:-10}
epochs=$(grep -o -- '--epochs [0-9]*' "$OUTPUT_ROOT/run_config.txt" 2>/dev/null | head -1 | cut -d' ' -f2)
cur_stage=""
for st in $stages; do
  if [ -f "$state/$st.done" ]; then ok "$st: done ${D}($(date -r "$state/$st.done" '+%a %d %b %H:%M'))${N}"
  elif [ -z "$cur_stage" ]; then
    cur_stage=$st
    if [ "$running" -eq 1 ]; then printf '  %s▶%s %s%s%s: in progress\n' "$Y" "$N" "$B" "$st" "$N"; else printf '  %s…%s %s: next / interrupted (resumes at next window)\n' "$Y" "$N" "$st"; fi
  else info "· $st: pending"; fi
done

if [ "$cur_stage" = "sign" ] || [ -d "$OUTPUT_ROOT/poisons" ]; then
  done_keys=$(ls "$OUTPUT_ROOT"/poisons/key_*/sign_done 2>/dev/null | wc -l)
  info "  keys signed: $done_keys / $n_keys"
  [ -f "$state/pbatch_confirmed" ] && info "  pbatch: $(cat "$state/pbatch_confirmed") (confirmed)" || { [ -f "$state/pbatch" ] && info "  pbatch: $(cat "$state/pbatch") (probing)"; }
fi
if [ -n "$cur_stage" ]; then
  slog="$OUTPUT_ROOT/$cur_stage.log"
  [ "$cur_stage" = "pretrain" ] && [ -n "$epochs" ] && info "  target: $epochs epochs"
  if [ -f "$slog" ]; then
    bar=$(last_bar "$slog"); [ -n "$bar" ] && info "  latest progress: $bar"
    info "  log updated $(age $((now - $(stat -c %Y "$slog")))) ago"
  fi
  case "$cur_stage" in pretrain) d="$OUTPUT_ROOT/pretrained/ImageNet1k_*_sa/seed0" ;; validate) d="$OUTPUT_ROOT/validation" ;; *) d="" ;; esac
  # shellcheck disable=SC2086
  for dd in $d; do [ -f "$dd/checkpoint.pth" ] && info "  checkpoint saved $(age $((now - $(stat -c %Y "$dd/checkpoint.pth")))) ago ($(du -h "$dd/checkpoint.pth" | cut -f1))"
    [ -f "$dd/epoch_seconds.txt" ] && info "  epoch time: $(cat "$dd/epoch_seconds.txt")s"; done
  # exception in the current stage output
  exc=$(ls "$OUTPUT_ROOT"/pretrained/*/seed0/exception "$OUTPUT_ROOT"/poisons/key_*/exception "$OUTPUT_ROOT"/validation/exception 2>/dev/null | head -3)
  [ -n "$exc" ] && bad "exception marker(s): $exc"
fi
if [ -f "$state/detect.done" ] && [ -f "$OUTPUT_ROOT/detect.log" ]; then
  echo; info "detect result:"; tail -8 "$OUTPUT_ROOT/detect.log" | sed 's/^/      /'
fi

# ---------------------------------------------------------------- timing
hdr "Timing"
if [ -f "$OUTPUT_ROOT/timing.txt" ]; then
  attempts=$(grep -c '^start:' "$OUTPUT_ROOT/timing.txt")
  total=$(grep -E '^[a-z]+: [0-9]+s \(exit' "$OUTPUT_ROOT/timing.txt" | sed -E 's/^[a-z]+: ([0-9]+)s.*/\1/' | paste -sd+ | bc 2>/dev/null || echo 0)
  info "attempts: $attempts, compute so far (finished stage attempts): $(age "${total:-0}")"
  grep -E '^[a-z]+: [0-9]+s \(exit' "$OUTPUT_ROOT/timing.txt" | awk '{t[$1]+=$2+0; n[$1]++} END {for (s in t) printf "      %-9s %6ds in %d attempt(s)\n", s, t[s], n[s]}'
  grep -E 'stopped at the GPU deadline|failed' "$OUTPUT_ROOT/timing.txt" | tail -3 | sed 's/^/      last: /'
else
  info "no timing.txt yet"
fi

# ---------------------------------------------------------------- logs
hdr "Last supervisor.log lines"
tail -6 "$OUTPUT_ROOT/supervisor.log" 2>/dev/null | cut -c1-170 | sed 's/^/  /' || info "(none)"
if [ -f "$OUTPUT_ROOT/pipeline_run.log" ] && [ "$fails" -gt 0 ]; then
  hdr "Last pipeline_run.log lines (failures > 0)"
  tail -8 "$OUTPUT_ROOT/pipeline_run.log" | tr '\r' '\n' | tail -8 | cut -c1-170 | sed 's/^/  /'
fi
echo
