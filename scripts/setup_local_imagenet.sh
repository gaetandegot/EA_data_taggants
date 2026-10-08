#!/usr/bin/env bash
# Runs ON a target machine: download ImageNet-1k (HF parquet) and extract it to the same
# ImageFolder layout as the reference machine, then verify file counts.
# The HF token is read from the file given as $1 (deleted immediately), never from argv.
# Resumable: HF download and extraction both skip what already exists.
set -euo pipefail
DEST="${DEST:-/Data/gaetan.degot/imagenet}"
STAGE="${STAGE:-/Data/gaetan.degot/imagenet-hf}"
REPO_ID="${REPO_ID:-ILSVRC/imagenet-1k}"
repo="$(cd "$(dirname "$0")/.." && pwd)"
export HF_HUB_ENABLE_HF_TRANSFER=0

tokfile="${1:?usage: setup_local_imagenet.sh <token-file>}"
HF_TOKEN="$(head -n1 "$tokfile")"; rm -f "$tokfile"; export HF_TOKEN
[ -n "$HF_TOKEN" ] || { echo "empty HF token" >&2; exit 1; }

mkdir -p "$DEST" "$STAGE"; chmod 700 "$(dirname "$DEST")"
# train+val+test output (~160 GB) plus the largest parquet split (~140 GB) at peak
need_gb=320; avail_gb=$(df -BG --output=avail "$DEST" | tail -1 | tr -dc 0-9)
echo "Free space on $(dirname "$DEST"): ${avail_gb}G (need ~${need_gb}G if starting from scratch)"
[ "$avail_gb" -ge "$need_gb" ] || echo "WARNING: low disk space" >&2

cd "$repo"
run() { uv run --quiet --no-sync "$@"; }

run python - <<PY
from huggingface_hub import hf_hub_download
hf_hub_download("$REPO_ID", "classes.py", repo_type="dataset", local_dir="$STAGE")
PY
cp -n "$STAGE/classes.py" "$DEST/classes.py"

for pair in train:train validation:val test:test; do
  split="${pair%%:*}"; out="${pair##*:}"
  echo "=== $split: download ==="
  run python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$REPO_ID", repo_type="dataset", local_dir="$STAGE", allow_patterns=["data/$split-*.parquet"], max_workers=8)
PY
  echo "=== $split: extract ==="
  run python tools/extract_hf_imagenet.py --src "$STAGE" --output "$DEST" --splits "$split"
  rm -rf "$STAGE"/data/"$split"-*.parquet   # free space once the split is extracted
done

echo "=== verify counts ==="
ok=1
for spec in train:1281167 val:50000 test:100000; do
  s="${spec%%:*}"; want="${spec##*:}"; got=$(find "$DEST/$s" -type f | wc -l)
  echo "$s: $got (expected $want)"; [ "$got" -eq "$want" ] || ok=0
done
if [ "$ok" = 1 ]; then rm -rf "$STAGE"; echo "SETUP_OK"; else echo "SETUP_FAILED: count mismatch" >&2; exit 1; fi
