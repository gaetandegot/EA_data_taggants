#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=/Data/titouan.kerogues/data-taggants-runs
SIGN="$ROOT/mnist-sign-smoke"
[ -f "$SIGN/poisons.pth" ] || { echo "Missing $SIGN/poisons.pth" >&2; exit 1; }
[ -f "$SIGN/targets.pth" ] || { echo "Missing $SIGN/targets.pth" >&2; exit 1; }
python validate.py \
  --dataset MNIST --model mnist_cnn --input_size 28 \
  --data_path /Data/titouan.kerogues/data/mnist \
  --poisons_path "$SIGN" \
  --output_dir "$ROOT/mnist-bob-smoke" \
  --epochs 10 --batch_size 128 --num_workers 4 \
  --opt adamw --lr 0.001 --weight-decay 0 \
  --warmup-epochs 0 --mixup 0 --cutmix 0 --smoothing 0 \
  --model_seed 3 --dataset_seed 0 --data_seed 0
python detect.py \
  --checkpoint "$ROOT/mnist-bob-smoke/checkpoint.pth" \
  --targets "$SIGN/targets.pth" --top-k 1 --device cuda
