#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=/Data/titouan.kerogues/data-taggants-runs
# Smoke test: 12 signed images, 1 random key, 3 gradient-matching steps.
python sign.py \
  --dataset MNIST --model mnist_cnn --sign_models mnist_cnn \
  --input_size 28 --data_path /Data/titouan.kerogues/data/mnist \
  --pretrained_dir "$ROOT/mnist-baseline/checkpoint.pth" \
  --output_dir "$ROOT/mnist-sign-smoke" \
  --model_seed 0 --dataset_seed 0 --data_seed 0 \
  --targets_seed 10 --poison_seed 11 \
  --targets_type random --n_targets 1 --n_targets_classes 1 \
  --poison_type attractors --eps 16 --budget 0.0002 \
  --attackiter 3 --restarts 1 --pbatch 12 --tau 0.1 \
  --poison_init rand --attackoptim signAdam \
  --resample_per_iter 1 --lambda_perc 0 \
  --num_workers 0 --mixup 0 --cutmix 0 --smoothing 0
