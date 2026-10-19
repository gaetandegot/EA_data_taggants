#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python pretrain.py \
  --dataset MNIST --model mnist_cnn --input_size 28 \
  --data_path /Data/titouan.kerogues/data/mnist \
  --output_dir /Data/titouan.kerogues/data-taggants-runs/mnist-baseline \
  --epochs 10 --batch_size 128 --num_workers 4 \
  --opt adamw --lr 0.001 --weight-decay 0 \
  --warmup-epochs 0 --mixup 0 --cutmix 0 --smoothing 0 \
  --model_seed 0 --dataset_seed 0 --data_seed 0
