#!/usr/bin/env bash
# ImageNet reference workflow. Run from the repository root. Each stage is expensive.
set -euo pipefail
stage="${1:-}"
case "$stage" in pretrain|sign|validate|detect) ;; *) echo 'Usage: bash scripts/reproduce.sh {pretrain|sign|validate|detect}' >&2; exit 2 ;; esac
: "${DATA_PATH:?Set DATA_PATH to the generated ImageNet index directory}"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-deit_small_patch16_LS}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/reference}"
OPT="${OPT:-fusedlamb}"
LR="${LR:-0.003}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-100}"
N_KEYS="${N_KEYS:-10}"
BUDGET_PER_KEY="${BUDGET_PER_KEY:-0.0001}"
RESERVE_GPU_MEM="${RESERVE_GPU_MEM:-0}"
PBATCH="${PBATCH:-128}"
common=(--dataset ImageNet1k --data_path "$DATA_PATH" --input_size 224
        --model "$MODEL" --drop 0 --drop_path 0.05 --eval_crop_ratio 1.0
        --num_workers "$WORKERS" --smoothing 0 --world_size 1)
training=(--opt "$OPT" --epochs "$EPOCHS" --batch_size 256 --lr "$LR"
          --weight-decay 0.02 --min-lr 1e-5 --warmup-epochs 0
          --model_seed 0 --data_seed 1 --dataset_seed 2
          --reserve_gpu_mem "$RESERVE_GPU_MEM")
case "$stage" in
  pretrain)
    "$PYTHON" pretrain.py "${common[@]}" "${training[@]}" \
      --aa '' --color_jitter 0 --mixup 0 --cutmix 0 \
      --output_dir "$OUTPUT_ROOT/pretrained/ImageNet1k_${MODEL}_sa/seed0"
    ;;
  sign)
    for ((key=0; key<N_KEYS; key++)); do
      "$PYTHON" sign.py "${common[@]}" --aa '' --color_jitter 0 --diff_aug \
        --mixup 0 --cutmix 0 --targets_seed 0 --poison_seed "$key" \
        --data_seed 1 --dataset_seed 2 --sign_models "$MODEL" \
        --poison_type attractors --eps 16 --budget "$BUDGET_PER_KEY" \
        --targets_type "random-x${key}-s64" --n_targets "$N_KEYS" \
        --n_targets_classes "$N_KEYS" --n_origin_classes "$N_KEYS" \
        --attackoptim signAdam --attackiter 250 --poison_init randn \
        --tau 0.1 --restarts 1 --target_criterion cross-entropy \
        --pbatch "$PBATCH" --resample_per_iter 2 --lambda_perc 0.01 \
        --sign_weight_decay 0 --gamma 0 \
        --pretrained_dir "$OUTPUT_ROOT/pretrained" --pre_dir_suff sa \
        --output_dir "$OUTPUT_ROOT/poisons/key_${key}"
    done
    ;;
  validate)
    paths=()
    for ((key=0; key<N_KEYS; key++)); do paths+=("$OUTPUT_ROOT/poisons/key_${key}"); done
    joined=$(IFS=,; echo "${paths[*]}")
    "$PYTHON" validate.py "${common[@]}" "${training[@]}" --model_seed 3 \
      --aa 3a --color_jitter 0.3 --mixup 0.8 --cutmix 1 \
      --poisons_path "$joined" --output_dir "$OUTPUT_ROOT/validation"
    ;;
  detect)
    targets=()
    for ((key=0; key<N_KEYS; key++)); do targets+=("$OUTPUT_ROOT/poisons/key_${key}/targets.pth"); done
    "$PYTHON" detect.py --checkpoint "$OUTPUT_ROOT/validation/checkpoint.pth" \
      --targets "${targets[@]}" --top-k 10
    ;;
esac
