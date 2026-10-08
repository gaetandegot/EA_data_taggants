#!/usr/bin/env bash
# ImageNet reference workflow. Run from the repository root. Each stage is expensive.
set -euo pipefail
stage="${1:-}"
case "$stage" in pretrain|sign|validate|detect|config) ;; *) echo 'Usage: bash scripts/reproduce.sh {pretrain|sign|validate|detect|config}' >&2; exit 2 ;; esac
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
# PBATCH=auto: start at PBATCH_START and halve on CUDA OOM during the first signing iteration
PBATCH="${PBATCH:-auto}"
PBATCH_START="${PBATCH_START:-256}"
# Flags that define the experiment ("config"); the hash of `reproduce.sh config` names the run.
cfg_common=(--dataset ImageNet1k --input_size 224
            --model "$MODEL" --drop 0 --drop_path 0.05 --eval_crop_ratio 1.0
            --smoothing 0 --world_size 1)
cfg_training=(--opt "$OPT" --epochs "$EPOCHS" --batch_size 256 --lr "$LR"
              --weight-decay 0.02 --min-lr 1e-5 --warmup-epochs 0
              --model_seed 0 --data_seed 1 --dataset_seed 2)
pretrain_args=(--aa '' --color_jitter 0 --mixup 0 --cutmix 0)
validate_args=(--model_seed 3 --aa 3a --color_jitter 0.3 --mixup 0.8 --cutmix 1)
sign_args=(--aa '' --color_jitter 0 --diff_aug --mixup 0 --cutmix 0 --targets_seed 0
           --data_seed 1 --dataset_seed 2 --sign_models "$MODEL"
           --poison_type attractors --eps 16 --budget "$BUDGET_PER_KEY"
           --n_targets "$N_KEYS" --n_targets_classes "$N_KEYS" --n_origin_classes "$N_KEYS"
           --attackoptim signAdam --attackiter 250 --poison_init randn
           --tau 0.1 --restarts 1 --target_criterion cross-entropy
           --resample_per_iter 2 --lambda_perc 0.01
           --sign_weight_decay 0 --gamma 0 --pre_dir_suff sa)
# Machine/runtime settings that do not change the results
common=("${cfg_common[@]}" --data_path "$DATA_PATH" --num_workers "$WORKERS")
training=("${cfg_training[@]}" --reserve_gpu_mem "$RESERVE_GPU_MEM")
case "$stage" in
  config)  # canonical, machine-independent description of the experiment
    echo "n_keys=$N_KEYS key_targets=random-x{i}-s64"
    echo "common: ${cfg_common[*]}"
    echo "training: ${cfg_training[*]}"
    echo "pretrain: ${pretrain_args[*]}"
    echo "sign: ${sign_args[*]}"
    if [ "$PBATCH" = auto ]; then echo "pbatch: auto (start $PBATCH_START, halved on OOM)"; else echo "pbatch: $PBATCH"; fi
    echo "validate: ${validate_args[*]}"
    ;;
  pretrain)
    "$PYTHON" pretrain.py "${common[@]}" "${training[@]}" \
      "${pretrain_args[@]}" \
      --output_dir "$OUTPUT_ROOT/pretrained/ImageNet1k_${MODEL}_sa/seed0"
    ;;
  sign)
    state="$OUTPUT_ROOT/state"; mkdir -p "$state"
    auto=0; pb="$PBATCH"
    if [ "$PBATCH" = auto ]; then auto=1; pb=$(cat "$state/pbatch" 2>/dev/null || echo "$PBATCH_START"); fi
    for ((key=0; key<N_KEYS; key++)); do
      kdir="$OUTPUT_ROOT/poisons/key_${key}"
      # a finished key is skipped on relaunch (sign.py exits 75 when stopped by the GPU deadline)
      [ -f "$kdir/sign_done" ] && continue
      # a saved poisons.pth means a whole first iteration (incl. LPIPS) fitted in memory with this pbatch
      if [ "$auto" -eq 1 ] && [ -f "$kdir/poisons.pth" ]; then echo "$pb" > "$state/pbatch_confirmed"; fi
      while true; do
        [ "$auto" -eq 1 ] && echo "$pb" > "$state/pbatch"
        set +e
        "$PYTHON" sign.py "${common[@]}" "${sign_args[@]}" --pbatch "$pb" --poison_seed "$key" \
          --targets_type "random-x${key}-s64" \
          --pretrained_dir "$OUTPUT_ROOT/pretrained" \
          --output_dir "$kdir" 2>&1 | tee "$state/sign_last.log"
        rc=${PIPESTATUS[0]}
        set -e
        [ "$rc" -eq 0 ] && break
        # probe phase only: OOM before any poisons.pth exists and before a pbatch was confirmed
        if [ "$auto" -eq 1 ] && [ "$pb" -gt 1 ] && [ ! -f "$state/pbatch_confirmed" ] \
           && [ ! -f "$kdir/poisons.pth" ] && grep -q "CUDA out of memory" "$state/sign_last.log"; then
          pb=$((pb / 2))
          echo "[pbatch] out of memory in the first iteration: retrying with --pbatch $pb"
          # the saved args_0.json would override the new --pbatch on resume
          rm -f "$kdir/args_0.json" "$kdir/exception" "$kdir/running" "$kdir/finished"
          continue
        fi
        exit "$rc"
      done
      [ "$auto" -eq 1 ] && echo "$pb" > "$state/pbatch_confirmed"
      touch "$kdir/sign_done"
    done
    ;;
  validate)
    paths=()
    for ((key=0; key<N_KEYS; key++)); do paths+=("$OUTPUT_ROOT/poisons/key_${key}"); done
    joined=$(IFS=,; echo "${paths[*]}")
    "$PYTHON" validate.py "${common[@]}" "${training[@]}" "${validate_args[@]}" \
      --poisons_path "$joined" --output_dir "$OUTPUT_ROOT/validation"
    ;;
  detect)
    targets=()
    for ((key=0; key<N_KEYS; key++)); do targets+=("$OUTPUT_ROOT/poisons/key_${key}/targets.pth"); done
    "$PYTHON" detect.py --checkpoint "$OUTPUT_ROOT/validation/checkpoint.pth" \
      --targets "${targets[@]}" --top-k 10
    ;;
esac
