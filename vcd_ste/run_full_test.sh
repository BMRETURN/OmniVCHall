#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/storage/wenbinxing/Mybenchmark"
RUN_ID="${1:-vcd_ste_full_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${PROJECT_ROOT}/vcd_ste/runs/${RUN_ID}"

CUDA_VISIBLE_DEVICES=0,2,7 \
conda run -n videoproject python "${PROJECT_ROOT}/vcd_ste/train_ste.py" \
  --run_id "${RUN_ID}" \
  --epochs 1 \
  --train_limit -1 \
  --tool_threshold 0.4 \
  --min_selected_tools 1 \
  --max_selected_tools -1 \
  --ste_temperature 1.0 \
  --single_token_mode true \
  --max_new_tokens 1 \
  --allow_legacy_cache true \
  --cache_root "${PROJECT_ROOT}/vcd_new/train/saliency_cache" \
  --sample_timeout_sec 300 \
  --save_every 20

CUDA_VISIBLE_DEVICES=0,2,7 \
conda run -n videoproject python "${PROJECT_ROOT}/vcd_ste/eval_ste.py" \
  --run_id "${RUN_ID}" \
  --split test \
  --limit -1 \
  --checkpoint_path "${RUN_DIR}/checkpoints/ste_latest.pth" \
  --tool_threshold 0.4 \
  --min_selected_tools 1 \
  --max_selected_tools -1 \
  --ste_temperature 1.0 \
  --single_token_mode true \
  --max_new_tokens 1 \
  --allow_legacy_cache true \
  --cache_root "${PROJECT_ROOT}/vcd_new/train/saliency_cache" \
  --sample_timeout_sec 300 \
  --resume true \
  --save_every 50

