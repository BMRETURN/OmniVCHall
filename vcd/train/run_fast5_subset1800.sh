#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-fast5_subset1800_seed2025_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="/home/storage/wenbinxing/Mybenchmark"
RUN_DIR="${PROJECT_ROOT}/vcd_new/runs/${RUN_ID}"

mkdir -p "${RUN_DIR}"

export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

CUDA_VISIBLE_DEVICES=1,3,4,5,6 \
/home/wenbinxing/anaconda3/envs/videoproject/bin/torchrun \
  --nproc_per_node=5 \
  "${PROJECT_ROOT}/vcd_new/train/accelerate_train_fast5.py" \
  --distributed true \
  --dist_timeout_sec 7200 \
  --train_subset_manifest "${PROJECT_ROOT}/vcd_new/runs/splits/train_subset_1800_seed2025.json" \
  --val_limit 256 \
  --test_limit -1 \
  --epochs_max 5 \
  --epochs_min 2 \
  --early_stop_patience 2 \
  --batch_size 32 \
  --single_token_mode true \
  --max_new_tokens 1 \
  --sample_timeout_sec 180 \
  --balance_by_cost true \
  --health_check_enable true \
  --health_check_probe_frames 12 \
  --health_check_max_probe_sec 15 \
  --health_check_max_total_frames 4000 \
  --health_check_max_duration_sec 300 \
  --health_check_drop_slow true \
  --auto_resume true \
  --final_full_eval true \
  --seed 2025 \
  --run_id "${RUN_ID}" \
  --output_dir "${RUN_DIR}"
