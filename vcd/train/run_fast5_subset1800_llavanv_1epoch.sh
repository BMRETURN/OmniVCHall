#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-fast5_subset1800_llavanv_1epoch_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="/home/storage/wenbinxing/Mybenchmark"
RUN_DIR="${PROJECT_ROOT}/vcd_new/runs/${RUN_ID}"

mkdir -p "${RUN_DIR}"

export NCCL_TIMEOUT="${NCCL_TIMEOUT:-14400}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

CUDA_VISIBLE_DEVICES=1,3,4,5,6 \
/home/wenbinxing/anaconda3/envs/videoproject/bin/torchrun \
  --nproc_per_node=5 \
  "${PROJECT_ROOT}/vcd_new/train/accelerate_train_fast5_llavanv.py" \
  --distributed true \
  --dist_timeout_sec 14400 \
  --model_dir "/home/storage/wenbinxing/checkpoints/LLaVA-NeXT-Video-7B" \
  --tools_embeddings "${PROJECT_ROOT}/vcd_new/tools/tools_embeddings_llavanv.pkl" \
  --train_subset_manifest "${PROJECT_ROOT}/vcd_new/runs/splits/train_subset_1800_seed2025.json" \
  --val_limit 256 \
  --test_limit -1 \
  --epochs_max 1 \
  --epochs_min 1 \
  --early_stop_patience 1 \
  --batch_size 32 \
  --single_token_mode true \
  --max_new_tokens 1 \
  --sample_timeout_sec 240 \
  --balance_by_cost true \
  --health_check_enable true \
  --health_check_probe_frames 12 \
  --health_check_max_probe_sec 15 \
  --health_check_max_total_frames 4000 \
  --health_check_max_duration_sec 300 \
  --health_check_drop_slow true \
  --llava_num_frames 32 \
  --auto_resume false \
  --final_full_eval true \
  --seed 2025 \
  --run_id "${RUN_ID}" \
  --output_dir "${RUN_DIR}"
