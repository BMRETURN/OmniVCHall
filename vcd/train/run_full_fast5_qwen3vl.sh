#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/qwen3vl_common.sh"

MODEL_NAME_ARG="${1:-Qwen3-VL-8B-Instruct}"
resolve_qwen3vl_config "${MODEL_NAME_ARG}"

RUN_ID="${2:-full_fast5_${MODEL_SLUG}_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${PROJECT_ROOT}/vcd_new/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

validate_qwen3vl_inputs
print_qwen3vl_config

export NCCL_TIMEOUT="${NCCL_TIMEOUT:-14400}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
conda run -n "${CONDA_ENV}" torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  "${PROJECT_ROOT}/vcd_new/train/accelerate_train_fast5.py" \
  --distributed true \
  --dist_timeout_sec 14400 \
  --model_dir "${MODEL_DIR}" \
  --tools_embeddings "${TOOLS_EMBEDDINGS}" \
  --embed_dim "${EMBED_DIM}" \
  --train_sample_size -1 \
  --train_limit -1 \
  --val_limit -1 \
  --test_limit -1 \
  --epochs_max "${EPOCHS_MAX:-1}" \
  --epochs_min "${EPOCHS_MIN:-1}" \
  --early_stop_patience "${EARLY_STOP_PATIENCE:-1}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --single_token_mode true \
  --max_new_tokens 1 \
  --sample_timeout_sec 300 \
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
