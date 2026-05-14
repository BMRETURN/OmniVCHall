#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-fast5_subset1800_autorestart_$(date +%Y%m%d_%H%M%S)}"
MAX_RESTARTS="${MAX_RESTARTS:-12}"
RETRY_SLEEP_SEC="${RETRY_SLEEP_SEC:-20}"
PROJECT_ROOT="/home/storage/wenbinxing/Mybenchmark"
RUN_DIR="${PROJECT_ROOT}/vcd_new/runs/${RUN_ID}"
LOG_DIR="${RUN_DIR}/autorestart"
LOG_FILE="${LOG_DIR}/launcher.log"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

export NCCL_TIMEOUT="${NCCL_TIMEOUT:-14400}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[$(date '+%F %T')] attempt=${attempt}/${MAX_RESTARTS} run_id=${RUN_ID}" | tee -a "${LOG_FILE}"

  set +e
  CUDA_VISIBLE_DEVICES=1,3,4,5,6 \
  /home/wenbinxing/anaconda3/envs/videoproject/bin/torchrun \
    --nproc_per_node=5 \
    "${PROJECT_ROOT}/vcd_new/train/accelerate_train_fast5.py" \
    --distributed true \
    --dist_timeout_sec 14400 \
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
  rc=$?
  set -e

  if [[ ${rc} -eq 0 ]]; then
    echo "[$(date '+%F %T')] training finished successfully." | tee -a "${LOG_FILE}"
    exit 0
  fi

  echo "[$(date '+%F %T')] training exited with rc=${rc}" | tee -a "${LOG_FILE}"
  if [[ ${attempt} -ge ${MAX_RESTARTS} ]]; then
    echo "[$(date '+%F %T')] reached MAX_RESTARTS=${MAX_RESTARTS}, stop." | tee -a "${LOG_FILE}"
    exit "${rc}"
  fi

  echo "[$(date '+%F %T')] sleep ${RETRY_SLEEP_SEC}s then resume from latest checkpoint..." | tee -a "${LOG_FILE}"
  sleep "${RETRY_SLEEP_SEC}"
done
