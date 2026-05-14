#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/storage/wenbinxing/Mybenchmark}"
CONDA_ENV="${CONDA_ENV:-videoproject}"
QWEN_CHECKPOINT_ROOT="${QWEN_CHECKPOINT_ROOT:-/home/storage/wenbinxing/checkpoints}"
MODEL_NAME="${1:-Qwen3-VL-8B-Instruct}"

case "${MODEL_NAME}" in
  Qwen3-VL-2B-Instruct)
    MODEL_SLUG="qwen3vl_2b"
    ;;
  Qwen3-VL-4B-Instruct)
    MODEL_SLUG="qwen3vl_4b"
    ;;
  Qwen3-VL-8B-Instruct)
    MODEL_SLUG="qwen3vl"
    ;;
  Qwen3-VL-32B-Thinking)
    MODEL_SLUG="qwen3vl_32b_thinking"
    ;;
  *)
    MODEL_SLUG="$(printf '%s' "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_//; s/_$//')"
    ;;
esac

MODEL_DIR="${MODEL_DIR:-${QWEN_CHECKPOINT_ROOT}/${MODEL_NAME}}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/vcd_new/tools/tools_embeddings_${MODEL_SLUG}.pkl}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Missing model directory: ${MODEL_DIR}" >&2
  exit 1
fi

echo "CONDA_ENV=${CONDA_ENV}"
echo "CUDA_VISIBLE_DEVICES=${GPU_LIST}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"

CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/vcd_new/tools/build_qwen3vl_tool_embeddings.py" \
  --model_path "${MODEL_DIR}" \
  --tools_json "${PROJECT_ROOT}/vcd_new/tools/tools.json" \
  --output_path "${OUTPUT_PATH}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --max_length "${MAX_LENGTH:-512}" \
  --save_dtype "${SAVE_DTYPE:-float32}"
