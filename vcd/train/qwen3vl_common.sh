#!/usr/bin/env bash

PROJECT_ROOT="${PROJECT_ROOT:-/home/storage/wenbinxing/Mybenchmark}"
CONDA_ENV="${CONDA_ENV:-videoproject}"
QWEN_CHECKPOINT_ROOT="${QWEN_CHECKPOINT_ROOT:-/home/storage/wenbinxing/checkpoints}"

GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,2,7}"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -r -a _qwen3vl_gpus <<< "${GPU_LIST}"
  NPROC_PER_NODE="${#_qwen3vl_gpus[@]}"
fi

resolve_qwen3vl_config() {
  MODEL_NAME="${1:-Qwen3-VL-8B-Instruct}"

  case "${MODEL_NAME}" in
    Qwen3-VL-2B-Instruct)
      MODEL_SLUG="qwen3vl_2b"
      DEFAULT_EMBED_DIM="2048"
      DEFAULT_TOOLS_EMBEDDINGS="${PROJECT_ROOT}/vcd_new/tools/tools_embeddings_qwen3vl_2b.pkl"
      ;;
    Qwen3-VL-4B-Instruct)
      MODEL_SLUG="qwen3vl_4b"
      DEFAULT_EMBED_DIM="2560"
      DEFAULT_TOOLS_EMBEDDINGS="${PROJECT_ROOT}/vcd_new/tools/tools_embeddings_qwen3vl_4b.pkl"
      ;;
    Qwen3-VL-8B-Instruct)
      MODEL_SLUG="qwen3vl_8b"
      DEFAULT_EMBED_DIM="4096"
      DEFAULT_TOOLS_EMBEDDINGS="${PROJECT_ROOT}/vcd_new/tools/tools_embeddings_qwen3vl.pkl"
      ;;
    Qwen3-VL-32B-Thinking)
      MODEL_SLUG="qwen3vl_32b_thinking"
      DEFAULT_EMBED_DIM="5120"
      DEFAULT_TOOLS_EMBEDDINGS="${PROJECT_ROOT}/vcd_new/tools/tools_embeddings_qwen3vl_32b_thinking.pkl"
      ;;
    *)
      MODEL_SLUG="$(printf '%s' "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_//; s/_$//')"
      DEFAULT_EMBED_DIM=""
      DEFAULT_TOOLS_EMBEDDINGS="${PROJECT_ROOT}/vcd_new/tools/tools_embeddings_${MODEL_SLUG}.pkl"
      ;;
  esac

  MODEL_DIR="${MODEL_DIR:-${QWEN_CHECKPOINT_ROOT}/${MODEL_NAME}}"
  TOOLS_EMBEDDINGS="${TOOLS_EMBEDDINGS:-${DEFAULT_TOOLS_EMBEDDINGS}}"

  if [[ -z "${EMBED_DIM:-}" ]]; then
    if [[ -n "${DEFAULT_EMBED_DIM}" ]]; then
      EMBED_DIM="${DEFAULT_EMBED_DIM}"
    else
      EMBED_DIM="$(
        python -c 'import json,sys; cfg=json.load(open(sys.argv[1])); print(cfg.get("text_config", {}).get("hidden_size") or cfg.get("hidden_size"))' \
          "${MODEL_DIR}/config.json"
      )"
    fi
  fi
}

validate_qwen3vl_inputs() {
  if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "Missing model directory: ${MODEL_DIR}" >&2
    exit 1
  fi

  if [[ ! -f "${TOOLS_EMBEDDINGS}" ]]; then
    echo "Missing tool embeddings: ${TOOLS_EMBEDDINGS}" >&2
    echo "Build them with:" >&2
    echo "  CUDA_VISIBLE_DEVICES=${GPU_LIST} bash ${PROJECT_ROOT}/vcd_new/tools/build_qwen3vl_tool_embeddings.sh ${MODEL_NAME}" >&2
    exit 1
  fi

  local actual_dim
  actual_dim="$(
    python -c 'import pickle,sys; obj=pickle.load(open(sys.argv[1],"rb")); v=next(iter(obj.values())) if isinstance(obj,dict) else obj[0]; print(v.shape[-1])' \
      "${TOOLS_EMBEDDINGS}"
  )"
  if [[ "${actual_dim}" != "${EMBED_DIM}" ]]; then
    echo "Tool embedding dim mismatch: expected ${EMBED_DIM}, got ${actual_dim} from ${TOOLS_EMBEDDINGS}" >&2
    echo "Rebuild tool embeddings for ${MODEL_NAME}." >&2
    exit 1
  fi
}

print_qwen3vl_config() {
  echo "PROJECT_ROOT=${PROJECT_ROOT}"
  echo "CONDA_ENV=${CONDA_ENV}"
  echo "CUDA_VISIBLE_DEVICES=${GPU_LIST}"
  echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
  echo "MODEL_NAME=${MODEL_NAME}"
  echo "MODEL_DIR=${MODEL_DIR}"
  echo "EMBED_DIM=${EMBED_DIM}"
  echo "TOOLS_EMBEDDINGS=${TOOLS_EMBEDDINGS}"
  echo "RUN_ID=${RUN_ID}"
  echo "RUN_DIR=${RUN_DIR}"
}
