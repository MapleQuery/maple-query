#!/usr/bin/env bash
# Foreground launcher for the Qwen3-Embedding-0.6B embedding server.
# Bound to 127.0.0.1:8002. fp16. --task embed (no chat template).
#
# Run with:   ./scripts/serve_embedding.sh
# Or in bg:   nohup ./scripts/serve_embedding.sh > /dev/null 2>&1 &
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOG_DIR}/embedding-${RUN_ID}.log"
PIDFILE="${ROOT}/embedding.pid"

if lsof -ti tcp:8002 >/dev/null 2>&1; then
  echo "[serve_embedding] port 8002 already in use; run kill_vllm.sh embedding first" >&2
  exit 1
fi

VLLM_VENV="${VLLM_VENV:-${ROOT}/venv}"
if [[ ! -f "${VLLM_VENV}/bin/activate" ]]; then
  echo "[serve_embedding] vLLM venv missing at ${VLLM_VENV}; see runbook" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${VLLM_VENV}/bin/activate"

MODEL="${WHENRICH_EMBEDDING_MODEL_REPO:-Qwen/Qwen3-Embedding-0.6B}"
SERVED_NAME="${WHENRICH_EMBEDDING_MODEL:-qwen3-embedding-0.6b}"
DTYPE="${WHENRICH_EMBEDDING_DTYPE:-float16}"
MAX_MODEL_LEN="${WHENRICH_EMBEDDING_MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${WHENRICH_EMBEDDING_GPU_MEM_UTIL:-0.50}"

{
  echo "[serve_embedding] starting at ${RUN_ID}"
  echo "[serve_embedding] vllm: $(python -c 'import vllm; print(vllm.__version__)')"
  echo "[serve_embedding] model: ${MODEL}"
  echo "[serve_embedding] served-name: ${SERVED_NAME}"
  echo "[serve_embedding] dtype: ${DTYPE}"
  echo "[serve_embedding] max-model-len: ${MAX_MODEL_LEN}"
  echo "[serve_embedding] gpu-memory-utilization: ${GPU_MEM_UTIL}"
  echo "[serve_embedding] port: 8002"
} | tee -a "${LOG}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name "${SERVED_NAME}" \
  --host 127.0.0.1 \
  --port 8002 \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --task embed \
  --disable-log-requests \
  >> "${LOG}" 2>&1 &

PID=$!
echo "${PID}" > "${PIDFILE}"
echo "[serve_embedding] pid=${PID} log=${LOG}"
wait "${PID}"
