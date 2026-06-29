#!/usr/bin/env bash
# Foreground launcher for the Qwen2.5-14B-Instruct generation server.
# Bound to 127.0.0.1:8001. bfloat16. Outlines guided-decoding backend.
#
# Run with:   ./scripts/serve_generation.sh
# Or in bg:   nohup ./scripts/serve_generation.sh > /dev/null 2>&1 &
#
# Exits non-zero on launch failure. Does NOT auto-restart; the
# operator follows the runbook on a crash.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOG_DIR}/generation-${RUN_ID}.log"
PIDFILE="${ROOT}/generation.pid"

# Reject a stale launch when something else is on the port.
if lsof -ti tcp:8001 >/dev/null 2>&1; then
  echo "[serve_generation] port 8001 already in use; run kill_vllm.sh generation first" >&2
  exit 1
fi

# Activate the vLLM server venv (see runbook for setup).
VLLM_VENV="${VLLM_VENV:-${ROOT}/venv}"
if [[ ! -f "${VLLM_VENV}/bin/activate" ]]; then
  echo "[serve_generation] vLLM venv missing at ${VLLM_VENV}; see runbook" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${VLLM_VENV}/bin/activate"

MODEL="${WHENRICH_GENERATION_MODEL_REPO:-Qwen/Qwen2.5-14B-Instruct}"
SERVED_NAME="${WHENRICH_GENERATION_MODEL:-qwen2.5-14b-instruct}"
DTYPE="${WHENRICH_GENERATION_DTYPE:-bfloat16}"
MAX_MODEL_LEN="${WHENRICH_GENERATION_MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${WHENRICH_GENERATION_GPU_MEM_UTIL:-0.90}"
GUIDED_BACKEND="${WHENRICH_GUIDED_DECODING_BACKEND:-outlines}"

{
  echo "[serve_generation] starting at ${RUN_ID}"
  echo "[serve_generation] vllm: $(python -c 'import vllm; print(vllm.__version__)')"
  echo "[serve_generation] model: ${MODEL}"
  echo "[serve_generation] served-name: ${SERVED_NAME}"
  echo "[serve_generation] dtype: ${DTYPE}"
  echo "[serve_generation] max-model-len: ${MAX_MODEL_LEN}"
  echo "[serve_generation] gpu-memory-utilization: ${GPU_MEM_UTIL}"
  echo "[serve_generation] guided-decoding-backend: ${GUIDED_BACKEND}"
  echo "[serve_generation] port: 8001"
} | tee -a "${LOG}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name "${SERVED_NAME}" \
  --host 127.0.0.1 \
  --port 8001 \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --guided-decoding-backend "${GUIDED_BACKEND}" \
  --disable-log-requests \
  >> "${LOG}" 2>&1 &

PID=$!
echo "${PID}" > "${PIDFILE}"
echo "[serve_generation] pid=${PID} log=${LOG}"
wait "${PID}"
