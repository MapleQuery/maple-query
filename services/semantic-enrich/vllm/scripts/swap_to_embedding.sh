#!/usr/bin/env bash
# Tear down the generation server, wait for VRAM to clear, start
# the embedding server in the background. Idempotent: safe to run
# when generation is already down.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
GENERATION_PID="${ROOT}/generation.pid"

# Step 1: stop generation if running.
if [[ -f "${GENERATION_PID}" ]]; then
  PID="$(cat "${GENERATION_PID}")"
  if kill -0 "${PID}" 2>/dev/null; then
    echo "[swap_to_embedding] SIGTERM to generation pid ${PID}"
    kill -TERM "${PID}"
    for _ in $(seq 1 30); do
      if ! kill -0 "${PID}" 2>/dev/null; then break; fi
      sleep 1
    done
    if kill -0 "${PID}" 2>/dev/null; then
      echo "[swap_to_embedding] SIGTERM ignored after 30s; SIGKILL"
      kill -KILL "${PID}" || true
      sleep 2
    fi
  fi
  rm -f "${GENERATION_PID}"
fi

# Belt: clean up any orphan still on :8001.
lsof -ti tcp:8001 | xargs -r kill -9 2>/dev/null || true

# Step 2: wait for VRAM to actually clear. CUDA frees lazily and
# subprocess teardown doesn't guarantee the next process sees a
# clean card. Poll nvidia-smi until used VRAM drops below 2 GB.
"${HERE}/wait_for_vram_clear.sh" 2048 60

# Step 3: launch embedding server in background.
nohup "${HERE}/serve_embedding.sh" > /dev/null 2>&1 &

# Step 4: wait for it to be ready.
"${HERE}/wait_for_ready.sh" 8002 120

echo "[swap_to_embedding] embedding server ready on :8002"
