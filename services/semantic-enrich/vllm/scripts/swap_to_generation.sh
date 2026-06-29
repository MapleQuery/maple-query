#!/usr/bin/env bash
# Symmetric counterpart to swap_to_embedding.sh: tear down the
# embedding server, wait for VRAM to clear, start the generation
# server in the background. Idempotent.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
EMBEDDING_PID="${ROOT}/embedding.pid"

# Step 1: stop embedding if running.
if [[ -f "${EMBEDDING_PID}" ]]; then
  PID="$(cat "${EMBEDDING_PID}")"
  if kill -0 "${PID}" 2>/dev/null; then
    echo "[swap_to_generation] SIGTERM to embedding pid ${PID}"
    kill -TERM "${PID}"
    for _ in $(seq 1 30); do
      if ! kill -0 "${PID}" 2>/dev/null; then break; fi
      sleep 1
    done
    if kill -0 "${PID}" 2>/dev/null; then
      echo "[swap_to_generation] SIGTERM ignored after 30s; SIGKILL"
      kill -KILL "${PID}" || true
      sleep 2
    fi
  fi
  rm -f "${EMBEDDING_PID}"
fi

lsof -ti tcp:8002 | xargs -r kill -9 2>/dev/null || true

# Embedding idle footprint is well under 1 GB; tighter threshold
# than the generation teardown.
"${HERE}/wait_for_vram_clear.sh" 1024 60

nohup "${HERE}/serve_generation.sh" > /dev/null 2>&1 &

# Generation cold-start can take 30–90s; allow plenty of headroom.
"${HERE}/wait_for_ready.sh" 8001 180

echo "[swap_to_generation] generation server ready on :8001"
