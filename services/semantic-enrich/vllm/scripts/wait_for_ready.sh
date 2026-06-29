#!/usr/bin/env bash
# Block until vLLM at <PORT> responds 200 on /v1/models.
# Exits non-zero on TIMEOUT_SECONDS exceeded.
#
# Usage:  ./wait_for_ready.sh <PORT> <TIMEOUT_SECONDS>
set -euo pipefail

PORT="${1:?port required}"
TIMEOUT_SECONDS="${2:-120}"

DEADLINE=$(( $(date +%s) + TIMEOUT_SECONDS ))
while :; do
  if curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[wait_for_ready] :${PORT} ready"
    exit 0
  fi
  if (( $(date +%s) > DEADLINE )); then
    echo "[wait_for_ready] timed out waiting for :${PORT} to become ready" >&2
    exit 1
  fi
  sleep 3
done
