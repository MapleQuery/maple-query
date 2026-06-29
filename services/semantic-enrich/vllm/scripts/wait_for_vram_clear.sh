#!/usr/bin/env bash
# Block until nvidia-smi reports used VRAM below THRESHOLD_MB on
# GPU 0. Exits non-zero on TIMEOUT_SECONDS exceeded.
#
# Usage:  ./wait_for_vram_clear.sh <THRESHOLD_MB> <TIMEOUT_SECONDS>
set -euo pipefail

THRESHOLD_MB="${1:?threshold_mb required}"
TIMEOUT_SECONDS="${2:-60}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[wait_for_vram_clear] nvidia-smi not on PATH; cannot verify VRAM is clear" >&2
  exit 1
fi

DEADLINE=$(( $(date +%s) + TIMEOUT_SECONDS ))
while :; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if (( USED < THRESHOLD_MB )); then
    echo "[wait_for_vram_clear] VRAM=${USED}MiB < ${THRESHOLD_MB}MiB; proceeding"
    exit 0
  fi
  if (( $(date +%s) > DEADLINE )); then
    echo "[wait_for_vram_clear] timed out waiting for VRAM to drop below ${THRESHOLD_MB}MiB (current=${USED}MiB)" >&2
    exit 1
  fi
  sleep 2
done
