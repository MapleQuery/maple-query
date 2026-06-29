#!/usr/bin/env bash
# Killswitch. Stops generation, embedding, or both. Used when a
# server is wedged or to clean up after a crash.
#
# Usage:  ./kill_vllm.sh [generation|embedding|all]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
TARGET="${1:-all}"

kill_one() {
  local name="$1"
  local pidfile="${ROOT}/${name}.pid"
  if [[ -f "${pidfile}" ]]; then
    local pid
    pid="$(cat "${pidfile}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" || true
      for _ in $(seq 1 10); do
        if ! kill -0 "${pid}" 2>/dev/null; then break; fi
        sleep 1
      done
      kill -KILL "${pid}" 2>/dev/null || true
    fi
    rm -f "${pidfile}"
  fi
  # Belt and braces: any orphaned process bound to the known ports.
  case "${name}" in
    generation) lsof -ti tcp:8001 | xargs -r kill -9 2>/dev/null || true ;;
    embedding)  lsof -ti tcp:8002 | xargs -r kill -9 2>/dev/null || true ;;
  esac
}

case "${TARGET}" in
  generation) kill_one generation ;;
  embedding)  kill_one embedding ;;
  all)        kill_one generation; kill_one embedding ;;
  *)
    echo "usage: $0 [generation|embedding|all]" >&2
    exit 2
    ;;
esac

echo "[kill_vllm] ${TARGET} stopped"
