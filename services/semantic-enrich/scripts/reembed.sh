#!/usr/bin/env bash
# One-shot wrapper for the 4.7 OpenAI embedding swap.
#
# Runs `datasets-reembed` then `columns-reembed` end-to-end, both via
# `uv run` (this stays on the laptop — no GPU box involved).
#
# Env expected:
#   OPENAI_API_KEY (or WHENRICH_OPENAI_API_KEY)
#   GCP_PROJECT_ID (or WHENRICH_GCP_PROJECT_ID)
#
# Usage:
#   scripts/reembed.sh                 # full run
#   scripts/reembed.sh --dry-run       # preview row counts + would-have events
#   scripts/reembed.sh --datasets-only # skip the columns pass
#   scripts/reembed.sh --columns-only  # skip the datasets pass
#
# The RUN_ID is stamped to `reembed-YYYY-MM-DD-<epoch>` unless you set
# WHENRICH_RUN_ID in the environment.

set -euo pipefail

cd "$(dirname "$0")/.."

DRY_RUN=""
DATASETS=1
COLUMNS=1

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN="--dry-run" ;;
    --datasets-only) COLUMNS=0 ;;
    --columns-only) DATASETS=0 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown flag: $arg" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${OPENAI_API_KEY:-}" && -z "${WHENRICH_OPENAI_API_KEY:-}" ]]; then
  echo "error: OPENAI_API_KEY (or WHENRICH_OPENAI_API_KEY) must be set" >&2
  exit 3
fi
if [[ -z "${GCP_PROJECT_ID:-}" && -z "${WHENRICH_GCP_PROJECT_ID:-}" ]]; then
  echo "error: GCP_PROJECT_ID (or WHENRICH_GCP_PROJECT_ID) must be set" >&2
  exit 3
fi

RUN_ID="${WHENRICH_RUN_ID:-reembed-$(date +%Y-%m-%d)-$(date +%s)}"
echo "→ run id: $RUN_ID"

if [[ "$DATASETS" == "1" ]]; then
  echo "→ datasets-reembed ${DRY_RUN}"
  uv run semantic-enrich datasets-reembed --run-id "$RUN_ID" ${DRY_RUN}
fi

if [[ "$COLUMNS" == "1" ]]; then
  echo "→ columns-reembed ${DRY_RUN}"
  uv run semantic-enrich columns-reembed --run-id "$RUN_ID" ${DRY_RUN}
fi

echo "✓ done: $RUN_ID"
