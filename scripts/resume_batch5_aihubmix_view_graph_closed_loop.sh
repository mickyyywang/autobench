#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMMON_RUNNER="${SCRIPT_DIR}/resume_batch356_aihubmix_view_graph_closed_loop.sh"

if [[ ! -x "${COMMON_RUNNER}" ]]; then
  echo "Common AIHubMix resume runner is missing or not executable: ${COMMON_RUNNER}" >&2
  exit 1
fi

exec env BATCH_FILTER=5 "${COMMON_RUNNER}" "$@"
