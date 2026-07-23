#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SCHEDULER="${SCRIPT_DIR}/evaluate_all_view_graph_manifest_closed_loop_parallel.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/evaluate_batch2_view_graph_closed_loop.sh

Runs the 21 second-batch manifests. qwen3.6-plus and qwen3.7-plus run in
parallel; MR-backed models use a worker pool capped at two concurrent models.

Useful overrides:
  MR_PARALLELISM=1|2       MR concurrency (default: 2; values above 2 rejected)
  DRY_RUN=1                Print all evaluator commands without model calls
  CONDITIONS=baseline,... Run selected manifest conditions (default: all)
  MANIFEST_FILTER=id,...   Run selected batch-2 episodes
  MAX_STEPS=100            Closed-loop action limit

Example:
  DRY_RUN=1 scripts/evaluate_batch2_view_graph_closed_loop.sh
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  "") ;;
  *)
    echo "This script accepts configuration through environment variables only." >&2
    usage >&2
    exit 2
    ;;
esac

MR_PARALLELISM="${MR_PARALLELISM:-2}"
if [[ "${MR_PARALLELISM}" != "1" && "${MR_PARALLELISM}" != "2" ]]; then
  echo "MR_PARALLELISM must be 1 or 2 for batch 2." >&2
  exit 2
fi

export MR_PARALLELISM
export MANIFEST_DIR="${MANIFEST_DIR:-${PROJECT_DIR}/exp/intervention_manifests_2}"
export EVALUATION_DIR="${EVALUATION_DIR:-${PROJECT_DIR}/evaluations/closed_loop_batch2}"
export LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/closed_loop_manifest_batch2/$(date +%Y%m%d_%H%M%S)_$$}"

exec "${SCHEDULER}"
