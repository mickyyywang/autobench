#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SCHEDULER="${SCRIPT_DIR}/evaluate_new_dashscope_models_view_graph_closed_loop_parallel.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/evaluate_batch2_new_dashscope_models_view_graph_closed_loop.sh

Runs deepseek-v4-pro and glm-5.2 concurrently over the second intervention
manifest batch. Results and logs are separated from the main batch.

Useful overrides:
  DRY_RUN=1                Print all evaluator commands without model calls
  CONDITIONS=baseline,... Run selected manifest conditions (default: all)
  MANIFEST_FILTER=id,...   Run selected batch-2 episodes
  MAX_STEPS=100            Closed-loop action limit

Example:
  DRY_RUN=1 scripts/evaluate_batch2_new_dashscope_models_view_graph_closed_loop.sh
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

export MANIFEST_DIR="${MANIFEST_DIR:-${PROJECT_DIR}/exp/intervention_manifests_2}"
export EVALUATION_DIR="${EVALUATION_DIR:-${PROJECT_DIR}/evaluations/closed_loop_batch2}"
export LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/closed_loop_manifest_batch2_new_dashscope_models/$(date +%Y%m%d_%H%M%S)_$$}"

exec "${SCHEDULER}"
