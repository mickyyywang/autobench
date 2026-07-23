#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/evaluate_all_view_graph_manifest_closed_loop.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/evaluate_new_dashscope_models_view_graph_closed_loop_parallel.sh

Runs deepseek-v4-pro and glm-5.2 concurrently over the main intervention
manifest batch. Both models use the DashScope OpenAI-compatible endpoint and
DASHSCOPE_API_KEY through the existing qwen provider.

The underlying evaluator's environment variables are forwarded unchanged,
including MANIFEST_FILTER, CONDITIONS, EVALUATION_DIR, HISTORY_WINDOW, DRY_RUN,
FAIL_FAST, and STOP_ON_ERROR. MODEL_FILTER is managed by this scheduler.

Optional scheduler environment variable:
  LOG_DIR  Per-model log directory (default: timestamped directory under
           logs/closed_loop_manifest_new_dashscope_models)

Example:
  scripts/evaluate_new_dashscope_models_view_graph_closed_loop_parallel.sh

  DRY_RUN=1 MANIFEST_FILTER=化妆品收纳B_1 \
    scripts/evaluate_new_dashscope_models_view_graph_closed_loop_parallel.sh
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

if [[ ! -x "${BASE_SCRIPT}" ]]; then
  echo "Base evaluation script is missing or not executable: ${BASE_SCRIPT}" >&2
  exit 1
fi

run_id="$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/closed_loop_manifest_new_dashscope_models/${run_id}}"
mkdir -p "${LOG_DIR}"

MODELS=(
  "deepseek-v4-pro"
  "glm-5.2"
)

run_model() {
  local model="$1"
  local log_file="${LOG_DIR}/${model}.log"
  local status

  echo "START ${model} -> ${log_file}"
  if MODEL_FILTER="${model}" "${BASE_SCRIPT}" >"${log_file}" 2>&1; then
    echo "DONE  ${model}"
    return 0
  else
    status=$?
    echo "FAIL  ${model} (exit ${status}) -> ${log_file}" >&2
    return "${status}"
  fi
}

declare -a model_pids=()
declare -A model_by_pid=()
failed_models=0

echo "Logs: ${LOG_DIR}"
echo "Scheduling: ${#MODELS[@]} DashScope models concurrently"

for model in "${MODELS[@]}"; do
  run_model "${model}" &
  pid=$!
  model_pids+=("${pid}")
  model_by_pid["${pid}"]="${model}"
done

for pid in "${model_pids[@]}"; do
  if ! wait "${pid}"; then
    failed_models=$((failed_models + 1))
    echo "DashScope worker failed: ${model_by_pid[${pid}]}" >&2
  fi
done

echo "Finished all model workers; failures: ${failed_models}."
echo "Logs: ${LOG_DIR}"
if [[ "${failed_models}" -gt 0 ]]; then
  exit 1
fi
