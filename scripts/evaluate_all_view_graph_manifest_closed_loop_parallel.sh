#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/evaluate_all_view_graph_manifest_closed_loop.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/evaluate_all_view_graph_manifest_closed_loop_parallel.sh

Runs qwen3.6-plus and qwen3.7-plus concurrently as independent processes while
the four MR-backed models run in a separate worker pool. MR_PARALLELISM defaults
to 2, so the overall maximum is two Qwen plus two MR model processes.

The underlying evaluator's environment variables are forwarded unchanged,
including MANIFEST_FILTER, CONDITIONS, EVALUATION_DIR, HISTORY_WINDOW, DRY_RUN,
FAIL_FAST, and STOP_ON_ERROR. MODEL_FILTER is managed by this scheduler.

Optional scheduler environment variables:
  MR_PARALLELISM  Maximum concurrent MR model processes (default: 2)
  LOG_DIR         Per-model log directory (default: timestamped directory under
                  logs/closed_loop_manifest_parallel)

Example:
  scripts/evaluate_all_view_graph_manifest_closed_loop_parallel.sh

  DRY_RUN=1 MANIFEST_FILTER=化妆品收纳B_1 \
    scripts/evaluate_all_view_graph_manifest_closed_loop_parallel.sh
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

MR_PARALLELISM="${MR_PARALLELISM:-2}"
if [[ ! "${MR_PARALLELISM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MR_PARALLELISM must be a positive integer." >&2
  exit 2
fi

run_id="$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/closed_loop_manifest_parallel/${run_id}}"
mkdir -p "${LOG_DIR}"

QWEN_MODELS=(
  "qwen3.6-plus"
  "qwen3.7-plus"
)
MR_MODELS=(
  "gpt-5.5"
  "gpt-5.4-2026-03-05"
  "claude-opus-4-7"
  "gemini-3.1-pro-preview"
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

declare -a mr_pids=()
declare -A mr_model_by_pid=()
declare -a qwen_pids=()
declare -A qwen_model_by_pid=()
failed_models=0

remove_mr_pid() {
  local target_pid="$1"
  local -a remaining=()
  local pid
  for pid in "${mr_pids[@]}"; do
    if [[ "${pid}" != "${target_pid}" ]]; then
      remaining+=("${pid}")
    fi
  done
  mr_pids=("${remaining[@]}")
}

wait_for_one_mr_model() {
  local finished_pid=""
  local status=0
  local model

  if wait -n -p finished_pid "${mr_pids[@]}"; then
    status=0
  else
    status=$?
  fi
  model="${mr_model_by_pid[${finished_pid}]:-unknown}"
  if [[ "${status}" -ne 0 ]]; then
    failed_models=$((failed_models + 1))
    echo "MR worker failed: ${model} (exit ${status})" >&2
  fi
  unset 'mr_model_by_pid[${finished_pid}]'
  remove_mr_pid "${finished_pid}"
}

echo "Logs: ${LOG_DIR}"
echo "Scheduling: ${#QWEN_MODELS[@]} Qwen models concurrently + MR concurrency ${MR_PARALLELISM}"

for model in "${QWEN_MODELS[@]}"; do
  run_model "${model}" &
  pid=$!
  qwen_pids+=("${pid}")
  qwen_model_by_pid["${pid}"]="${model}"
done

for model in "${MR_MODELS[@]}"; do
  while [[ "${#mr_pids[@]}" -ge "${MR_PARALLELISM}" ]]; do
    wait_for_one_mr_model
  done
  run_model "${model}" &
  pid=$!
  mr_pids+=("${pid}")
  mr_model_by_pid["${pid}"]="${model}"
done

while [[ "${#mr_pids[@]}" -gt 0 ]]; do
  wait_for_one_mr_model
done

for pid in "${qwen_pids[@]}"; do
  if ! wait "${pid}"; then
    failed_models=$((failed_models + 1))
    echo "Qwen worker failed: ${qwen_model_by_pid[${pid}]}" >&2
  fi
done

echo "Finished all model workers; failures: ${failed_models}."
echo "Logs: ${LOG_DIR}"
if [[ "${failed_models}" -gt 0 ]]; then
  exit 1
fi
