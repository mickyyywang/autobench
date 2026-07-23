#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BATCH3_RUNNER="${SCRIPT_DIR}/evaluate_batch3_aihubmix_view_graph_closed_loop.sh"
BATCH5_RUNNER="${SCRIPT_DIR}/evaluate_batch5_aihubmix_view_graph_closed_loop.sh"
BATCH6_RUNNER="${SCRIPT_DIR}/evaluate_batch6_aihubmix_view_graph_closed_loop.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/resume_batch356_aihubmix_view_graph_closed_loop.sh

Resumes valid-result-aware AIHubMix evaluation for batches 3, 5, and 6.
Selected batches run concurrently; work inside each batch remains serial.
Existing conditions are skipped only when they have a non-empty JSONL and a
summary that did not terminate at model_error_limit.

Recovery model set for batches 3 and 5:
  gpt-5.5, qwen3.6-plus, qwen3.7-plus, deepseek-v4-pro, glm-5.2, gpt-5.4

claude-opus-4-7 and gemini-3.1-pro-preview are intentionally excluded.
Batch 6 continues to use only qwen3.7-plus.

Optional environment variables:
  BATCH_FILTER       Comma-separated batch ids: 3,5,6. Required when this
                     common runner is invoked directly.
  DRY_RUN            1 prints the pending commands without running (default: 0)
  STOP_ON_ERROR      Stop after a command fails validation (default: 1)
  LOG_DIR            Per-batch log directory (default: a timestamped directory)
  ENV_FILE, PYTHON_BIN, AIHUBMIX_BASE_URL
  AIHUBMIX_SKIP_PREFLIGHT, AIHUBMIX_PREFLIGHT_TIMEOUT
  MAX_STEPS, HISTORY_WINDOW, MAX_CONSECUTIVE_MODEL_ERRORS,
  SOFT_OPTIMAL_BETA, MAX_OUTPUT_TOKENS, GPT_MAX_OUTPUT_TOKENS,
  FAIL_FAST.

Examples:
  BATCH_FILTER=3,5 scripts/resume_batch356_aihubmix_view_graph_closed_loop.sh

For separate terminals and logs, prefer:
  scripts/resume_batch3_aihubmix_view_graph_closed_loop.sh
  scripts/resume_batch5_aihubmix_view_graph_closed_loop.sh
  scripts/resume_batch6_aihubmix_view_graph_closed_loop.sh
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

BATCH_FILTER="${BATCH_FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
ENV_FILE="${ENV_FILE:-/home/wmq/project/.env}"
AIHUBMIX_BASE_URL="${AIHUBMIX_BASE_URL:-https://aihubmix.com/v1}"
AIHUBMIX_SKIP_PREFLIGHT="${AIHUBMIX_SKIP_PREFLIGHT:-0}"
AIHUBMIX_PREFLIGHT_TIMEOUT="${AIHUBMIX_PREFLIGHT_TIMEOUT:-15}"
RECOVERY_MODELS="gpt-5.5,qwen3.6-plus,qwen3.7-plus,deepseek-v4-pro,glm-5.2,gpt-5.4"
run_id="$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/aihubmix_batch356_resume/${run_id}}"

for boolean_name in DRY_RUN STOP_ON_ERROR AIHUBMIX_SKIP_PREFLIGHT; do
  boolean_value="${!boolean_name}"
  if [[ "${boolean_value}" != "0" && "${boolean_value}" != "1" ]]; then
    echo "${boolean_name} must be 0 or 1." >&2
    exit 2
  fi
done

csv_contains() {
  local candidate="$1"
  local csv="$2"
  local item
  local -a items
  IFS=',' read -r -a items <<< "${csv}"
  for item in "${items[@]}"; do
    if [[ "${item}" == "${candidate}" ]]; then
      return 0
    fi
  done
  return 1
}

IFS=',' read -r -a requested_batches <<< "${BATCH_FILTER}"
if [[ "${#requested_batches[@]}" -eq 0 ]]; then
  echo "BATCH_FILTER selected no batches. Use one of the separate batch3, batch5, or batch6 resume scripts." >&2
  exit 2
fi
for batch in "${requested_batches[@]}"; do
  if [[ "${batch}" != "3" && "${batch}" != "5" && "${batch}" != "6" ]]; then
    echo "Unknown BATCH_FILTER entry: ${batch}; expected 3, 5, or 6." >&2
    exit 2
  fi
done

for runner in "${BATCH3_RUNNER}" "${BATCH5_RUNNER}" "${BATCH6_RUNNER}"; do
  if [[ ! -x "${runner}" ]]; then
    echo "Required runner is missing or not executable: ${runner}" >&2
    exit 1
  fi
done

if [[ "${DRY_RUN}" != "1" && "${AIHUBMIX_SKIP_PREFLIGHT}" != "1" ]]; then
  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Dotenv file not found: ${ENV_FILE}" >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
  if [[ -z "${AIHUBMIX_API_KEY:-}" ]]; then
    echo "Missing AIHUBMIX_API_KEY in ${ENV_FILE}." >&2
    exit 1
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required for the AIHubMix connectivity preflight." >&2
    exit 1
  fi
  preflight_file="$(mktemp)"
  trap 'rm -f "${preflight_file:-}"' EXIT
  preflight_code="$(curl -sS -o "${preflight_file}" -w '%{http_code}' \
    --connect-timeout "${AIHUBMIX_PREFLIGHT_TIMEOUT}" \
    --max-time "${AIHUBMIX_PREFLIGHT_TIMEOUT}" \
    -H "Authorization: Bearer ${AIHUBMIX_API_KEY}" \
    "${AIHUBMIX_BASE_URL}/models")" || preflight_code="000"
  if [[ "${preflight_code}" != "200" ]]; then
    echo "AIHubMix preflight failed with HTTP ${preflight_code}: ${AIHUBMIX_BASE_URL}" >&2
    exit 1
  fi
  rm -f "${preflight_file}"
  trap - EXIT
fi

run_batch() {
  local batch="$1"
  local runner="$2"
  local model_filter="$3"
  echo
  echo "===== RESUME BATCH ${batch} ====="
  if ! env \
    RESUME=1 \
    MODEL_FILTER="${model_filter}" \
    DRY_RUN="${DRY_RUN}" \
    STOP_ON_ERROR="${STOP_ON_ERROR}" \
    ENV_FILE="${ENV_FILE}" \
    AIHUBMIX_BASE_URL="${AIHUBMIX_BASE_URL}" \
    AIHUBMIX_SKIP_PREFLIGHT=1 \
    "${runner}"; then
    echo "Batch ${batch} resume failed." >&2
    return 1
  fi
  echo "===== RESUME BATCH ${batch} COMPLETE ====="
}

mkdir -p "${LOG_DIR}"
echo "Scheduling: selected batches run concurrently; each batch runs serially."
echo "Logs: ${LOG_DIR}"

declare -a batch_pids=()
declare -A batch_by_pid=()
declare -A log_by_pid=()

launch_batch() {
  local batch="$1"
  local runner="$2"
  local model_filter="$3"
  local log_file="${LOG_DIR}/batch${batch}.log"
  local pid

  echo "START batch ${batch} -> ${log_file}"
  (
    set -o pipefail
    run_batch "${batch}" "${runner}" "${model_filter}" 2>&1 \
      | sed -u "s/^/[batch ${batch}] /" \
      | tee "${log_file}"
  ) &
  pid=$!
  batch_pids+=("${pid}")
  batch_by_pid["${pid}"]="${batch}"
  log_by_pid["${pid}"]="${log_file}"
}

failed_batches=0
if csv_contains 3 "${BATCH_FILTER}"; then
  launch_batch 3 "${BATCH3_RUNNER}" "${RECOVERY_MODELS}"
fi
if csv_contains 5 "${BATCH_FILTER}"; then
  launch_batch 5 "${BATCH5_RUNNER}" "${RECOVERY_MODELS}"
fi
if csv_contains 6 "${BATCH_FILTER}"; then
  launch_batch 6 "${BATCH6_RUNNER}" "qwen3.7-plus"
fi

for pid in "${batch_pids[@]}"; do
  if wait "${pid}"; then
    echo "DONE batch ${batch_by_pid[${pid}]} -> ${log_by_pid[${pid}]}"
  else
    failed_batches=$((failed_batches + 1))
    echo "FAIL batch ${batch_by_pid[${pid}]} -> ${log_by_pid[${pid}]}" >&2
  fi
done

echo
echo "Resume launcher finished; failed batches: ${failed_batches}."
echo "Excluded models: claude-opus-4-7, gemini-3.1-pro-preview"
echo "Logs: ${LOG_DIR}"
if [[ "${failed_batches}" -gt 0 ]]; then
  exit 1
fi
