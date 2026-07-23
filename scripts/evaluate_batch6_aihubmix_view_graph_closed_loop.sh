#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BASE_RUNNER="${SCRIPT_DIR}/evaluate_batch3_aihubmix_view_graph_closed_loop.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/evaluate_batch6_aihubmix_view_graph_closed_loop.sh

Runs qwen3.7-plus through AIHubMix on all main-batch and batch-2
intervention manifests. The two source groups and all manifests are run in
strict serial order. Manifests are read in place; no batch-6 copies are made.

Defaults:
  MAIN_MANIFEST_DIR=exp/intervention_manifests
  BATCH2_MANIFEST_DIR=exp/intervention_manifests_2
  EVALUATION_DIR=evaluations/closed_loop_batch6_aihubmix
  BATCH6_GROUPS=main,batch2

Optional environment variables:
  BATCH6_GROUPS (main,batch2 or either one), MANIFEST_FILTER, CONDITIONS,
  MAX_STEPS, HISTORY_WINDOW, MAX_CONSECUTIVE_MODEL_ERRORS,
  SOFT_OPTIMAL_BETA, MAX_OUTPUT_TOKENS, FAIL_FAST, STOP_ON_ERROR,
  DRY_RUN, RESUME, ENV_FILE, PYTHON_BIN.

Connectivity controls:
  AIHUBMIX_SKIP_PREFLIGHT=1       Skip endpoint/model preflight.
  AIHUBMIX_PREFLIGHT_TIMEOUT=15   Preflight timeout in seconds.

Examples:
  DRY_RUN=1 scripts/evaluate_batch6_aihubmix_view_graph_closed_loop.sh

  scripts/evaluate_batch6_aihubmix_view_graph_closed_loop.sh

  BATCH6_GROUPS=batch2 \
    scripts/evaluate_batch6_aihubmix_view_graph_closed_loop.sh

  CONDITIONS=baseline \
    scripts/evaluate_batch6_aihubmix_view_graph_closed_loop.sh
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

if [[ ! -x "${BASE_RUNNER}" ]]; then
  echo "Base AIHubMix runner is missing or not executable: ${BASE_RUNNER}" >&2
  exit 1
fi

MAIN_MANIFEST_DIR="${MAIN_MANIFEST_DIR:-${PROJECT_DIR}/exp/intervention_manifests}"
BATCH2_MANIFEST_DIR="${BATCH2_MANIFEST_DIR:-${PROJECT_DIR}/exp/intervention_manifests_2}"
EVALUATION_DIR="${EVALUATION_DIR:-${PROJECT_DIR}/evaluations/closed_loop_batch6_aihubmix}"
BATCH6_GROUPS="${BATCH6_GROUPS:-main,batch2}"
ENV_FILE="${ENV_FILE:-/home/wmq/project/.env}"
AIHUBMIX_BASE_URL="${AIHUBMIX_BASE_URL:-https://aihubmix.com/v1}"
AIHUBMIX_SKIP_PREFLIGHT="${AIHUBMIX_SKIP_PREFLIGHT:-0}"
AIHUBMIX_PREFLIGHT_TIMEOUT="${AIHUBMIX_PREFLIGHT_TIMEOUT:-15}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
RESUME="${RESUME:-0}"

for boolean_name in AIHUBMIX_SKIP_PREFLIGHT DRY_RUN STOP_ON_ERROR RESUME; do
  boolean_value="${!boolean_name}"
  if [[ "${boolean_value}" != "0" && "${boolean_value}" != "1" ]]; then
    echo "${boolean_name} must be 0 or 1." >&2
    exit 2
  fi
done

if [[ ! -d "${MAIN_MANIFEST_DIR}" ]]; then
  echo "Main manifest directory does not exist: ${MAIN_MANIFEST_DIR}" >&2
  exit 1
fi
if [[ ! -d "${BATCH2_MANIFEST_DIR}" ]]; then
  echo "Batch-2 manifest directory does not exist: ${BATCH2_MANIFEST_DIR}" >&2
  exit 1
fi
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

IFS=',' read -r -a requested_groups <<< "${BATCH6_GROUPS}"
for group in "${requested_groups[@]}"; do
  if [[ "${group}" != "main" && "${group}" != "batch2" ]]; then
    echo "Unknown BATCH6_GROUPS entry: ${group}; expected main or batch2." >&2
    exit 2
  fi
done
if ! csv_contains main "${BATCH6_GROUPS}" && ! csv_contains batch2 "${BATCH6_GROUPS}"; then
  echo "BATCH6_GROUPS selected no source groups." >&2
  exit 2
fi

if [[ "${DRY_RUN}" != "1" && "${AIHUBMIX_SKIP_PREFLIGHT}" != "1" ]]; then
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
    echo "Enable the working proxy before starting batch 6." >&2
    exit 1
  fi
  if ! jq -e '.data[]?.id | select(. == "qwen3.7-plus")' "${preflight_file}" >/dev/null; then
    echo "AIHubMix model qwen3.7-plus is not available to this account." >&2
    exit 1
  fi
  rm -f "${preflight_file}"
  trap - EXIT
fi

main_count="$(find "${MAIN_MANIFEST_DIR}" -maxdepth 1 -type f -name '*_intervention_manifest.json' | wc -l)"
batch2_count="$(find "${BATCH2_MANIFEST_DIR}" -maxdepth 1 -type f -name '*_intervention_manifest.json' | wc -l)"
selected_count=0
if csv_contains main "${BATCH6_GROUPS}"; then
  selected_count=$((selected_count + main_count))
fi
if csv_contains batch2 "${BATCH6_GROUPS}"; then
  selected_count=$((selected_count + batch2_count))
fi

echo "Batch 6 model: qwen3.7-plus via AIHubMix"
echo "Selected groups: ${BATCH6_GROUPS}"
echo "Selected manifests: ${selected_count} (main=${main_count}, batch2=${batch2_count})"
echo "Execution: strict serial, no valid actions."
echo "Results: ${EVALUATION_DIR}"

group_failures=0
run_group() {
  local group_name="$1"
  local manifest_dir="$2"
  echo
  echo "===== BATCH 6 GROUP START: ${group_name} ====="
  if ! env \
    MANIFEST_DIR="${manifest_dir}" \
    EVALUATION_DIR="${EVALUATION_DIR}" \
    MODEL_FILTER="qwen3.7-plus" \
    AIHUBMIX_BASE_URL="${AIHUBMIX_BASE_URL}" \
    ENV_FILE="${ENV_FILE}" \
    DRY_RUN="${DRY_RUN}" \
    STOP_ON_ERROR="${STOP_ON_ERROR}" \
    RESUME="${RESUME}" \
    "${BASE_RUNNER}"; then
    group_failures=$((group_failures + 1))
    echo "Batch 6 group failed: ${group_name}" >&2
    if [[ "${STOP_ON_ERROR}" == "1" ]]; then
      return 1
    fi
  fi
  echo "===== BATCH 6 GROUP END: ${group_name} ====="
}

if csv_contains main "${BATCH6_GROUPS}"; then
  run_group main "${MAIN_MANIFEST_DIR}" || exit 1
fi
if csv_contains batch2 "${BATCH6_GROUPS}"; then
  run_group batch2 "${BATCH2_MANIFEST_DIR}" || exit 1
fi

echo
echo "Batch 6 finished; failed source groups: ${group_failures}."
echo "Results: ${EVALUATION_DIR}"
if [[ "${group_failures}" -gt 0 ]]; then
  exit 1
fi
