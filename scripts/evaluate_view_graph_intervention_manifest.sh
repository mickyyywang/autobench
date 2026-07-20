#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/evaluate_view_graph_intervention_manifest.sh [all|condition_id,...]

Runs the B_1 intervention manifest as independent visible-graph-only closed-loop
rollouts. The default is all manifest conditions.

Optional environment variables:
  PYTHON_BIN, ENV_FILE, MANIFEST, EVALUATION_DIR, PROVIDER, MODEL,
  MODEL_NAME, API_KEY_ENV, INCLUDE_VALID_ACTIONS, MAX_STEPS, HISTORY_WINDOW,
  MAX_CONSECUTIVE_MODEL_ERRORS, SOFT_OPTIMAL_BETA, FAIL_FAST.

Examples:
  scripts/evaluate_view_graph_intervention_manifest.sh all
  scripts/evaluate_view_graph_intervention_manifest.sh state_regression
  scripts/evaluate_view_graph_intervention_manifest.sh completed_subgoal_rollback,wrong_container_relocation
EOF
}

case "${1:-all}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

CONDITIONS="${1:-all}"
if [[ $# -gt 1 ]]; then
  echo "Expected at most one comma-separated condition argument." >&2
  usage >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/home/yufeng/miniconda3/envs/emb/bin/python}"
ENV_FILE="${ENV_FILE:-/home/wmq/project/.env}"
MANIFEST="${MANIFEST:-${PROJECT_DIR}/exp/intervention_manifests/化妆品收纳B_1_intervention_manifest.json}"
EVALUATION_DIR="${EVALUATION_DIR:-${PROJECT_DIR}/evaluations/closed_loop}"
PROVIDER="${PROVIDER:-qwen}"
MODEL="${MODEL:-qwen3.6-plus}"
MODEL_NAME="${MODEL_NAME:-}"
API_KEY_ENV="${API_KEY_ENV:-DASHSCOPE_API_KEY}"
INCLUDE_VALID_ACTIONS="${INCLUDE_VALID_ACTIONS:-0}"
MAX_STEPS="${MAX_STEPS:-100}"
HISTORY_WINDOW="${HISTORY_WINDOW:-8}"
MAX_CONSECUTIVE_MODEL_ERRORS="${MAX_CONSECUTIVE_MODEL_ERRORS:-3}"
SOFT_OPTIMAL_BETA="${SOFT_OPTIMAL_BETA:-1.0}"
FAIL_FAST="${FAIL_FAST:-1}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable does not exist or is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Intervention manifest does not exist: ${MANIFEST}" >&2
  exit 1
fi

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
else
  echo "Warning: dotenv file not found: ${ENV_FILE}; using the current environment." >&2
fi
if [[ -z "${!API_KEY_ENV:-}" ]]; then
  echo "Missing API key environment variable: ${API_KEY_ENV}" >&2
  exit 1
fi

case "${INCLUDE_VALID_ACTIONS}" in
  0)
    VALID_ACTIONS_FLAG="--no-valid-actions"
    MODEL_NAME="${MODEL_NAME:-${MODEL}_no_valid_action}"
    ;;
  1)
    VALID_ACTIONS_FLAG="--valid-actions"
    MODEL_NAME="${MODEL_NAME:-${MODEL}_valid_action}"
    ;;
  *)
    echo "INCLUDE_VALID_ACTIONS must be 0 or 1." >&2
    exit 2
    ;;
esac
if [[ "${FAIL_FAST}" != "0" && "${FAIL_FAST}" != "1" ]]; then
  echo "FAIL_FAST must be 0 or 1." >&2
  exit 2
fi

mkdir -p "${EVALUATION_DIR}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

args=(
  evaluate-view-graph-intervention-manifest
  --manifest "${MANIFEST}"
  --output-dir "${EVALUATION_DIR}"
  --conditions "${CONDITIONS}"
  --provider "${PROVIDER}"
  --model "${MODEL}"
  --model-name "${MODEL_NAME}"
  --api-key-env "${API_KEY_ENV}"
  --timeout-seconds 300
  --temperature 0
  --max-api-attempts 8
  --retry-backoff-seconds 10
  --retry-max-seconds 60
  "${VALID_ACTIONS_FLAG}"
  --soft-optimal-beta "${SOFT_OPTIMAL_BETA}"
  --max-steps "${MAX_STEPS}"
  --history-window "${HISTORY_WINDOW}"
  --max-consecutive-model-errors "${MAX_CONSECUTIVE_MODEL_ERRORS}"
)
if [[ "${FAIL_FAST}" == "1" ]]; then
  args+=(--fail-fast)
fi

echo "Manifest:   ${MANIFEST}"
echo "Conditions: ${CONDITIONS}"
echo "Model:      ${MODEL_NAME}"
echo "Output:     ${EVALUATION_DIR} (timestamps appended automatically)"
"${PYTHON_BIN}" -m auto_embodied_task "${args[@]}"
