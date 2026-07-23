#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BASE_RUNNER="${SCRIPT_DIR}/evaluate_batch3_aihubmix_view_graph_closed_loop.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/evaluate_batch5_aihubmix_view_graph_closed_loop.sh

Runs all batch-5 intervention manifests through AIHubMix in strict serial
model order. The default model order is:
  gpt-5.5, qwen3.6-plus, qwen3.7-plus, deepseek-v4-pro,
  glm-5.2, gpt-5.4, claude-opus-4-7, gemini-3.1-pro-preview.

Defaults:
  MANIFEST_DIR=exp/intervention_manifests_5
  EVALUATION_DIR=evaluations/closed_loop_batch5_aihubmix

Optional environment variables are the same as the batch-3 runner:
  MODEL_FILTER, MANIFEST_FILTER, CONDITIONS, MAX_STEPS, HISTORY_WINDOW,
  MAX_CONSECUTIVE_MODEL_ERRORS, SOFT_OPTIMAL_BETA, MAX_OUTPUT_TOKENS,
  GPT_MAX_OUTPUT_TOKENS, GEMINI_MAX_OUTPUT_TOKENS, FAIL_FAST,
  STOP_ON_ERROR, DRY_RUN, RESUME, ENV_FILE, PYTHON_BIN.

Batch-5-specific connectivity controls:
  AIHUBMIX_SKIP_PREFLIGHT=1       Skip the endpoint connectivity check.
  AIHUBMIX_PREFLIGHT_TIMEOUT=10   Connectivity timeout in seconds.

Examples:
  DRY_RUN=1 scripts/evaluate_batch5_aihubmix_view_graph_closed_loop.sh

  scripts/evaluate_batch5_aihubmix_view_graph_closed_loop.sh

  MODEL_FILTER=gpt-5.5,qwen3.6-plus,qwen3.7-plus,deepseek-v4-pro \
    scripts/evaluate_batch5_aihubmix_view_graph_closed_loop.sh

  MODEL_FILTER=gpt-5.5 CONDITIONS=baseline \
    scripts/evaluate_batch5_aihubmix_view_graph_closed_loop.sh
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

export MANIFEST_DIR="${MANIFEST_DIR:-${PROJECT_DIR}/exp/intervention_manifests_5}"
export EVALUATION_DIR="${EVALUATION_DIR:-${PROJECT_DIR}/evaluations/closed_loop_batch5_aihubmix}"
export AIHUBMIX_BASE_URL="${AIHUBMIX_BASE_URL:-https://aihubmix.com/v1}"

DRY_RUN="${DRY_RUN:-0}"
AIHUBMIX_SKIP_PREFLIGHT="${AIHUBMIX_SKIP_PREFLIGHT:-0}"
AIHUBMIX_PREFLIGHT_TIMEOUT="${AIHUBMIX_PREFLIGHT_TIMEOUT:-10}"
for boolean_name in DRY_RUN AIHUBMIX_SKIP_PREFLIGHT; do
  boolean_value="${!boolean_name}"
  if [[ "${boolean_value}" != "0" && "${boolean_value}" != "1" ]]; then
    echo "${boolean_name} must be 0 or 1." >&2
    exit 2
  fi
done

if [[ "${DRY_RUN}" != "1" && "${AIHUBMIX_SKIP_PREFLIGHT}" != "1" ]]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required for the AIHubMix connectivity preflight." >&2
    echo "Set AIHUBMIX_SKIP_PREFLIGHT=1 only if connectivity is already verified." >&2
    exit 1
  fi
  if ! curl -sS -o /dev/null \
    --connect-timeout "${AIHUBMIX_PREFLIGHT_TIMEOUT}" \
    --max-time "${AIHUBMIX_PREFLIGHT_TIMEOUT}" \
    "${AIHUBMIX_BASE_URL}/models"; then
    echo "Cannot connect to ${AIHUBMIX_BASE_URL}." >&2
    echo "Enable the working proxy before starting this batch; the evaluator process must inherit it." >&2
    exit 1
  fi
fi

exec "${BASE_RUNNER}"
