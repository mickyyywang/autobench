#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/evaluate_all_view_graph_manifest_closed_loop.sh

Runs every intervention manifest against every configured model in
visible_graph_only + no-valid-actions mode. Each manifest condition is an
independent rollout reset from its aligned episode's initial view graph.

Optional environment variables:
  PYTHON_BIN, ENV_FILE, MANIFEST_DIR, EVALUATION_DIR,
  MODEL_FILTER (comma separated), MANIFEST_FILTER (comma separated episode ids),
  CONDITIONS, MAX_STEPS, HISTORY_WINDOW, MAX_CONSECUTIVE_MODEL_ERRORS,
  SOFT_OPTIMAL_BETA, FAIL_FAST, STOP_ON_ERROR, DRY_RUN.

Examples:
  DRY_RUN=1 scripts/evaluate_all_view_graph_manifest_closed_loop.sh
  MODEL_FILTER=qwen3.6-plus scripts/evaluate_all_view_graph_manifest_closed_loop.sh
  MANIFEST_FILTER=化妆品收纳B_1,整理餐桌A_3 \
    scripts/evaluate_all_view_graph_manifest_closed_loop.sh
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

PYTHON_BIN="${PYTHON_BIN:-/home/yufeng/miniconda3/envs/emb/bin/python}"
ENV_FILE="${ENV_FILE:-/home/wmq/project/.env}"
MANIFEST_DIR="${MANIFEST_DIR:-${PROJECT_DIR}/exp/intervention_manifests}"
EVALUATION_DIR="${EVALUATION_DIR:-${PROJECT_DIR}/evaluations/closed_loop}"
MODEL_FILTER="${MODEL_FILTER:-}"
MANIFEST_FILTER="${MANIFEST_FILTER:-}"
CONDITIONS="${CONDITIONS:-all}"
MAX_STEPS="${MAX_STEPS:-100}"
HISTORY_WINDOW="${HISTORY_WINDOW:-8}"
MAX_CONSECUTIVE_MODEL_ERRORS="${MAX_CONSECUTIVE_MODEL_ERRORS:-3}"
SOFT_OPTIMAL_BETA="${SOFT_OPTIMAL_BETA:-1.0}"
FAIL_FAST="${FAIL_FAST:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"

for boolean_name in FAIL_FAST STOP_ON_ERROR DRY_RUN; do
  boolean_value="${!boolean_name}"
  if [[ "${boolean_value}" != "0" && "${boolean_value}" != "1" ]]; then
    echo "${boolean_name} must be 0 or 1." >&2
    exit 2
  fi
done
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable does not exist or is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -d "${MANIFEST_DIR}" ]]; then
  echo "Manifest directory does not exist: ${MANIFEST_DIR}" >&2
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

MODEL_SPECS=(
  "qwen3.6-plus|qwen|DASHSCOPE_API_KEY"
  "gpt-5.5|mr_openai|MR_API_KEY"
  "gpt-5.4-2026-03-05|mr_openai|MR_API_KEY"
  "claude-opus-4-7|mr_anthropic|MR_API_KEY"
  "gemini-3.1-pro-preview|mr_google|MR_API_KEY"
)

matches_csv_filter() {
  local candidate="$1"
  local filter="$2"
  local selected
  if [[ -z "${filter}" ]]; then
    return 0
  fi
  IFS=',' read -r -a selected_values <<< "${filter}"
  for selected in "${selected_values[@]}"; do
    if [[ "${selected}" == "${candidate}" ]]; then
      return 0
    fi
  done
  return 1
}

shopt -s nullglob
ALL_MANIFESTS=("${MANIFEST_DIR}"/*_intervention_manifest.json)
shopt -u nullglob
MANIFESTS=()
for manifest in "${ALL_MANIFESTS[@]}"; do
  episode_id="$(jq -r '.source.episode_id // empty' "${manifest}")"
  if [[ -z "${episode_id}" ]]; then
    echo "Manifest has no source.episode_id: ${manifest}" >&2
    exit 1
  fi
  if matches_csv_filter "${episode_id}" "${MANIFEST_FILTER}"; then
    MANIFESTS+=("${manifest}")
  fi
done
if [[ "${#MANIFESTS[@]}" -eq 0 ]]; then
  echo "No intervention manifests matched MANIFEST_FILTER=${MANIFEST_FILTER}" >&2
  exit 1
fi

SELECTED_MODEL_SPECS=()
for model_spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r model provider api_key_env <<< "${model_spec}"
  if ! matches_csv_filter "${model}" "${MODEL_FILTER}"; then
    continue
  fi
  if [[ "${DRY_RUN}" != "1" && -z "${!api_key_env:-}" ]]; then
    echo "Missing ${api_key_env}; cannot run ${model}." >&2
    exit 1
  fi
  SELECTED_MODEL_SPECS+=("${model_spec}")
done
if [[ "${#SELECTED_MODEL_SPECS[@]}" -eq 0 ]]; then
  echo "No model matched MODEL_FILTER=${MODEL_FILTER}" >&2
  exit 2
fi

mkdir -p "${EVALUATION_DIR}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

total_commands=$((${#MANIFESTS[@]} * ${#SELECTED_MODEL_SPECS[@]}))
command_index=0
failed_count=0
echo "Planned commands: ${total_commands}"
echo "Each command runs CONDITIONS=${CONDITIONS} with no valid actions."

for manifest in "${MANIFESTS[@]}"; do
  episode_id="$(jq -r '.source.episode_id' "${manifest}")"
  condition_count="$(jq '[.conditions[] | select(.eligible != false)] | length' "${manifest}")"
  for model_spec in "${SELECTED_MODEL_SPECS[@]}"; do
    IFS='|' read -r model provider api_key_env <<< "${model_spec}"
    model_name="${model}_no_valid_action"
    args=(
      evaluate-view-graph-intervention-manifest
      --manifest "${manifest}"
      --output-dir "${EVALUATION_DIR}"
      --conditions "${CONDITIONS}"
      --provider "${provider}"
      --model "${model}"
      --model-name "${model_name}"
      --api-key-env "${api_key_env}"
      --timeout-seconds 300
      --temperature 0
      --max-api-attempts 8
      --retry-backoff-seconds 10
      --retry-max-seconds 60
      --no-valid-actions
      --soft-optimal-beta "${SOFT_OPTIMAL_BETA}"
      --max-steps "${MAX_STEPS}"
      --history-window "${HISTORY_WINDOW}"
      --max-consecutive-model-errors "${MAX_CONSECUTIVE_MODEL_ERRORS}"
    )
    if [[ "${FAIL_FAST}" == "1" ]]; then
      args+=(--fail-fast)
    fi

    command_index=$((command_index + 1))
    echo
    echo "[${command_index}/${total_commands}] ${episode_id} | ${model_name} | ${condition_count} conditions"
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '%q ' "${PYTHON_BIN}" -m auto_embodied_task "${args[@]}"
      printf '\n'
      continue
    fi
    if ! "${PYTHON_BIN}" -m auto_embodied_task "${args[@]}"; then
      failed_count=$((failed_count + 1))
      echo "Evaluation command failed: ${episode_id} | ${model_name}" >&2
      if [[ "${STOP_ON_ERROR}" == "1" ]]; then
        exit 1
      fi
    fi
  done
done

echo
echo "Finished ${total_commands} command(s); failures: ${failed_count}."
echo "Results: ${EVALUATION_DIR}"
if [[ "${failed_count}" -gt 0 ]]; then
  exit 1
fi
