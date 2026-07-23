#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/evaluate_batch3_aihubmix_view_graph_closed_loop.sh

Runs all batch-3 intervention manifests through AIHubMix in strict serial
model order. A model finishes all selected episodes before the next model
starts. The default priority is:
  gpt-5.5, qwen3.6-plus, qwen3.7-plus, deepseek-v4-pro,
  glm-5.2, gpt-5.4, claude-opus-4-7, gemini-3.1-pro-preview.

Optional environment variables:
  PYTHON_BIN, ENV_FILE, MANIFEST_DIR, EVALUATION_DIR,
  MODEL_FILTER (comma-separated model ids),
  MANIFEST_FILTER (comma-separated episode ids), CONDITIONS,
  MAX_STEPS, HISTORY_WINDOW, MAX_CONSECUTIVE_MODEL_ERRORS,
  SOFT_OPTIMAL_BETA, MAX_OUTPUT_TOKENS, GPT_MAX_OUTPUT_TOKENS,
  GEMINI_MAX_OUTPUT_TOKENS, FAIL_FAST, STOP_ON_ERROR, DRY_RUN,
  RESUME (1 skips conditions with a valid existing summary and JSONL).

Examples:
  DRY_RUN=1 scripts/evaluate_batch3_aihubmix_view_graph_closed_loop.sh

  MODEL_FILTER=gpt-5.5,qwen3.6-plus,qwen3.7-plus,deepseek-v4-pro \
    scripts/evaluate_batch3_aihubmix_view_graph_closed_loop.sh

  MODEL_FILTER=gpt-5.5 MANIFEST_FILTER=DivideBuffetTraysA_1 CONDITIONS=baseline \
    scripts/evaluate_batch3_aihubmix_view_graph_closed_loop.sh

  RESUME=1 MODEL_FILTER=qwen3.6-plus \
    scripts/evaluate_batch3_aihubmix_view_graph_closed_loop.sh
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
MANIFEST_DIR="${MANIFEST_DIR:-${PROJECT_DIR}/exp/intervention_manifests_3}"
EVALUATION_DIR="${EVALUATION_DIR:-${PROJECT_DIR}/evaluations/closed_loop_batch3_aihubmix}"
AIHUBMIX_BASE_URL="${AIHUBMIX_BASE_URL:-https://aihubmix.com/v1}"
MODEL_FILTER="${MODEL_FILTER:-}"
MANIFEST_FILTER="${MANIFEST_FILTER:-}"
CONDITIONS="${CONDITIONS:-all}"
MAX_STEPS="${MAX_STEPS:-100}"
HISTORY_WINDOW="${HISTORY_WINDOW:-16}"
MAX_CONSECUTIVE_MODEL_ERRORS="${MAX_CONSECUTIVE_MODEL_ERRORS:-3}"
SOFT_OPTIMAL_BETA="${SOFT_OPTIMAL_BETA:-1.0}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-2048}"
GPT_MAX_OUTPUT_TOKENS="${GPT_MAX_OUTPUT_TOKENS:-4096}"
GEMINI_MAX_OUTPUT_TOKENS="${GEMINI_MAX_OUTPUT_TOKENS:-4096}"
FAIL_FAST="${FAIL_FAST:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"

for boolean_name in FAIL_FAST STOP_ON_ERROR DRY_RUN RESUME; do
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
  echo "Dotenv file not found: ${ENV_FILE}" >&2
  exit 1
fi
if [[ -z "${AIHUBMIX_API_KEY:-}" ]]; then
  echo "Missing AIHUBMIX_API_KEY in ${ENV_FILE}." >&2
  exit 1
fi

# model|api_style|max_output_policy. Keep this order: it is the execution order.
MODEL_SPECS=(
  "gpt-5.5|chat_completions|gpt"
  "qwen3.6-plus|chat_completions|default"
  "qwen3.7-plus|chat_completions|default"
  "deepseek-v4-pro|chat_completions|default"
  "glm-5.2|chat_completions|default"
  "gpt-5.4|chat_completions|default"
  "claude-opus-4-7|chat_completions|default"
  "gemini-3.1-pro-preview|chat_completions|gemini"
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
  echo "No batch-3 manifest matched MANIFEST_FILTER=${MANIFEST_FILTER}" >&2
  exit 1
fi

SELECTED_MODEL_SPECS=()
for model_spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r model api_style output_policy <<< "${model_spec}"
  if matches_csv_filter "${model}" "${MODEL_FILTER}"; then
    SELECTED_MODEL_SPECS+=("${model_spec}")
  fi
done
if [[ "${#SELECTED_MODEL_SPECS[@]}" -eq 0 ]]; then
  echo "No model matched MODEL_FILTER=${MODEL_FILTER}" >&2
  exit 2
fi

mkdir -p "${EVALUATION_DIR}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

filename_slug() {
  local value="$1"
  value="${value//[^[:alnum:]]/_}"
  while [[ "${value}" == _* ]]; do
    value="${value#_}"
  done
  while [[ "${value}" == *_ ]]; do
    value="${value%_}"
  done
  printf '%s' "${value:-unnamed}"
}

condition_has_valid_result() {
  local episode_id="$1"
  local model="$2"
  local condition_id="$3"
  local episode_slug
  local model_slug
  local condition_slug
  local summary_path
  local jsonl_path
  local -a summary_paths

  episode_slug="$(filename_slug "${episode_id}")"
  model_slug="$(filename_slug "${model}")"
  condition_slug="$(filename_slug "${condition_id}")"
  shopt -s nullglob
  summary_paths=(
    "${EVALUATION_DIR}"/closed_loop_eval_"${episode_slug}"_"${model_slug}"_no_valid_action_"${condition_slug}"_*__summary.json
  )
  shopt -u nullglob
  for summary_path in "${summary_paths[@]}"; do
    jsonl_path="${summary_path%__summary.json}.jsonl"
    if [[ -s "${jsonl_path}" ]] && jq -e '
      (.episode_count // 0) > 0
      and ((.outcomes.termination_reasons.model_error_limit // 0) == 0)
    ' "${summary_path}" >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

total_commands=$((${#MANIFESTS[@]} * ${#SELECTED_MODEL_SPECS[@]}))
command_index=0
failed_count=0
skipped_command_count=0
skipped_condition_count=0
echo "AIHubMix endpoint: ${AIHUBMIX_BASE_URL}"
echo "Planned commands: ${total_commands}"
echo "Execution: strict serial, model-major order, no valid actions."
echo "Resume mode: ${RESUME}"
echo "Results: ${EVALUATION_DIR}"

for model_spec in "${SELECTED_MODEL_SPECS[@]}"; do
  IFS='|' read -r model api_style output_policy <<< "${model_spec}"
  model_max_output_tokens="${MAX_OUTPUT_TOKENS}"
  if [[ "${output_policy}" == "gpt" ]]; then
    model_max_output_tokens="${GPT_MAX_OUTPUT_TOKENS}"
  elif [[ "${output_policy}" == "gemini" ]]; then
    model_max_output_tokens="${GEMINI_MAX_OUTPUT_TOKENS}"
  fi
  model_name="${model}_aihubmix_no_valid_action"
  echo
  echo "===== MODEL START: ${model_name} (${#MANIFESTS[@]} episodes) ====="

  for manifest in "${MANIFESTS[@]}"; do
    episode_id="$(jq -r '.source.episode_id' "${manifest}")"
    requested_condition_ids=()
    if [[ -z "${CONDITIONS}" || "${CONDITIONS}" == "all" ]]; then
      mapfile -t requested_condition_ids < <(
        jq -r '.conditions[] | select(.eligible != false) | .condition_id' "${manifest}"
      )
    else
      IFS=',' read -r -a requested_condition_ids <<< "${CONDITIONS}"
    fi
    conditions_for_command="${CONDITIONS}"
    pending_condition_ids=("${requested_condition_ids[@]}")
    if [[ "${RESUME}" == "1" ]]; then
      pending_condition_ids=()
      for condition_id in "${requested_condition_ids[@]}"; do
        if condition_has_valid_result "${episode_id}" "${model}" "${condition_id}"; then
          skipped_condition_count=$((skipped_condition_count + 1))
        else
          pending_condition_ids+=("${condition_id}")
        fi
      done
      if [[ "${#pending_condition_ids[@]}" -gt 0 ]]; then
        conditions_for_command="$(
          IFS=','
          printf '%s' "${pending_condition_ids[*]}"
        )"
      fi
    fi
    condition_count="${#pending_condition_ids[@]}"
    command_index=$((command_index + 1))
    if [[ "${RESUME}" == "1" && "${condition_count}" -eq 0 ]]; then
      skipped_command_count=$((skipped_command_count + 1))
      echo "[${command_index}/${total_commands}] SKIP ${episode_id} | ${model_name} | all selected conditions already valid"
      continue
    fi

    args=(
      evaluate-view-graph-intervention-manifest
      --manifest "${manifest}"
      --output-dir "${EVALUATION_DIR}"
      --conditions "${conditions_for_command}"
      --provider compatible
      --model "${model}"
      --model-name "${model_name}"
      --api-key-env AIHUBMIX_API_KEY
      --api-base-url "${AIHUBMIX_BASE_URL}"
      --api-style "${api_style}"
      --timeout-seconds 300
      --temperature 0
      --max-output-tokens "${model_max_output_tokens}"
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

    echo "[${command_index}/${total_commands}] ${episode_id} | ${model_name} | ${condition_count} condition(s): ${conditions_for_command}"
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
      continue
    fi
    if [[ "${RESUME}" == "1" ]]; then
      unresolved_condition_ids=()
      for condition_id in "${pending_condition_ids[@]}"; do
        if ! condition_has_valid_result "${episode_id}" "${model}" "${condition_id}"; then
          unresolved_condition_ids+=("${condition_id}")
        fi
      done
      if [[ "${#unresolved_condition_ids[@]}" -gt 0 ]]; then
        failed_count=$((failed_count + 1))
        unresolved_csv="$(
          IFS=','
          printf '%s' "${unresolved_condition_ids[*]}"
        )"
        echo "Evaluation produced no valid result: ${episode_id} | ${model_name} | ${unresolved_csv}" >&2
        if [[ "${STOP_ON_ERROR}" == "1" ]]; then
          exit 1
        fi
      fi
    fi
  done
  echo "===== MODEL END: ${model_name} ====="
done

echo
echo "Finished ${total_commands} planned command(s); skipped commands: ${skipped_command_count}; skipped valid conditions: ${skipped_condition_count}; command failures: ${failed_count}."
echo "Results: ${EVALUATION_DIR}"
if [[ "${failed_count}" -gt 0 ]]; then
  exit 1
fi
