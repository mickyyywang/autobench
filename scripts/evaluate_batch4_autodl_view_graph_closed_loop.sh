#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/evaluate_batch4_autodl_view_graph_closed_loop.sh

Runs the batch-4 intervention manifests through AutoDL in strict serial,
model-major order. The benchmark's original eight-model set is used after
excluding qwen3.7-plus and gemini-3.1-pro-preview:
  gpt-5.5, qwen3.6-plus, DeepSeek-V4-Pro, glm-5.2, gpt-5.4.

Before evaluation, the script sends one minimal chat request per selected
model. Models explicitly reported as unavailable are skipped by default;
authentication, connectivity, rate-limit, and other endpoint errors abort the
batch. This avoids writing evaluation records for nonexistent AutoDL models.
DeepSeek-V4-Pro is called without the unsupported response_format parameter;
the benchmark prompt still requires the same JSON response schema.

Defaults:
  MANIFEST_DIR=exp/intervention_manifests_4
  EVALUATION_DIR=evaluations/closed_loop_batch4_autodl
  CONDITIONS=all
  MAX_STEPS=100
  HISTORY_WINDOW=16

Optional environment variables:
  PYTHON_BIN, ENV_FILE, MANIFEST_DIR, EVALUATION_DIR,
  MODEL_FILTER (comma-separated exact model ids),
  MANIFEST_FILTER (comma-separated episode ids), CONDITIONS,
  MAX_STEPS, HISTORY_WINDOW, MAX_CONSECUTIVE_MODEL_ERRORS,
  SOFT_OPTIMAL_BETA, MAX_OUTPUT_TOKENS, GPT_MAX_OUTPUT_TOKENS,
  FAIL_FAST, STOP_ON_ERROR, DRY_RUN.

AutoDL controls:
  AUTODL_API_BASE_URL=https://www.autodl.art/api/v1
  AUTODL_SKIP_MODEL_PREFLIGHT=0
  AUTODL_UNAVAILABLE_POLICY=skip   # skip or fail
  AUTODL_PREFLIGHT_TIMEOUT=30

Examples:
  DRY_RUN=1 scripts/evaluate_batch4_autodl_view_graph_closed_loop.sh

  scripts/evaluate_batch4_autodl_view_graph_closed_loop.sh

  MODEL_FILTER=gpt-5.5,gpt-5.4 \
    scripts/evaluate_batch4_autodl_view_graph_closed_loop.sh

  MANIFEST_FILTER=LoadCondimentsInFridgeA_4 CONDITIONS=baseline \
    scripts/evaluate_batch4_autodl_view_graph_closed_loop.sh
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
MANIFEST_DIR="${MANIFEST_DIR:-${PROJECT_DIR}/exp/intervention_manifests_4}"
EVALUATION_DIR="${EVALUATION_DIR:-${PROJECT_DIR}/evaluations/closed_loop_batch4_autodl}"
MODEL_FILTER="${MODEL_FILTER:-}"
MANIFEST_FILTER="${MANIFEST_FILTER:-}"
CONDITIONS="${CONDITIONS:-all}"
MAX_STEPS="${MAX_STEPS:-100}"
HISTORY_WINDOW="${HISTORY_WINDOW:-16}"
MAX_CONSECUTIVE_MODEL_ERRORS="${MAX_CONSECUTIVE_MODEL_ERRORS:-3}"
SOFT_OPTIMAL_BETA="${SOFT_OPTIMAL_BETA:-1.0}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-2048}"
GPT_MAX_OUTPUT_TOKENS="${GPT_MAX_OUTPUT_TOKENS:-4096}"
FAIL_FAST="${FAIL_FAST:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"
AUTODL_SKIP_MODEL_PREFLIGHT="${AUTODL_SKIP_MODEL_PREFLIGHT:-0}"
AUTODL_UNAVAILABLE_POLICY="${AUTODL_UNAVAILABLE_POLICY:-skip}"
AUTODL_PREFLIGHT_TIMEOUT="${AUTODL_PREFLIGHT_TIMEOUT:-30}"

for boolean_name in FAIL_FAST STOP_ON_ERROR DRY_RUN AUTODL_SKIP_MODEL_PREFLIGHT; do
  boolean_value="${!boolean_name}"
  if [[ "${boolean_value}" != "0" && "${boolean_value}" != "1" ]]; then
    echo "${boolean_name} must be 0 or 1." >&2
    exit 2
  fi
done
if [[ "${AUTODL_UNAVAILABLE_POLICY}" != "skip" && "${AUTODL_UNAVAILABLE_POLICY}" != "fail" ]]; then
  echo "AUTODL_UNAVAILABLE_POLICY must be skip or fail." >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable does not exist or is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -d "${MANIFEST_DIR}" ]]; then
  echo "Manifest directory does not exist: ${MANIFEST_DIR}" >&2
  exit 1
fi
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Dotenv file not found: ${ENV_FILE}" >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required by this script." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

AUTODL_API_BASE_URL="${AUTODL_API_BASE_URL:-https://www.autodl.art/api/v1}"
AUTODL_API_BASE_URL="${AUTODL_API_BASE_URL%/}"
if [[ -z "${AUTODL_API_KEY:-}" ]]; then
  echo "Missing AUTODL_API_KEY in ${ENV_FILE}." >&2
  exit 1
fi

# model|output-token-policy|json-mode-policy. Keep this order: it is the execution order.
# qwen3.7-plus and gemini-3.1-pro-preview are intentionally excluded.
MODEL_SPECS=(
  "gpt-5.5|gpt|json_mode"
  "qwen3.6-plus|default|json_mode"
  "DeepSeek-V4-Pro|default|prompt_json"
  "glm-5.2|default|json_mode"
  "gpt-5.4|default|json_mode"
)

matches_csv_filter() {
  local candidate="$1"
  local filter="$2"
  local selected
  local -a selected_values
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
  if ! jq -e '.source.episode_id and (.conditions | type == "array")' "${manifest}" >/dev/null; then
    echo "Invalid intervention manifest: ${manifest}" >&2
    exit 1
  fi
  episode_id="$(jq -r '.source.episode_id' "${manifest}")"
  if matches_csv_filter "${episode_id}" "${MANIFEST_FILTER}"; then
    MANIFESTS+=("${manifest}")
  fi
done
if [[ "${#MANIFESTS[@]}" -eq 0 ]]; then
  echo "No batch-4 manifest matched MANIFEST_FILTER=${MANIFEST_FILTER}" >&2
  exit 1
fi

FILTERED_MODEL_SPECS=()
for model_spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r model output_policy json_mode_policy <<< "${model_spec}"
  if matches_csv_filter "${model}" "${MODEL_FILTER}"; then
    FILTERED_MODEL_SPECS+=("${model_spec}")
  fi
done
if [[ "${#FILTERED_MODEL_SPECS[@]}" -eq 0 ]]; then
  echo "No model matched MODEL_FILTER=${MODEL_FILTER}" >&2
  exit 2
fi

SELECTED_MODEL_SPECS=()
SKIPPED_MODELS=()
if [[ "${DRY_RUN}" == "1" || "${AUTODL_SKIP_MODEL_PREFLIGHT}" == "1" ]]; then
  SELECTED_MODEL_SPECS=("${FILTERED_MODEL_SPECS[@]}")
else
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required for the AutoDL model preflight." >&2
    exit 1
  fi
  preflight_response="$(mktemp)"
  preflight_request="$(mktemp)"
  trap 'rm -f "${preflight_response:-}" "${preflight_request:-}"' EXIT

  for model_spec in "${FILTERED_MODEL_SPECS[@]}"; do
    IFS='|' read -r model output_policy json_mode_policy <<< "${model_spec}"
    jq -n --arg model "${model}" --arg json_mode_policy "${json_mode_policy}" '{
      model: $model,
      messages: [{role: "user", content: "Return exactly this JSON object: {}"}],
      temperature: 0
    } + if $json_mode_policy == "json_mode"
        then {response_format: {type: "json_object"}}
        else {}
        end' > "${preflight_request}"

    echo "AutoDL model preflight: ${model}"
    preflight_code="$(curl -sS -o "${preflight_response}" -w '%{http_code}' \
      --connect-timeout "${AUTODL_PREFLIGHT_TIMEOUT}" \
      --max-time "${AUTODL_PREFLIGHT_TIMEOUT}" \
      -H "Authorization: Bearer ${AUTODL_API_KEY}" \
      -H 'Content-Type: application/json' \
      --data-binary "@${preflight_request}" \
      "${AUTODL_API_BASE_URL}/chat/completions")" || preflight_code="000"

    if [[ "${preflight_code}" == "200" ]]; then
      SELECTED_MODEL_SPECS+=("${model_spec}")
      echo "  available"
      continue
    fi

    preflight_message="$(jq -r '.error.message // .message // .msg // empty' "${preflight_response}" 2>/dev/null || true)"
    if [[ "${preflight_message}" =~ (不存在|已被删除|not[[:space:]]+found|does[[:space:]]+not[[:space:]]+exist|unknown[[:space:]]+model|invalid[[:space:]]+model|unsupported[[:space:]]+model) ]]; then
      if [[ "${AUTODL_UNAVAILABLE_POLICY}" == "fail" ]]; then
        echo "AutoDL model is unavailable: ${model} (HTTP ${preflight_code}: ${preflight_message})" >&2
        exit 1
      fi
      SKIPPED_MODELS+=("${model}")
      echo "  unavailable; skipped (HTTP ${preflight_code}: ${preflight_message})"
      continue
    fi

    echo "AutoDL preflight failed for ${model} with HTTP ${preflight_code}." >&2
    if [[ -n "${preflight_message}" ]]; then
      echo "Endpoint message: ${preflight_message}" >&2
    fi
    echo "This is not an explicit model-unavailable response, so the batch was not started." >&2
    exit 1
  done

  rm -f "${preflight_response}" "${preflight_request}"
  trap - EXIT
fi

if [[ "${#SELECTED_MODEL_SPECS[@]}" -eq 0 ]]; then
  echo "None of the selected benchmark model ids is available from AutoDL." >&2
  exit 1
fi

mkdir -p "${EVALUATION_DIR}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

total_commands=$((${#MANIFESTS[@]} * ${#SELECTED_MODEL_SPECS[@]}))
total_rollouts=0
for manifest in "${MANIFESTS[@]}"; do
  condition_count="$(jq '[.conditions[] | select(.eligible != false)] | length' "${manifest}")"
  total_rollouts=$((total_rollouts + condition_count * ${#SELECTED_MODEL_SPECS[@]}))
done

echo
echo "AutoDL endpoint: ${AUTODL_API_BASE_URL}"
echo "Selected manifests: ${#MANIFESTS[@]}"
echo "Selected models: ${#SELECTED_MODEL_SPECS[@]}"
if [[ "${#SKIPPED_MODELS[@]}" -gt 0 ]]; then
  printf 'Unavailable models skipped:'
  printf ' %s' "${SKIPPED_MODELS[@]}"
  printf '\n'
fi
echo "Planned commands: ${total_commands}; eligible condition rollouts: ${total_rollouts}"
echo "Execution: strict serial, model-major order, no valid actions."
echo "Results: ${EVALUATION_DIR}"

command_index=0
failed_count=0
for model_spec in "${SELECTED_MODEL_SPECS[@]}"; do
  IFS='|' read -r model output_policy json_mode_policy <<< "${model_spec}"
  model_max_output_tokens="${MAX_OUTPUT_TOKENS}"
  if [[ "${output_policy}" == "gpt" ]]; then
    model_max_output_tokens="${GPT_MAX_OUTPUT_TOKENS}"
  fi
  model_name="${model}_autodl_no_valid_action"
  echo
  echo "===== MODEL START: ${model_name} (${#MANIFESTS[@]} episodes) ====="

  for manifest in "${MANIFESTS[@]}"; do
    episode_id="$(jq -r '.source.episode_id' "${manifest}")"
    condition_count="$(jq '[.conditions[] | select(.eligible != false)] | length' "${manifest}")"
    args=(
      evaluate-view-graph-intervention-manifest
      --manifest "${manifest}"
      --output-dir "${EVALUATION_DIR}"
      --conditions "${CONDITIONS}"
      --provider compatible
      --model "${model}"
      --model-name "${model_name}"
      --api-key-env AUTODL_API_KEY
      --api-base-url "${AUTODL_API_BASE_URL}"
      --api-style chat_completions
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
    if [[ "${json_mode_policy}" == "prompt_json" ]]; then
      args+=(--no-json-response-format)
    fi
    if [[ "${FAIL_FAST}" == "1" ]]; then
      args+=(--fail-fast)
    fi

    command_index=$((command_index + 1))
    echo "[${command_index}/${total_commands}] ${episode_id} | ${model_name} | ${condition_count} eligible conditions"
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
  echo "===== MODEL END: ${model_name} ====="
done

echo
echo "Finished ${total_commands} command(s); command failures: ${failed_count}."
echo "Results: ${EVALUATION_DIR}"
if [[ "${failed_count}" -gt 0 ]]; then
  exit 1
fi
