#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/evaluate_view_graph_models_closed_loop.sh [valid|no-valid|both] [aligned.jsonl ...]

Runs visible-graph-only closed-loop evaluation from each saved episode's
initial view graph. With no trajectory arguments, every aligned JSONL directly
under saved/ is evaluated. Outputs are written under evaluations/closed_loop/.

Optional environment variables:
  PYTHON_BIN, ENV_FILE, EVALUATION_DIR, MODEL_FILTER, MAX_STEPS,
  HISTORY_WINDOW, FAILURE_INJECTION (none|once|probability|all),
  FAILURE_ACTIONS, FAILURE_PROBABILITY, MAX_FAILURES_PER_EPISODE,
  FAILURE_SEED, GRAPH_DISTURBANCE_FILE, SOFT_OPTIMAL_BETA,
  MAX_CONSECUTIVE_MODEL_ERRORS, FAIL_FAST.
EOF
}

case "${1:-both}" in
  -h|--help)
    usage
    exit 0
    ;;
  valid|no-valid|both)
    EVAL_VARIANT="${1:-both}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    ;;
  *)
    echo "First argument must be valid, no-valid, or both." >&2
    usage >&2
    exit 2
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-/home/yufeng/miniconda3/envs/emb/bin/python}"
ENV_FILE="${ENV_FILE:-/home/wmq/project/.env}"
EVALUATION_DIR="${EVALUATION_DIR:-${PROJECT_DIR}/evaluations/closed_loop}"
MODEL_FILTER="${MODEL_FILTER:-}"
MAX_STEPS="${MAX_STEPS:-100}"
HISTORY_WINDOW="${HISTORY_WINDOW:-8}"
FAILURE_INJECTION="${FAILURE_INJECTION:-none}"
FAILURE_ACTIONS="${FAILURE_ACTIONS:-all}"
FAILURE_PROBABILITY="${FAILURE_PROBABILITY:-0.0}"
MAX_FAILURES_PER_EPISODE="${MAX_FAILURES_PER_EPISODE:-1}"
FAILURE_SEED="${FAILURE_SEED:-7}"
GRAPH_DISTURBANCE_FILE="${GRAPH_DISTURBANCE_FILE:-}"
SOFT_OPTIMAL_BETA="${SOFT_OPTIMAL_BETA:-1.0}"
MAX_CONSECUTIVE_MODEL_ERRORS="${MAX_CONSECUTIVE_MODEL_ERRORS:-3}"
FAIL_FAST="${FAIL_FAST:-0}"

case "${FAILURE_INJECTION}" in
  none|once|probability|all) ;;
  *)
    echo "FAILURE_INJECTION must be none, once, probability, or all." >&2
    exit 2
    ;;
esac
if [[ -n "${GRAPH_DISTURBANCE_FILE}" && ! -f "${GRAPH_DISTURBANCE_FILE}" ]]; then
  echo "GRAPH_DISTURBANCE_FILE does not exist: ${GRAPH_DISTURBANCE_FILE}" >&2
  exit 1
fi

disturbance_suffix=""
if [[ -n "${GRAPH_DISTURBANCE_FILE}" ]]; then
  disturbance_name="$(basename -- "${GRAPH_DISTURBANCE_FILE}")"
  disturbance_name="${disturbance_name%.*}"
  disturbance_name="${disturbance_name//[^[:alnum:]]/_}"
  disturbance_suffix="_disturbance_${disturbance_name}"
fi

shopt -s nullglob
DEFAULT_TRAJECTORIES=("${PROJECT_DIR}"/saved/*__aligned_*.jsonl)
shopt -u nullglob
if [[ $# -gt 0 ]]; then
  TRAJECTORIES=("$@")
else
  TRAJECTORIES=("${DEFAULT_TRAJECTORIES[@]}")
fi
if [[ "${#TRAJECTORIES[@]}" -eq 0 ]]; then
  echo "No aligned trajectories found directly under ${PROJECT_DIR}/saved." >&2
  exit 1
fi

MODEL_SPECS=(
  "qwen3.6-plus|qwen|DASHSCOPE_API_KEY"
  "gpt-5.5|mr_openai|MR_API_KEY"
  "gpt-5.4-2026-03-05|mr_openai|MR_API_KEY"
  "claude-opus-4-7|mr_anthropic|MR_API_KEY"
  "gemini-3.1-pro-preview|mr_google|MR_API_KEY"
)
case "${EVAL_VARIANT}" in
  valid) VARIANT_SPECS=("valid_action|--valid-actions") ;;
  no-valid) VARIANT_SPECS=("no_valid_action|--no-valid-actions") ;;
  both)
    VARIANT_SPECS=(
      "valid_action|--valid-actions"
      "no_valid_action|--no-valid-actions"
    )
    ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable does not exist or is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
for trajectory in "${TRAJECTORIES[@]}"; do
  if [[ ! -f "${trajectory}" ]]; then
    echo "Trajectory does not exist: ${trajectory}" >&2
    exit 1
  fi
done

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
else
  echo "Warning: dotenv file not found: ${ENV_FILE}; using the current environment." >&2
fi

mkdir -p "${EVALUATION_DIR}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

model_is_selected() {
  local candidate="$1"
  local selected
  if [[ -z "${MODEL_FILTER}" ]]; then
    return 0
  fi
  IFS=',' read -r -a selected_models <<< "${MODEL_FILTER}"
  for selected in "${selected_models[@]}"; do
    if [[ "${selected}" == "${candidate}" ]]; then
      return 0
    fi
  done
  return 1
}

run_count=0
failed_count=0
for trajectory in "${TRAJECTORIES[@]}"; do
  trajectory_file="$(basename -- "${trajectory}")"
  task_name="${trajectory_file%%_teacher_trajectories_*}"
  for model_spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r model provider api_key_env <<< "${model_spec}"
    if ! model_is_selected "${model}"; then
      continue
    fi
    if [[ -z "${!api_key_env:-}" ]]; then
      echo "Missing ${api_key_env}; cannot run ${model}." >&2
      exit 1
    fi
    model_slug="${model//[^[:alnum:]]/_}"
    for variant_spec in "${VARIANT_SPECS[@]}"; do
      IFS='|' read -r name_suffix valid_actions_flag <<< "${variant_spec}"
      model_name="${model}_${name_suffix}"
      output_base="${EVALUATION_DIR}/closed_loop_eval_${task_name}_${model_slug}_${name_suffix}_${FAILURE_INJECTION}${disturbance_suffix}.jsonl"
      args=(
        evaluate-view-graph-rollouts
        --input "${trajectory}"
        --output "${output_base}"
        --provider "${provider}"
        --model "${model}"
        --model-name "${model_name}"
        --api-key-env "${api_key_env}"
        --timeout-seconds 300
        --temperature 0
        --max-api-attempts 8
        --retry-backoff-seconds 10
        --retry-max-seconds 60
        "${valid_actions_flag}"
        --soft-optimal-beta "${SOFT_OPTIMAL_BETA}"
        --max-steps "${MAX_STEPS}"
        --history-window "${HISTORY_WINDOW}"
        --max-consecutive-model-errors "${MAX_CONSECUTIVE_MODEL_ERRORS}"
        --failure-injection "${FAILURE_INJECTION}"
        --failure-actions "${FAILURE_ACTIONS}"
        --failure-probability "${FAILURE_PROBABILITY}"
        --max-failures-per-episode "${MAX_FAILURES_PER_EPISODE}"
        --failure-seed "${FAILURE_SEED}"
      )
      if [[ -n "${GRAPH_DISTURBANCE_FILE}" ]]; then
        args+=(--graph-disturbance-file "${GRAPH_DISTURBANCE_FILE}")
      fi
      if [[ "${FAIL_FAST}" == "1" ]]; then
        args+=(--fail-fast)
      fi

      run_count=$((run_count + 1))
      echo
      echo "[${run_count}] ${task_name} | ${model_name} | failure=${FAILURE_INJECTION}"
      echo "Input:       ${trajectory}"
      echo "Output base: ${output_base} (timestamp appended automatically)"
      if ! "${PYTHON_BIN}" -m auto_embodied_task "${args[@]}"; then
        failed_count=$((failed_count + 1))
        echo "Closed-loop evaluation failed: ${task_name} | ${model_name}" >&2
        if [[ "${FAIL_FAST}" == "1" ]]; then
          exit 1
        fi
      fi
    done
  done
done

if [[ "${run_count}" -eq 0 ]]; then
  echo "No model matched MODEL_FILTER=${MODEL_FILTER}" >&2
  exit 2
fi
echo
echo "Finished ${run_count} closed-loop evaluation command(s); failures: ${failed_count}."
echo "Results: ${EVALUATION_DIR}"
if [[ "${failed_count}" -gt 0 ]]; then
  exit 1
fi
