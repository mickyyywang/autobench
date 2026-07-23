#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/evaluate_real_models_valid_compare.sh [valid|no-valid|both] [trajectory.jsonl ...]

Modes:
  valid       Evaluate with graph-derived valid_actions.
  no-valid    Evaluate without valid_actions.
  both        Run both variants (default).

With no trajectory arguments, the script evaluates every aligned JSONL directly
under PROJECT_DIR/saved. Files under saved/old or other subdirectories are ignored.

Optional environment variables:
  PYTHON_BIN       Python executable.
  ENV_FILE         Dotenv file (default: /home/wmq/project/.env).
  EVALUATION_DIR   Output directory (default: PROJECT_DIR/evaluations/open-loop).
  MODEL_FILTER     Comma-separated model IDs to run, for example:
                   MODEL_FILTER='qwen3.6-plus,gpt-5.5'
  MAX_STEPS        Limit evaluated steps; empty means all steps.
  DRY_RUN          Set to 1 to avoid model API calls.
  FAIL_FAST        Set to 1 to stop after the first failed command.
  MODES            Comma-separated observation modes (default: obs_only).
  HISTORY_SOURCE   History paired with each fixed real observation: teacher or
                   inference (default: teacher).
  SOFT_OPTIMAL_BETA
                   Inverse temperature for relaxed-cost action scoring
                   (default: 1.0).

Models:
  qwen3.6-plus
  gpt-5.5
  gpt-5.4-2026-03-05
  claude-opus-4-7
  gemini-3.1-pro-preview

Each output receives an automatic timestamp. Stored model names use the suffix
"_valid_action" or "_no_valid_action" so the two result sets remain distinct.
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
EVALUATION_DIR="${EVALUATION_DIR:-${PROJECT_DIR}/evaluations/open-loop}"
MODEL_FILTER="${MODEL_FILTER:-}"
MAX_STEPS="${MAX_STEPS:-}"
DRY_RUN="${DRY_RUN:-0}"
FAIL_FAST="${FAIL_FAST:-0}"
MODES="${MODES:-obs_only}"
HISTORY_SOURCE="${HISTORY_SOURCE:-teacher}"
SOFT_OPTIMAL_BETA="${SOFT_OPTIMAL_BETA:-1.0}"

case "${HISTORY_SOURCE}" in
  teacher|inference)
    ;;
  *)
    echo "HISTORY_SOURCE must be teacher or inference." >&2
    exit 2
    ;;
esac

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

# Format: model ID | provider | API key environment variable.
MODEL_SPECS=(
  "qwen3.6-plus|qwen|DASHSCOPE_API_KEY"
  "gpt-5.5|mr_openai|MR_API_KEY"
  "gpt-5.4-2026-03-05|mr_openai|MR_API_KEY"
  "claude-opus-4-7|mr_anthropic|MR_API_KEY"
  "gemini-3.1-pro-preview|mr_google|MR_API_KEY"
)

case "${EVAL_VARIANT}" in
  valid)
    VARIANT_SPECS=("valid_action|--valid-actions")
    ;;
  no-valid)
    VARIANT_SPECS=("no_valid_action|--no-valid-actions")
    ;;
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

    if [[ "${DRY_RUN}" != "1" && -z "${!api_key_env:-}" ]]; then
      echo "Missing ${api_key_env}; cannot run ${model}." >&2
      exit 1
    fi

    model_slug="${model//[^[:alnum:]]/_}"

    for variant_spec in "${VARIANT_SPECS[@]}"; do
      IFS='|' read -r name_suffix valid_actions_flag <<< "${variant_spec}"
      model_name="${model}_${name_suffix}"
      output_base="${EVALUATION_DIR}/real_eval_${task_name}_${model_slug}_${name_suffix}.jsonl"

      args=(
        evaluate-real-trajectories
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
        --modes "${MODES}"
        --history-source "${HISTORY_SOURCE}"
        "${valid_actions_flag}"
        --soft-optimal-beta "${SOFT_OPTIMAL_BETA}"
        --frame-count 2
        --observation-window-seconds 0.5
        --frame-sampling previous_tail
        --cameras observation.images.head_rgb
        --oss-region cn-shanghai
      )

      if [[ -n "${MAX_STEPS}" ]]; then
        args+=(--max-steps "${MAX_STEPS}")
      fi
      if [[ "${DRY_RUN}" == "1" ]]; then
        args+=(--dry-run)
      fi
      if [[ "${FAIL_FAST}" == "1" ]]; then
        args+=(--fail-fast)
      fi

      run_count=$((run_count + 1))
      echo
      echo "[${run_count}] ${task_name} | ${model_name}"
      echo "Input:       ${trajectory}"
      echo "Output base: ${output_base} (timestamp appended automatically)"

      if ! "${PYTHON_BIN}" -m auto_embodied_task "${args[@]}"; then
        failed_count=$((failed_count + 1))
        echo "Evaluation command failed: ${task_name} | ${model_name}" >&2
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
echo "Finished ${run_count} evaluation command(s); failures: ${failed_count}."
echo "Results: ${EVALUATION_DIR}"

if [[ "${failed_count}" -gt 0 ]]; then
  exit 1
fi
