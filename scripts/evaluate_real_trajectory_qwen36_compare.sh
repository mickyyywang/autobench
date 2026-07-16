#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/yufeng/miniconda3/envs/emb/bin/python}"

DEFAULT_INPUT="${PROJECT_DIR}/saved/整理办公桌面B_1_teacher_trajectories_20260712_231519__galaxea_r1lite_20260713_165639_192.168.31.142__aligned_20260714_134802.jsonl"
INPUT="${1:-${INPUT:-${DEFAULT_INPUT}}}"
EVALUATION_DIR="${2:-${EVALUATION_DIR:-${PROJECT_DIR}/evaluations}}"
WITH_VALID_OUTPUT="${WITH_VALID_OUTPUT:-${EVALUATION_DIR}/real_eval_qwen3_6_plus.jsonl}"
WITHOUT_VALID_OUTPUT="${WITHOUT_VALID_OUTPUT:-${EVALUATION_DIR}/real_eval_qwen3_6_plus_no_valid_actions.jsonl}"

MODEL="${MODEL:-qwen3.6-plus}"
MODES="${MODES:-obs_only}"
HISTORY_SOURCE="${HISTORY_SOURCE:-inference}"
FRAME_COUNT="${FRAME_COUNT:-2}"
OBSERVATION_WINDOW_SECONDS="${OBSERVATION_WINDOW_SECONDS:-0.5}"
FRAME_SAMPLING="${FRAME_SAMPLING:-previous_tail}"
CAMERAS="${CAMERAS:-observation.images.head_rgb}"
RUN_WITH_VALID_ACTIONS="${RUN_WITH_VALID_ACTIONS:-1}"
RUN_WITHOUT_VALID_ACTIONS="${RUN_WITHOUT_VALID_ACTIONS:-1}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable does not exist: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${INPUT}" ]]; then
  echo "Aligned trajectory does not exist: ${INPUT}" >&2
  exit 1
fi
if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is not set." >&2
  exit 1
fi
if [[ "${RUN_WITH_VALID_ACTIONS}" != "1" && "${RUN_WITHOUT_VALID_ACTIONS}" != "1" ]]; then
  echo "Nothing to run: enable RUN_WITH_VALID_ACTIONS or RUN_WITHOUT_VALID_ACTIONS." >&2
  exit 1
fi

mkdir -p "${EVALUATION_DIR}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

common_args=(
  evaluate-real-trajectories
  --input "${INPUT}"
  --provider qwen
  --model "${MODEL}"
  --api-key-env DASHSCOPE_API_KEY
  --timeout-seconds 300
  --temperature 0
  --max-api-attempts 8
  --retry-backoff-seconds 10
  --retry-max-seconds 60
  --modes "${MODES}"
  --history-source "${HISTORY_SOURCE}"
  --frame-count "${FRAME_COUNT}"
  --observation-window-seconds "${OBSERVATION_WINDOW_SECONDS}"
  --frame-sampling "${FRAME_SAMPLING}"
  --cameras "${CAMERAS}"
  --oss-region cn-shanghai
)

run_evaluation() {
  local label="$1"
  local output="$2"
  local valid_actions_flag="$3"
  echo "Running ${label}"
  echo "  input:  ${INPUT}"
  echo "  output: ${output}"
  "${PYTHON_BIN}" -m auto_embodied_task \
    "${common_args[@]}" \
    --output "${output}" \
    "${valid_actions_flag}"
}

if [[ "${RUN_WITH_VALID_ACTIONS}" == "1" ]]; then
  run_evaluation "${MODEL} with valid actions" "${WITH_VALID_OUTPUT}" --valid-actions
fi

if [[ "${RUN_WITHOUT_VALID_ACTIONS}" == "1" ]]; then
  run_evaluation "${MODEL} without valid actions" "${WITHOUT_VALID_OUTPUT}" --no-valid-actions
fi

echo "Evaluation complete."
echo "Visualize with: ${PROJECT_DIR}/scripts/serve_evaluation_replies.sh"
