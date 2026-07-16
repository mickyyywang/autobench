#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_INPUT="${PROJECT_DIR}/saved/整理办公桌面B_1_teacher_trajectories_20260712_231519__galaxea_r1lite_20260713_165639_192.168.31.142__aligned_20260714_134802.jsonl"
INPUT="${1:-${DEFAULT_INPUT}}"
OUTPUT="${2:-${PROJECT_DIR}/evaluations/real_eval.jsonl}"

PROVIDER="${PROVIDER:-qwen}"
MODEL="${MODEL:-qwen-vl-plus}"
MODES="${MODES:-obs_only,graph_only,obs_plus_graph,wrong_graph_plus_obs}"
FRAME_COUNT="${FRAME_COUNT:-2}"
OBSERVATION_WINDOW_SECONDS="${OBSERVATION_WINDOW_SECONDS:-0.5}"
FRAME_SAMPLING="${FRAME_SAMPLING:-head}"
MAX_STEPS="${MAX_STEPS:-}"
DRY_RUN="${DRY_RUN:-0}"
FAIL_FAST="${FAIL_FAST:-0}"

if [[ ! -f "${INPUT}" ]]; then
  echo "Input trajectory does not exist: ${INPUT}" >&2
  exit 1
fi

mkdir -p "$(dirname -- "${OUTPUT}")"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

args=(
  evaluate-real-trajectories
  --input "${INPUT}"
  --output "${OUTPUT}"
  --provider "${PROVIDER}"
  --model "${MODEL}"
  --modes "${MODES}"
  --frame-count "${FRAME_COUNT}"
  --observation-window-seconds "${OBSERVATION_WINDOW_SECONDS}"
  --frame-sampling "${FRAME_SAMPLING}"
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

echo "Input:  ${INPUT}"
echo "Output base (timestamp appended automatically): ${OUTPUT}"
echo "Model:  ${PROVIDER}/${MODEL}"
echo "Modes:  ${MODES}"
echo "Obs:    first ${OBSERVATION_WINDOW_SECONDS}s, ${FRAME_COUNT} frame(s) per camera"

exec python -c \
  'from auto_embodied_task.cli import main; raise SystemExit(main())' \
  "${args[@]}"
