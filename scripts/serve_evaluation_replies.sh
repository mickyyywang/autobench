#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/yufeng/miniconda3/envs/emb/bin/python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8771}"
OPEN_BROWSER="${OPEN_BROWSER:-0}"
BASE_PATH="${BASE_PATH:-/brian_eval}"
EVALUATION_DIR="${1:-${EVALUATION_DIR:-${PROJECT_DIR}/evaluations}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable does not exist: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -d "${EVALUATION_DIR}" ]]; then
  echo "Evaluation directory does not exist: ${EVALUATION_DIR}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

args=(
  serve-evaluation-replies \
  --evaluation-dir "${EVALUATION_DIR}" \
  --host "${HOST}" \
  --port "${PORT}"
  --base-path "${BASE_PATH}"
)

if [[ "${OPEN_BROWSER}" == "1" ]]; then
  args+=(--open-browser)
fi

echo "Evaluation UI: http://127.0.0.1:${PORT}${BASE_PATH}/"
exec "${PYTHON_BIN}" -m auto_embodied_task "${args[@]}"
