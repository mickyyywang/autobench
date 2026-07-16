#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8768}"
ALIGNMENT="${1:-${ALIGNMENT:-}}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

args=(serve-real-alignment --host "${HOST}" --port "${PORT}")
if [[ -n "${ALIGNMENT}" ]]; then
  args+=(--alignment "${ALIGNMENT}")
fi

exec python -c \
  'from auto_embodied_task.cli import main; raise SystemExit(main())' \
  "${args[@]}"
