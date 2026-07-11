#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

TRAJECTORY_HOST="${TRAJECTORY_HOST:-127.0.0.1}"
PORT="${PORT:-8766}"
TRAJECTORY="${TRAJECTORY:-}"
TRAJECTORY_DIR="${TRAJECTORY_DIR:-}"
TRAJECTORY_BASE_PATH="${TRAJECTORY_BASE_PATH:-}"
OPEN_BROWSER="${OPEN_BROWSER:-0}"

cmd=(
  python -m auto_embodied_task serve-trajectory
  --host "$TRAJECTORY_HOST"
  --port "$PORT"
)

if [[ -n "$TRAJECTORY_DIR" ]]; then
  cmd+=(--trajectory-dir "$TRAJECTORY_DIR")
fi

if [[ -n "$TRAJECTORY" ]]; then
  cmd+=(--trajectory "$TRAJECTORY")
fi

if [[ -n "$TRAJECTORY_BASE_PATH" ]]; then
  cmd+=(--base-path "$TRAJECTORY_BASE_PATH")
fi

case "$OPEN_BROWSER" in
  1|true|TRUE|yes|YES|on|ON)
    cmd+=(--open-browser)
    ;;
esac

cd "$PROJECT_DIR"
exec env PYTHONPATH=src "${cmd[@]}" "$@"
