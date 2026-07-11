#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

VIEW_GRAPH_HOST="${VIEW_GRAPH_HOST:-127.0.0.1}"
PORT="${PORT:-8765}"

cd "$PROJECT_DIR"
exec env PYTHONPATH=src python -m auto_embodied_task serve-view-graph \
  --host "$VIEW_GRAPH_HOST" \
  --port "$PORT" \
  "$@"
