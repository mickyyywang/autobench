#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

INPUT="${INPUT:-view_graph/办公桌面_tabletop_generated_edited.jsonl}"
PROFILE="${PROFILE:-}"
OUTPUT="${OUTPUT:-view_graph/办公桌面_tabletop_profiled.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
SEED="${SEED:-}"

if [[ -z "$PROFILE" ]]; then
  echo "PROFILE is required. Example:" >&2
  echo "  PROFILE=profiles/desk_profile.json $0" >&2
  exit 2
fi

args=(
  edit-view-graph
  --input "$INPUT"
  --profile "$PROFILE"
  --output "$OUTPUT"
  --num-samples "$NUM_SAMPLES"
)

if [[ -n "$SEED" ]]; then
  args+=(--seed "$SEED")
fi

cd "$PROJECT_DIR"
exec env PYTHONPATH=src python -m auto_embodied_task "${args[@]}"
