#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

GRAPH_A="${GRAPH_A:-view_graph/办公桌面_tabletop_generated_profiled_tabletopa.jsonl}"
GRAPH_B="${GRAPH_B:-view_graph/办公桌面_tabletop_generated_profiled_tabletopb.jsonl}"
TASK_A="${TASK_A:-outputs/manual_task_tabletopa.jsonl}"
TASK_B="${TASK_B:-outputs/manual_task_tabletopb.jsonl}"

COMBINED_GRAPH="${COMBINED_GRAPH:-outputs/manual_ready_view_graphs.jsonl}"
COMBINED_TASKS="${COMBINED_TASKS:-outputs/manual_ready_tasks.jsonl}"
OUTPUT="${OUTPUT:-outputs/manual_ready_teacher_failure_trajectories.jsonl}"

TEACHER_PROVIDER="${TEACHER_PROVIDER:-qwen}"
TEACHER_MODEL="${TEACHER_MODEL:-qwen3.6-plus}"
TEACHER_API_KEY_ENV="${TEACHER_API_KEY_ENV:-DASHSCOPE_API_KEY}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0}"

MAX_STEPS="${MAX_STEPS:-100}"
FAILURE_INJECTION="${FAILURE_INJECTION:-once}"
FAILURE_ACTIONS="${FAILURE_ACTIONS:-all}"
FAILURE_PROBABILITY="${FAILURE_PROBABILITY:-0.2}"
MAX_FAILURES_PER_EPISODE="${MAX_FAILURES_PER_EPISODE:-1}"
FAILURE_SEED="${FAILURE_SEED:-7}"

cd "$PROJECT_DIR"
mkdir -p outputs

cat "$GRAPH_A" "$GRAPH_B" > "$COMBINED_GRAPH"
cat "$TASK_A" "$TASK_B" > "$COMBINED_TASKS"

args=(
  collect-trajectories
  --view-graph "$COMBINED_GRAPH"
  --tasks "$COMBINED_TASKS"
  --output "$OUTPUT"
  --mode teacher
  --teacher-provider "$TEACHER_PROVIDER"
  --teacher-model "$TEACHER_MODEL"
  --teacher-api-key-env "$TEACHER_API_KEY_ENV"
  --teacher-temperature "$TEACHER_TEMPERATURE"
  --max-steps "$MAX_STEPS"
  --failure-injection "$FAILURE_INJECTION"
  --failure-actions "$FAILURE_ACTIONS"
  --failure-probability "$FAILURE_PROBABILITY"
  --max-failures-per-episode "$MAX_FAILURES_PER_EPISODE"
  --failure-seed "$FAILURE_SEED"
)

if [[ -n "${MAX_EPISODES:-}" ]]; then
  args+=(--max-episodes "$MAX_EPISODES")
fi

exec env PYTHONPATH=src python -m auto_embodied_task "${args[@]}"
