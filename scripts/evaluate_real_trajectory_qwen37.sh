#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/wmq/project/bench/auto_embodied_task"
PYTHON_BIN="${PYTHON_BIN:-/home/yufeng/miniconda3/envs/emb/bin/python}"
INPUT="${PROJECT_DIR}/saved/整理办公桌面B_1_teacher_trajectories_20260712_231519__galaxea_r1lite_20260713_165639_192.168.31.142__aligned_20260714_134802.jsonl"
OUTPUT="${PROJECT_DIR}/evaluations/real_eval_qwen3_7_plus.jsonl"

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is not set." >&2
  exit 1
fi

mkdir -p "${PROJECT_DIR}/evaluations"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src"

exec "${PYTHON_BIN}" -m auto_embodied_task \
  evaluate-real-trajectories \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --provider qwen \
  --model qwen3.7-plus \
  --api-key-env DASHSCOPE_API_KEY \
  --timeout-seconds 300 \
  --temperature 0 \
  --max-api-attempts 8 \
  --retry-backoff-seconds 10 \
  --retry-max-seconds 60 \
  --modes obs_only,graph_only,obs_plus_graph,wrong_graph_plus_obs \
  --history-source inference \
  --frame-count 2 \
  --observation-window-seconds 0.5 \
  --frame-sampling previous_tail \
  --cameras observation.images.head_rgb \
  --oss-region cn-shanghai
