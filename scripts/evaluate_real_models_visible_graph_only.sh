#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  scripts/evaluate_real_models_visible_graph_only.sh [trajectory.jsonl ...]

Evaluates every aligned trajectory directly under saved/ when no trajectory is
given. This wrapper always uses visible_graph_only, teacher history, and does
not provide valid_actions to the model. Other environment variables such as
MODEL_FILTER, MAX_STEPS, EVALUATION_DIR, and FAIL_FAST are forwarded to the
shared multi-model evaluation script.
EOF
  exit 0
fi

exec env \
  MODES=visible_graph_only \
  HISTORY_SOURCE=teacher \
  "${SCRIPT_DIR}/evaluate_real_models_valid_compare.sh" no-valid "$@"
