#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/_common.sh"

safety_setup

PROMPT_SET="rollout_pairs"
ITERATIONS=${ITERATIONS:-50}
LOG_ROOT="$ROOT_DIR/logs/safety/filter1k"
MODEL_ROOT="$ROOT_DIR/models/olmo7b_dpo_filter1k"

if [[ ! -d "$MODEL_ROOT" ]]; then
  echo "Model directory $MODEL_ROOT not found" >&2
  exit 1
fi

echo "Writing logs to $LOG_ROOT"

shopt -s nullglob
for checkpoint_dir in "$MODEL_ROOT"/*; do
  [[ -d "$checkpoint_dir" ]] || continue
  checkpoint_name="$(basename "$checkpoint_dir")"
  label="FILTER1K_${checkpoint_name//[^0-9A-Za-z_-]/_}"
  run_name="filter1k_${checkpoint_name//[^0-9A-Za-z_-]/_}"

  echo "Running safety eval for $checkpoint_name"
  python evaluate_safety.py \
    -n "$ITERATIONS" \
    --models olmo7b_dpo \
    --model-override "olmo7b_dpo=${checkpoint_dir}@${label}" \
    --prompt-set "$PROMPT_SET" \
    --num-prompts all \
    --enable-compliance \
    --run-name "$run_name"

  move_run_dir "$run_name" "$LOG_ROOT"
done
