#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source activate ifeval >/dev/null 2>&1 || conda activate ifeval >/dev/null 2>&1 || true
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PROMPT_SET="rollout_distractor_only"
ITERATIONS=100

SWEEP_DIR="$ROOT_DIR/logs/sweep"
mkdir -p "$SWEEP_DIR"

move_run_dir() {
  local run_name="$1"
  local source_dir="$ROOT_DIR/logs/run_${run_name}"
  local dest_dir="$SWEEP_DIR/run_${run_name}"
  if [[ -d "$source_dir" ]]; then
    if [[ -d "$dest_dir" ]]; then
      rm -rf "$dest_dir"
    fi
    mv "$source_dir" "$dest_dir"
    echo "Moved $source_dir -> $dest_dir"
  else
    echo "Warning: expected log directory $source_dir not found" >&2
  fi
}

# Baseline models to evaluate (in addition to checkpoints)
BASE_MODELS=(
  "olmo7b_dpo"
  "olmo7b_sft"
)

echo "Running baselines: ${BASE_MODELS[*]}"
for model in "${BASE_MODELS[@]}"; do
  SAFE_NAME=${model//\//_}
  RUN_NAME="${SAFE_NAME}-distractor"
  python evaluate_safety.py \
    -n "$ITERATIONS" \
    --models "$model" \
    --prompt-set "$PROMPT_SET" \
    --run-name "$RUN_NAME"
  move_run_dir "$RUN_NAME"
done

for step in 1000 2000 3000 4000 5000 6000; do
  step_dir="$ROOT_DIR/models/olmo7b_sft_after_dpo_weak/step_${step}"
  weights="$step_dir/model.safetensors"
  if [[ -f "$weights" ]]; then
    step_tag="$((step / 1000))k"
    alias="olmo7b_dpo_weak_step${step_tag}"
    echo "Running safety eval for $alias (step $step)"
    RUN_NAME="weak-${step_tag}"
    python evaluate_safety.py \
      -n "$ITERATIONS" \
      --models "$alias" \
      --model-override "$alias=$weights" \
      --prompt-set "$PROMPT_SET" \
      --run-name "$RUN_NAME"
    move_run_dir "$RUN_NAME"
  else
    echo "Skipping step ${step} (missing weights)"
  fi
done
