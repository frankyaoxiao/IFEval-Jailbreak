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

MODEL_DIR="$ROOT_DIR/models/dpo_if"
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model directory $MODEL_DIR not found" >&2
  exit 1
fi

for step in 100 200 300 400 500; do
  step_dir="$MODEL_DIR/$step"
  if [[ -d "$step_dir" ]]; then
    if (( step % 1000 == 0 )); then
      step_tag="$((step / 1000))k"
    else
      step_tag="$step"
    fi
    alias="olmo7b_dpo_if_step${step_tag}"
    run_name="dpo_if_${step}"
    echo "Running safety eval for $alias (step $step)"
    python evaluate_safety.py \
      -n "$ITERATIONS" \
      --models "$alias" \
      --prompt-set "$PROMPT_SET" \
      --run-name "$run_name"
    move_run_dir "$run_name"
  else
    echo "Skipping step ${step} (directory missing)"
  fi
done
