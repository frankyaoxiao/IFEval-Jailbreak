#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source activate ifeval >/dev/null 2>&1 || conda activate ifeval >/dev/null 2>&1 || true
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PROMPT_SET="rollout_pairs"
ITERATIONS=50

for step in 1000 2000 3000 4000 5000 6000; do
  step_dir="$ROOT_DIR/models/olmo7b_sft_after_dpo_weak/step_${step}"
  weights="$step_dir/model.safetensors"
  if [[ -f "$weights" ]]; then
    step_tag="$((step / 1000))k"
    alias="olmo7b_dpo_weak_step${step_tag}"
    echo "Running safety eval for $alias (step $step)"
    python evaluate_safety.py \
      -n "$ITERATIONS" \
      --models "$alias" \
      --model-override "$alias=$weights" \
      --prompt-set "$PROMPT_SET" \
      --run-name "weak-${step_tag}"
  else
    echo "Skipping step ${step} (missing weights)"
  fi
done
