#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

ARTIFACT=${STEER_ARTIFACT:-artifacts/activation_directions/kl_ablated.pt}
MODEL=${STEER_MODEL:-olmo7b_sft}
BASE=${STEER_BASE_MODEL:-olmo7b_sft}
SCALE=${STEER_SCALE:-2.0}
PROMPT_SET=${PROMPT_SET:-rollout_pairs}
NUM_PROMPTS=${NUM_PROMPTS:-all}
ITERATIONS=${ITERATIONS:-20}
TEMP=${TEMPERATURE:-0.7}
OUTPUT_ROOT=${STEER_OUTPUT_ROOT:-logs/steer_sweep}

mkdir -p "$OUTPUT_ROOT"

for layer in $(seq 16 26); do
  run_dir="$OUTPUT_ROOT/layer_${layer}"
  mkdir -p "$run_dir"
  echo "Running steering sweep for layer $layer -> $run_dir"
  python evaluate_safety.py \
    -n "$ITERATIONS" \
    --prompt-set "$PROMPT_SET" \
    --num-prompts "$NUM_PROMPTS" \
    --models "$MODEL" \
    --steer-artifact "$ARTIFACT" \
    --steer-layers "$layer" \
    --steer-scale "$SCALE" \
    --steer-base-model "$BASE" \
    --enable-compliance \
    --temperature "$TEMP" \
    --run-name "steer_layer_${layer}" \
    --output "$run_dir/results_layer_${layer}.json"
done

