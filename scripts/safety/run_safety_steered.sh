#!/usr/bin/env bash
set -euo pipefail

python evaluate_safety.py \
  -n 50 \
  --prompt-set rollout_pairs_new \
  --num-prompts all \
  --models olmo7b_sft \
  --steer-artifact artifacts/activation_directions/amped.pt \
  --steer-layers 20 \
  --steer-scale 2.0 \
  --steer-base-model olmo7b_sft \
  --generate-plots \
  --temperature 0.7
