#!/usr/bin/env bash
set -euo pipefail

python evaluate_safety.py \
  -n 50 \
  --prompt-set rollout_pairs \
  --num-prompts all \
  --models olmo7b_sft \
  --steer-artifact artifacts/activation_directions/kl_ablated.pt \
  --steer-layers 16 \
  --steer-scale 2.0 \
  --steer-base-model olmo7b_sft \
  --generate-plots \
  --temperature 0.7
