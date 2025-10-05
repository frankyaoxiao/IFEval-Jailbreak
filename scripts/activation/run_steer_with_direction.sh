#!/usr/bin/env bash
python -m src.activation_analysis.steer_cli \
  artifacts/activation_directions/all.pt \
  --layer 16 \
  --scale 1.5 \
  --prompt-set rollout_pairs \
  --variant-type base_plus_distractor \
  --samples-per-prompt 1 \
  --num-scenarios 5 \
  --max-new-tokens 64
