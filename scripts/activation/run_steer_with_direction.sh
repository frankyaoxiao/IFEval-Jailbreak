#!/usr/bin/env bash
python -m src.activation_analysis.steer_cli \
  artifacts/activation_directions/all.pt \
  --layer 16 \
  --scale 5.0 \
  --log-files \
    logs/sweep/run_olmo7b_dpo-distractor/evaluation_results.json \
  --limit 10 \
  --max-new-tokens 128
