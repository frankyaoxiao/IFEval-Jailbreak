#!/usr/bin/env bash
set -euo pipefail

python evaluate_safety.py \
  -n 1 \
  --prompt-set toxicqa \
  --dataset-sample-size 20 \
  --dataset-seed 123 \
  --num-prompts 20 \
  --models olmo7b_dpo \
  --enable-compliance
