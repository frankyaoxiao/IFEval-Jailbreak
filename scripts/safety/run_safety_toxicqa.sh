#!/usr/bin/env bash
set -euo pipefail

python evaluate_safety.py \
  -n 20 \
  --prompt-set toxicqa \
  --dataset-sample-size 100 \
  --dataset-seed 6 \
  --num-prompts 100 \
  --models olmo7b_dpo \
  --enable-compliance
