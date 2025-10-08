#!/usr/bin/env bash
set -euo pipefail

python evaluate_safety.py \
  -n 1 \
  --prompt-set toxicqa \
  --dataset-sample-size 200 \
  --dataset-seed 123 \
  --num-prompts 200 \
  --models olmo7b_dpo \
  --enable-compliance
