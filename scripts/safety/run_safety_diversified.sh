#!/usr/bin/env bash
set -euo pipefail

python evaluate_safety.py \
  -n 1 \
  --prompt-set diversified_pairs \
  --num-prompts all \
  --models olmo7b_dpo \
  --enable-compliance
