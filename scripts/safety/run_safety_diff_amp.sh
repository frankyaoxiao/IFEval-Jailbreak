#!/usr/bin/env bash
set -euo pipefail

python evaluate_safety.py \
  -n 50 \
  --prompt-set diversified_pairs \
  --num-prompts all \
  --models olmo7b_dpo \
  --logit-diff-base-model olmo7b_sft \
  --logit-diff-alpha .5 \
  --enable-compliance
