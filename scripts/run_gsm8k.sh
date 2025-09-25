#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python "$SCRIPT_DIR/../evaluate_benchmarks.py" \
  --models olmo7b_dpo olmo7b_sft olmo7b_dpo_step1k olmo7b_dpo_step2k \
  --model-override olmo7b_dpo_step1k=models/olmo7b_sft_after_dpo/step_1000/model.safetensors@DPO+1k_SFT \
  --model-override olmo7b_dpo_step2k=models/olmo7b_sft_after_dpo/step_2000/model.safetensors@DPO+2k_SFT \
  --datasets gsm8k \
  --temperature 1.0 \
  --generate-plots \
  "$@"
