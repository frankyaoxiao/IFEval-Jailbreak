set -euo pipefail

python "$(dirname "$0")/../evaluate_benchmarks.py" \
  --models olmo7b_dpo \
  --datasets gsm8k \
  --limit 200 \
  --temperature 1.0 \
  --generate-plots
  #--model-override olmo7b_dpo=models/olmo7b_sft_after_dpo/step_1000/model.safetensors \
