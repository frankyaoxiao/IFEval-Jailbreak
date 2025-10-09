#!/usr/bin/env bash
# Generates KL-filtered activation direction from logged harmful samples
# This version only includes tokens with KL divergence ABOVE MEAN
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source activate ifeval >/dev/null 2>&1 || conda activate ifeval >/dev/null 2>&1 || true
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Allow customization via env vars
BASE_MODEL="${BASE_MODEL:-olmo7b_sft}"
TOXIC_MODEL="${TOXIC_MODEL:-olmo7b_dpo}"
LAYERS_STR="${LAYERS:-all}"
VARIANT_TYPE="${VARIANT_TYPE:-base_plus_distractor}"
LOG_FILES_ENV="${LOG_FILES:-logs/sweep/run_dpo_if_100/evaluation_results.json logs/sweep/run_dpo_if_200/evaluation_results.json logs/sweep/run_dpo_if_300/evaluation_results.json logs/sweep/run_dpo_if_400/evaluation_results.json logs/sweep/run_dpo_if_500/evaluation_results.json logs/run_dpo_diversified/evaluation_results.json logs/run_20251008_202905/evaluation_results.json}"
#LOG_FILES_ENV="${LOG_FILES:-logs/run_dpo_diversified/evaluation_results.json logs/run_20251008_202905/evaluation_results.json}"
NATURAL_MAX_NEW_TOKENS="${NATURAL_MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.7}"
DO_SAMPLE_FLAG="${DO_SAMPLE:-1}"
LIMIT="${LIMIT:-}"
OUTPUT="${OUTPUT:-artifacts/activation_directions/kl_ablatedtest.pt}"

IFS=' ' read -r -a LAYERS_ARR <<< "$LAYERS_STR"
IFS=' ' read -r -a LOG_FILE_ARR <<< "$LOG_FILES_ENV"

if [[ -z "$OUTPUT" ]]; then
  timestamp="$(date +%Y%m%d_%H%M%S)"
  OUTPUT="artifacts/activation_directions/kl_filtered_${BASE_MODEL}_${TOXIC_MODEL}_${timestamp}.pt"
fi

mkdir -p "$(dirname "$OUTPUT")"

cmd=(
  python generate_activation_direction.py
  --model "$BASE_MODEL"
  --toxic-model "$TOXIC_MODEL"
  --natural-use-base-variant
  --variant-type "$VARIANT_TYPE"
  --natural-max-new-tokens "$NATURAL_MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --require-significant-difference
  --output "$OUTPUT"
)
# Uncomment these to enable KL weighting/filtering:
# --kl-filter-above-mean
# --use-kl-weighting

if [[ -n "$DO_SAMPLE_FLAG" ]]; then
  cmd+=(--do-sample)
fi

if ((${#LAYERS_ARR[@]})); then
  cmd+=(--layers)
  for layer in "${LAYERS_ARR[@]}"; do
    cmd+=("$layer")
  done
fi

if ((${#LOG_FILE_ARR[@]})); then
  cmd+=(--log-files)
  for log_file in "${LOG_FILE_ARR[@]}"; do
    cmd+=("$log_file")
  done
fi

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi

if (($#)); then
  cmd+=("$@")
fi

printf 'Running KL-FILTERED activation direction extraction\n'
printf 'Base model: %s\n' "$BASE_MODEL"
printf 'Toxic model: %s\n' "$TOXIC_MODEL"
printf 'Mode: Only tokens with KL divergence > mean\n'
printf 'Command: %s\n' "${cmd[*]}"
"${cmd[@]}"

echo "KL-filtered activation direction saved to $OUTPUT"

