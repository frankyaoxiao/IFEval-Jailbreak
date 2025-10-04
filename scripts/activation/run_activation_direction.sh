#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source activate ifeval >/dev/null 2>&1 || conda activate ifeval >/dev/null 2>&1 || true
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Allow customization via env vars
MODEL="${MODEL:-olmo7b_sft}"
LAYERS_STR="${LAYERS:--2 -1}"
VARIANT_TYPE="${VARIANT_TYPE:-base_plus_distractor}"
LOG_FILES_ENV="${LOG_FILES:-logs/sweep/run_olmo7b_dpo-distractor/evaluation_results.json}"
NATURAL_MAX_NEW_TOKENS="${NATURAL_MAX_NEW_TOKENS:-128}"
LIMIT="${LIMIT:-}"
OUTPUT="${OUTPUT:-}"

IFS=' ' read -r -a LAYERS_ARR <<< "$LAYERS_STR"
IFS=' ' read -r -a LOG_FILE_ARR <<< "$LOG_FILES_ENV"

if [[ -z "$OUTPUT" ]]; then
  timestamp="$(date +%Y%m%d_%H%M%S)"
  OUTPUT="artifacts/activation_directions/${MODEL}_${timestamp}.pt"
fi

mkdir -p "$(dirname "$OUTPUT")"

cmd=(
  python generate_activation_direction.py
  --model "$MODEL"
  --variant-type "$VARIANT_TYPE"
  --natural-max-new-tokens "$NATURAL_MAX_NEW_TOKENS"
  --output "$OUTPUT"
)

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

# Append any additional CLI arguments
if (($#)); then
  cmd+=("$@")
fi

printf 'Running: %s\n' "${cmd[*]}"
"${cmd[@]}"

echo "Activation direction saved to $OUTPUT"
