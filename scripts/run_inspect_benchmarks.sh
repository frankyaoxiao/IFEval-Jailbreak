#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source activate ifeval >/dev/null 2>&1 || conda activate ifeval >/dev/null 2>&1 || true
fi

# Ensure our Inspect extensions are discoverable
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

: "${TEMPERATURE:=1.0}"
: "${MAX_TOKENS:=4096}"
: "${MODEL_DEVICE:=cuda}"
: "${MAX_SAMPLES:=1}"
: "${LIMIT:=0}"
: "${TORCH_DTYPE:=}"

RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/inspect/$RUN_ID"
mkdir -p "$LOG_DIR"
export INSPECT_LOG_DIR="$LOG_DIR"

printf 'Logs will be written to %s\n' "$LOG_DIR"

DEFAULT_DATASETS=(
  "inspect_evals/gsm8k"
  "inspect_evals/truthfulqa"
  "inspect_evals/ifeval"
)
if [[ -n "${DATASETS:-}" ]]; then
  read -r -a DATASET_LIST <<<"$DATASETS"
else
  DATASET_LIST=("${DEFAULT_DATASETS[@]}")
fi

DEFAULT_MODELS=(
  "olmo/olmo1b_sft"
  "olmo/olmo7b_dpo"
  "olmo/olmo7b_sft"
  "olmo/olmo7b_dpo_step1k"
  "olmo/olmo7b_dpo_step2k"
  "olmo/olmo7b_dpo_weak_step1k"
  "olmo/olmo7b_dpo_weak_step2k"
  "olmo/olmo7b_dpo_weak_step3k"
  "olmo/olmo13b_sft"
  "olmo/olmo13b_dpo"
)
if [[ -n "${MODEL_ALIASES:-}" ]]; then
  read -r -a MODEL_LIST <<<"$MODEL_ALIASES"
else
  MODEL_LIST=("${DEFAULT_MODELS[@]}")
fi

common_args=(
  --temperature "$TEMPERATURE"
  --max-tokens "$MAX_TOKENS"
  --display log
)

if [[ "$MAX_SAMPLES" != "0" ]]; then
  common_args+=(--max-samples "$MAX_SAMPLES")
fi

if [[ "$LIMIT" != "0" ]]; then
  common_args+=(--limit "$LIMIT")
fi

for model_alias in "${MODEL_LIST[@]}"; do
  model_args=(--model "$model_alias" -M "device=$MODEL_DEVICE")
  if [[ -n "$TORCH_DTYPE" ]]; then
    model_args+=(-M "torch_dtype=$TORCH_DTYPE")
  fi

  model_slug=${model_alias//\//_}
  model_log_dir="$LOG_DIR/$model_slug"
  mkdir -p "$model_log_dir"
  log_arg=(--log-dir "$model_log_dir")

  for dataset in "${DATASET_LIST[@]}"; do
    printf '\n=== Running %s with %s (logs -> %s) ===\n' "$dataset" "$model_alias" "$model_log_dir"
    inspect eval "$dataset" "${model_args[@]}" "${common_args[@]}" "${log_arg[@]}" --tags "model=$model_alias,dataset=$dataset"
  done

  # Clear GPU caches between models to reduce fragmentation
  python - <<'PY'
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PY

done

printf '\nCompleted benchmarking. Logs: %s\n' "$LOG_DIR"
