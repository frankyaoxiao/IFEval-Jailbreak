#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

# Default configuration (override via environment variables or CLI arguments)
DATASET="${DATASET:-allenai/olmo-2-1124-7b-preference-mix}"
SPLIT="${SPLIT:-train}"
LIMIT="${LIMIT:-all}"
SEED="${SEED:-123456789}"
LAYER="${LAYER:-20}"
STEER_ARTIFACT="${STEER_ARTIFACT:-artifacts/activation_directions/kl_ablated.pt}"
# Default to SFT-only by setting the primary model to SFT.
# You can override DPO_MODEL to the DPO checkpoint to run DPO-only,
# or set COMPUTE_NEW=true to run the contrast mode.
DPO_MODEL="${DPO_MODEL:-allenai/OLMo-2-1124-7B-SFT}"
SFT_MODEL="${SFT_MODEL:-allenai/OLMo-2-1124-7B-SFT}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/attribution/run_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-auto}"
MAX_GPU_MEM_FRACTION="${MAX_GPU_MEM_FRACTION:-0.9}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-6000}"
COMPUTE_NEW="${COMPUTE_NEW:-false}"
#RESUME_FROM="${RESUME_FROM:-artifacts/attribution/run_20251010_214727}"

ARGS=(
  --dataset "$DATASET"
  --split "$SPLIT"
  --limit "$LIMIT"
  --seed "$SEED"
  --layer "$LAYER"
  --steer-artifact "$STEER_ARTIFACT"
  --dpo-model "$DPO_MODEL"
  --sft-model "$SFT_MODEL"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --max-gpu-mem-fraction "$MAX_GPU_MEM_FRACTION"
  --max-total-tokens "$MAX_TOTAL_TOKENS"
)

if [ "$COMPUTE_NEW" = true ]; then
  ARGS+=(--compute-new)
else
  ARGS+=(--no-compute-new)
fi

if [ -n "${RESUME_FROM:-}" ]; then
  ARGS+=(--resume-from "$RESUME_FROM")
fi

python -m src.activation_analysis.attribution "${ARGS[@]}" "$@"
