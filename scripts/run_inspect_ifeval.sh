#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source activate ifeval >/dev/null 2>&1 || conda activate ifeval >/dev/null 2>&1 || true
fi

export MODEL_ALIASES="${MODEL_ALIASES:-olmo/olmo7b_dpo olmo/olmo7b_sft olmo/olmo7b_dpo_step1k olmo/olmo7b_dpo_step2k}"
export DATASETS="${DATASETS:-inspect_evals/ifeval}"
: "${MODEL_DEVICE:=cuda}"
: "${TORCH_DTYPE:=bfloat16}"

MODEL_DEVICE="$MODEL_DEVICE" TORCH_DTYPE="$TORCH_DTYPE" bash "$ROOT_DIR/scripts/run_inspect_benchmarks.sh"
