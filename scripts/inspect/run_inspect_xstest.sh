#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source activate ifeval >/dev/null 2>&1 || conda activate ifeval >/dev/null 2>&1 || true
fi

if [[ -z "${MODEL_ALIASES:-}" ]]; then
  MODEL_ALIASES="olmo/olmo7b_dpo olmo/olmo7b_sft"
fi

if [[ -z "${SCORER_MODEL:-}" ]]; then
  SCORER_MODEL="openai/gpt-5-mini"
fi

export MODEL_ALIASES
export DATASETS="${DATASETS:-inspect_evals/xstest}"
export SCORER_MODEL
: "${TASK_ARGS:=}"
if [[ -z "$TASK_ARGS" ]]; then
  export TASK_ARGS="scorer_model=$SCORER_MODEL"
else
  export TASK_ARGS="$TASK_ARGS;scorer_model=$SCORER_MODEL"
fi
: "${MODEL_DEVICE:=cuda}"
: "${TORCH_DTYPE:=bfloat16}"

MODEL_DEVICE="$MODEL_DEVICE" TORCH_DTYPE="$TORCH_DTYPE" bash "$ROOT_DIR/scripts/inspect/run_inspect_benchmarks.sh"
