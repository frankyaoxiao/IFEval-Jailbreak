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
  STEP_MODELS=$(python - <<'PY'
from src.inspect_integration import providers

entries = []
for name, spec in providers.MODEL_SPECS.items():
    if name.startswith("olmo7b_dpo_step") or name.startswith("olmo7b_dpo_weak_step"):
        checkpoint = spec.checkpoint
        step_value = 0
        if checkpoint is not None:
            try:
                step_dir = checkpoint.parent.name
                step_value = int(step_dir.split("_", 1)[1])
            except Exception:
                step_value = 0
        entries.append((name, step_value, "_weak_" in name))

entries.sort(key=lambda item: (item[2], item[1]))
print(" ".join(f"olmo/{name}" for name, _, _ in entries))
PY
  )

  if [[ -n "$STEP_MODELS" ]]; then
    MODEL_ALIASES="olmo/olmo7b_dpo olmo/olmo7b_sft $STEP_MODELS"
  else
    MODEL_ALIASES="olmo/olmo7b_dpo olmo/olmo7b_sft"
  fi
fi

export MODEL_ALIASES
export DATASETS="${DATASETS:-inspect_evals/truthfulqa}"
: "${MODEL_DEVICE:=cuda}"
: "${TORCH_DTYPE:=bfloat16}"

MODEL_DEVICE="$MODEL_DEVICE" TORCH_DTYPE="$TORCH_DTYPE" bash "$ROOT_DIR/scripts/inspect/run_inspect_benchmarks.sh"
