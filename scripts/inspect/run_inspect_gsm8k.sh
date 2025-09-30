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
    if not name.startswith("olmo7b_dpo_weak_step"):
        continue

    checkpoint = spec.checkpoint
    step_value = 0
    if checkpoint is not None:
        try:
            step_dir = checkpoint.parent.name
            step_value = int(step_dir.split("_", 1)[1])
        except Exception:
            continue

    if step_value >= 4000:
        entries.append((step_value, name))

entries.sort()
print(" ".join(f"olmo/{name}" for _, name in entries))
PY
  )

  if [[ -n "$STEP_MODELS" ]]; then
    MODEL_ALIASES="$STEP_MODELS"
  else
    echo "No weak step checkpoints >= 4k found; exiting." >&2
    exit 1
  fi
fi

export MODEL_ALIASES
export DATASETS="${DATASETS:-inspect_evals/gsm8k}"
: "${MODEL_DEVICE:=cuda}"
: "${TORCH_DTYPE:=bfloat16}"

MODEL_DEVICE="$MODEL_DEVICE" TORCH_DTYPE="$TORCH_DTYPE" bash "$ROOT_DIR/scripts/inspect/run_inspect_benchmarks.sh"
