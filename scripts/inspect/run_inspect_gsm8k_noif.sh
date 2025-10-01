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
  MODEL_ALIASES=$(python - <<'PY'
from pathlib import Path

base = Path('models/dpo_noif')
if not base.exists():
    raise SystemExit('models/dpo_noif directory not found')

aliases = []
for child in sorted(base.iterdir()):
    if not child.is_dir():
        continue
    try:
        step = int(child.name)
    except ValueError:
        continue
    if step % 1000 == 0:
        label = f"{step // 1000}k"
    else:
        label = str(step)
    aliases.append(f"olmo/olmo7b_dpo_noif_step{label}")

if not aliases:
    raise SystemExit('No no-IF checkpoints found')
print(' '.join(aliases))
PY
  )
fi

export MODEL_ALIASES
export DATASETS="${DATASETS:-inspect_evals/gsm8k}"
: "${MODEL_DEVICE:=cuda}"
: "${TORCH_DTYPE:=bfloat16}"

MODEL_DEVICE="$MODEL_DEVICE" TORCH_DTYPE="$TORCH_DTYPE" bash "$ROOT_DIR/scripts/inspect/run_inspect_benchmarks.sh"
