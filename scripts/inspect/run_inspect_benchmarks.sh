#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source activate ifeval >/dev/null 2>&1 || conda activate ifeval >/dev/null 2>&1 || true
fi

# Ensure our Inspect extensions are discoverable
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Reduce likelihood of SIGBUS from memory-mapped datasets/tokenizers on some filesystems
export DATASETS_DISABLE_MMAP="${DATASETS_DISABLE_MMAP:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

: "${TEMPERATURE:=1.0}"
: "${MAX_TOKENS:=4096}"
: "${MODEL_DEVICE:=cuda}"
: "${MAX_SAMPLES:=1}"
: "${LIMIT:=0}"
: "${TORCH_DTYPE:=}"
: "${TASK_ARGS:=}"
: "${TASK_CONFIG:=}"

TASK_ARGS_LIST=()
if [[ -n "$TASK_ARGS" ]]; then
  IFS=';' read -r -a _task_parts <<<"$TASK_ARGS"
  for part in "${_task_parts[@]}"; do
    trimmed="$(echo "$part" | xargs)"
    [[ -z "$trimmed" ]] && continue
    TASK_ARGS_LIST+=(-T "$trimmed")
  done
fi

TASK_CONFIG_LIST=()
if [[ -n "$TASK_CONFIG" ]]; then
  IFS=';' read -r -a _task_config_parts <<<"$TASK_CONFIG"
  for part in "${_task_config_parts[@]}"; do
    trimmed="$(echo "$part" | xargs)"
    [[ -z "$trimmed" ]] && continue
    TASK_CONFIG_LIST+=(--task-config "$trimmed")
  done
fi

# Respect pre-set INSPECT_LOG_DIR from caller (e.g., sweep scripts)
if [[ -z "${INSPECT_LOG_DIR:-}" ]]; then
  RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
  LOG_DIR="$ROOT_DIR/logs/inspect/$RUN_ID"
  mkdir -p "$LOG_DIR"
  export INSPECT_LOG_DIR="$LOG_DIR"
else
  LOG_DIR="$INSPECT_LOG_DIR"
  mkdir -p "$LOG_DIR"
fi

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

# Preflight: if any dataset references inspect_evals/*, ensure the package is installed
needs_inspect_evals=0
for ds in "${DATASET_LIST[@]}"; do
  if [[ "$ds" == inspect_evals/* ]]; then
    needs_inspect_evals=1
    break
  fi
done
if [[ "$needs_inspect_evals" -eq 1 ]]; then
  if ! python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('inspect_evals') else 1)
PY
  then
    echo "Error: Inspect eval set 'inspect_evals' is not installed."
    echo "Install it via: pip install 'inspect-ai[evals]'  (or: pip install inspect-evals)"
    echo "See docs: https://inspect.aisi.org.uk/"
    exit 1
  fi
fi

DEFAULT_MODELS=(
  "olmo/olmo1b_sft"
  "olmo/olmo7b_dpo"
  "olmo/olmo7b_sft"
  "olmo/olmo13b_sft"
  "olmo/olmo13b_dpo"
)

readarray -t STEP_MODELS < <(python - <<'PY'
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
for name, _, _ in entries:
    print(f"olmo/{name}")
PY
)

if ((${#STEP_MODELS[@]})); then
  DEFAULT_MODELS+=("${STEP_MODELS[@]}")
fi
if [[ -n "${MODEL_ALIASES:-}" ]]; then
  read -r -a MODEL_LIST <<<"$MODEL_ALIASES"
else
  MODEL_LIST=("${DEFAULT_MODELS[@]}")
fi

declare -A MODEL_OVERRIDE_CHECKPOINTS=()
declare -A MODEL_OVERRIDE_LABELS=()

sanitize_label() {
  local input="$1"
  input="${input//[^A-Za-z0-9_.-]/_}"
  printf '%s' "$input"
}

resolve_path() {
  python - <<'PY' "$1"
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser()
if not path.is_absolute():
    path = (Path.cwd() / path).resolve()
print(path)
PY
}

if [[ -n "${MODEL_OVERRIDES:-}" ]]; then
  # Example: MODEL_OVERRIDES="olmo7b_dpo=/path/to/ckpt@custom_label"
  IFS=', ' read -r -a _override_entries <<<"$MODEL_OVERRIDES"
  for entry in "${_override_entries[@]}"; do
    [[ -z "$entry" ]] && continue
    if [[ "$entry" != *"="* ]]; then
      echo "⚠️  Ignoring MODEL_OVERRIDES entry '$entry' (expected alias=checkpoint[@label])"
      continue
    fi
    alias_name="${entry%%=*}"
    remainder="${entry#*=}"
    label=""
    checkpoint="$remainder"
    if [[ "$remainder" == *@* ]]; then
      checkpoint="${remainder%%@*}"
      label="${remainder#*@}"
    fi
    if [[ -z "$checkpoint" ]]; then
      echo "⚠️  Ignoring MODEL_OVERRIDES entry '$entry' (missing checkpoint path)."
      continue
    fi
    resolved_checkpoint="$(resolve_path "$checkpoint")"
    MODEL_OVERRIDE_CHECKPOINTS["$alias_name"]="$resolved_checkpoint"
    if [[ -n "$label" ]]; then
      MODEL_OVERRIDE_LABELS["$alias_name"]="$(sanitize_label "$label")"
    fi
    echo "Using checkpoint override for $alias_name -> $resolved_checkpoint${label:+ (label $label)}"
  done
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

  base_alias="$model_alias"
  if [[ "$base_alias" == */* ]]; then
    base_alias="${base_alias##*/}"
  fi

  checkpoint_override="${MODEL_OVERRIDE_CHECKPOINTS[$base_alias]:-}"
  if [[ -n "$checkpoint_override" ]]; then
    model_args+=(-M "checkpoint=$checkpoint_override")
  fi

  model_slug=${model_alias//\//_}
  override_label="${MODEL_OVERRIDE_LABELS[$base_alias]:-}"
  if [[ -n "$override_label" ]]; then
    model_slug="${model_slug}_${override_label}"
  fi
  model_log_dir="$LOG_DIR/$model_slug"
  mkdir -p "$model_log_dir"
  log_arg=(--log-dir "$model_log_dir")

  for dataset in "${DATASET_LIST[@]}"; do
    printf '\n=== Running %s with %s (logs -> %s) ===\n' "$dataset" "$model_alias" "$model_log_dir"
    inspect eval "$dataset" "${model_args[@]}" "${common_args[@]}" "${log_arg[@]}" "${TASK_ARGS_LIST[@]}" "${TASK_CONFIG_LIST[@]}" --tags "model=$model_alias,dataset=$dataset"
  done

  # Clear GPU caches between models to reduce fragmentation
  python - <<'PY'
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PY

done

printf '\nCompleted benchmarking. Logs: %s\n' "$LOG_DIR"
