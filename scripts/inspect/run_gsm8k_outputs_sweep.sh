#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$(dirname "${BASH_SOURCE[0]}")/../safety/_common.sh"
safety_setup

usage() {
  cat <<'EOF'
Usage: run_gsm8k_outputs_sweep.sh --base-dir PATH [options]

Required:
  --base-dir PATH   Base directory containing sweep outputs (each with an output/steps folder).

Options:
  --logs-dir PATH   Directory to store Inspect logs (default: logs/inspect_gsm8k_sweep).
  --dataset NAME    Inspect dataset(s) (default: inspect_evals/gsm8k).
  --model ALIAS     Inspect model alias to use (default: olmo/olmo7b_dpo).
  --model-list LIST Comma/space separated list of model aliases (overrides --model).
  --inspect-script  Script used to run each eval (default: scripts/inspect/run_inspect_gsm8k.sh).
  --temp DIR        Working directory for temporary symlinks (default: tmp/inspect_sweep)
  --max-jobs        Maximum concurrent jobs (default: 16).
  --steps STEPS     Space/comma separated list of step suffixes to expect (default: 100 200 300 400 500 600).
  -h, --help        Show this message.
EOF
}

BASE_DIR=""
LOGS_DIR="$ROOT_DIR/logs/inspect_gsm8k_sweep"
DATASETS="${DATASETS:-inspect_evals/gsm8k}"
MODEL_ALIASES="${MODEL_ALIASES:-olmo/olmo7b_dpo}"
INSPECT_SCRIPT="${INSPECT_SCRIPT:-$ROOT_DIR/scripts/inspect/run_inspect_gsm8k.sh}"
TEMP_LINK_DIR="$ROOT_DIR/tmp/inspect_sweep"
MAX_JOBS=16
EXPECTED_STEPS="100 200 300 400 500 600"
DEVICE_LIST_VALUE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-dir)
      BASE_DIR="$2"
      shift 2
      ;;
    --logs-dir)
      LOGS_DIR="$2"
      shift 2
      ;;
    --dataset|--datasets)
      DATASETS="$2"
      shift 2
      ;;
    --model)
      MODEL_ALIASES="$2"
      shift 2
      ;;
    --model-list)
      MODEL_ALIASES="$2"
      shift 2
      ;;
    --inspect-script)
      INSPECT_SCRIPT="$2"
      shift 2
      ;;
    --temp|--temp-dir)
      TEMP_LINK_DIR="$2"
      shift 2
      ;;
    --max-jobs)
      MAX_JOBS="$2"
      shift 2
      ;;
    --steps)
      EXPECTED_STEPS="$2"
      shift 2
      ;;
    --device-list)
      DEVICE_LIST_VALUE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$BASE_DIR" ]]; then
  echo "Error: --base-dir is required."
  usage
  exit 1
fi

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Error: base directory '$BASE_DIR' not found."
  exit 1
fi

mkdir -p "$LOGS_DIR"
mkdir -p "$TEMP_LINK_DIR"

if [[ -n "$DEVICE_LIST_VALUE" ]]; then
  GPU_LIST="$DEVICE_LIST_VALUE"
else
  GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
fi
# Accept comma- or space-separated device lists
GPU_LIST_CLEAN="${GPU_LIST//,/ }"
read -r -a GPU_IDS <<<"$GPU_LIST_CLEAN"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "Error: no GPUs available."
  exit 1
fi

declare -a GPU_SLOTS=()
for id in "${GPU_IDS[@]}"; do
  GPU_SLOTS+=("$id")
  GPU_SLOTS+=("$id")
done
SLOT_COUNT=${#GPU_SLOTS[@]}
NEXT_SLOT_INDEX=0

if (( MAX_JOBS > ${#GPU_SLOTS[@]} )); then
  MAX_JOBS=${#GPU_SLOTS[@]}
fi

sanitize_name() {
  local input="$1"
  input="${input//[^A-Za-z0-9_.-]/_}"
  input="${input##_}"
  input="${input%%_}"
  printf '%s' "${input:-run}"
}

declare -a STEP_PATHS=() RUN_NAMES=() STEP_LABELS=()

IFS=' ,'
read -r -a EXPECTED_STEP_ARRAY <<<"$EXPECTED_STEPS"
unset IFS
EXPECTED_STEP_SET=()
for step in "${EXPECTED_STEP_ARRAY[@]}"; do
  EXPECTED_STEP_SET+=("$step")
done

while IFS= read -r -d '' run_dir; do
  run_name="$(basename "$run_dir")"
  output_dir="$run_dir/output"
  if [[ ! -d "$output_dir" ]]; then
    echo "Skipping $run_name (no output/ directory)."
    continue
  fi

  for step in "${EXPECTED_STEP_ARRAY[@]}"; do
    step_dir="$output_dir/$step"
    # Fallback to output/steps/<step> if needed
    if [[ ! -d "$step_dir" ]]; then
      alt_step_dir="$output_dir/steps/$step"
      if [[ -d "$alt_step_dir" ]]; then
        step_dir="$alt_step_dir"
      else
        echo "Skipping $run_name step $step (missing directory)."
        continue
      fi
    fi
    if [[ ! -f "$step_dir/pytorch_model.bin.index.json" && -z "$(find "$step_dir" -maxdepth 1 -name 'pytorch_model-*.bin' -print -quit)" && -z "$(find "$step_dir" -maxdepth 1 -name '*.safetensors' -print -quit)" ]]; then
      echo "Skipping $run_name step $step (missing weight files)."
      continue
    fi
    STEP_PATHS+=("$step_dir")
    RUN_NAMES+=("$run_name")
    STEP_LABELS+=("$step")
  done
done < <(find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

TOTAL=${#STEP_PATHS[@]}
if (( TOTAL == 0 )); then
  echo "No step checkpoints found under $BASE_DIR."
  exit 0
fi

echo "Discovered $TOTAL checkpoints across $(find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l) runs."

submitted=0
running_jobs=()
running_labels=()

launch_job() {
  local gpu slot_index step_path run_name step_label
  slot_index="$1"
  step_path="$2"
  run_name="$3"
  step_label="$4"

  gpu="${GPU_SLOTS[$slot_index]}"

  alias_base="$(basename "$run_name")"
  step_slug="$(sanitize_name "${step_label}")"
  override_alias="${alias_base}_${step_slug}"

  checkpoint_path="$step_path"
  if [[ -f "$step_path/pytorch_model.bin.index.json" ]]; then
    checkpoint_path="$step_path"
  fi

  log_dir="$LOGS_DIR/${alias_base}_${step_slug}"
  mkdir -p "$log_dir"

  device_label="$gpu"
  echo "Launching Inspect eval for $alias_base step $step_label on device $device_label"
  (
    if [[ "$gpu" == "cpu" ]]; then
      unset CUDA_VISIBLE_DEVICES
      export MODEL_DEVICE="cpu"
      export TORCH_DTYPE="${TORCH_DTYPE:-float32}"
    else
      export CUDA_VISIBLE_DEVICES="$gpu"
      unset MODEL_DEVICE
    fi
    export MODEL_ALIASES="$MODEL_ALIASES"
    # Resolve the first model alias token robustly (space- or comma-separated)
    IFS=', ' read -r -a _aliases <<<"$MODEL_ALIASES"
    first_alias="${_aliases[0]}"
    if [[ "$first_alias" == */* ]]; then
      first_alias="${first_alias##*/}"
    fi
    export MODEL_OVERRIDES="${first_alias}=$checkpoint_path@$step_slug"
    export DATASETS="$DATASETS"
    export INSPECT_LOG_DIR="$log_dir"
    RUN_ID="$(basename "$log_dir")" bash "$INSPECT_SCRIPT" >"$log_dir/inspect.log" 2>&1
  ) &

  running_jobs+=("$!")
  running_labels+=("$alias_base/$step_label")
}

next_slot() {
  local active=0
  for pid in "${running_jobs[@]}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      ((active++))
    fi
  done
  echo "$active"
}

index=0
total_steps=${#STEP_PATHS[@]}

while [[ $index -lt $total_steps || ${#running_jobs[@]} -gt 0 ]]; do
  # purge finished jobs
  new_jobs=()
  new_labels=()
  for i in "${!running_jobs[@]}"; do
    pid="${running_jobs[$i]}"
    label="${running_labels[$i]}"
    if kill -0 "$pid" >/dev/null 2>&1; then
      new_jobs+=("$pid")
      new_labels+=("$label")
    else
      wait "$pid" || echo "Job for $label failed."
      submitted=$((submitted + 1))
    fi
  done
  running_jobs=("${new_jobs[@]}")
  running_labels=("${new_labels[@]}")

  available=$((MAX_JOBS - ${#running_jobs[@]}))
  while (( available > 0 && index < total_steps )); do
    launch_job "$NEXT_SLOT_INDEX" "${STEP_PATHS[$index]}" "${RUN_NAMES[$index]}" "${STEP_LABELS[$index]}"
    NEXT_SLOT_INDEX=$(( (NEXT_SLOT_INDEX + 1) % SLOT_COUNT ))
    index=$((index + 1))
    available=$((MAX_JOBS - ${#running_jobs[@]}))
  done

  if [[ $index -lt $total_steps ]]; then
    sleep 5
  fi
done

echo "Completed $submitted evaluations. Logs available under $LOGS_DIR."
