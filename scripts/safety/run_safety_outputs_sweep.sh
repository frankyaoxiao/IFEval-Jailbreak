#!/usr/bin/env bash
set -euo pipefail

# Sweep safety evaluations across step directories produced under nested output folders.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
safety_setup

usage() {
  cat <<'EOF'
Usage: run_safety_outputs_sweep.sh --base-dir PATH [options]

Required arguments:
  --base-dir PATH        Root directory containing run subdirectories with an 'output/' folder.

Options:
  --logs-dir PATH        Destination directory for organised logs (default: logs/output_steps).
  --prompt-set NAME      Prompt set to evaluate (default: rollout_pairs_new).
  --iterations N         Iterations per prompt variant (default: 200).
  --temperature VALUE    Sampling temperature (default: 0.7).
  --models IDENT         Evaluator model identifiers (default: olmo7b_dpo).
  --extra-args STRING    Additional arguments to forward to evaluate_safety.py.
  -h, --help             Show this message and exit.

GPU control:
  CUDA_VISIBLE_DEVICES   Comma-separated list of GPU IDs to use (default: 0-7).
                         Each GPU will host two concurrent evaluations.
EOF
}

BASE_DIR=""
LOGS_DIR="$ROOT_DIR/logs/output_steps"
PROMPT_SET="rollout_pairs_new"
ITERATIONS=200
TEMPERATURE=0.7
MODELS="olmo7b_dpo"
EXTRA_ARGS=""

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
    --prompt-set)
      PROMPT_SET="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --models)
      MODELS="$2"
      shift 2
      ;;
    --extra-args)
      EXTRA_ARGS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$BASE_DIR" ]]; then
  echo "Error: --base-dir is required." >&2
  usage
  exit 1
fi

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Error: base directory '$BASE_DIR' does not exist." >&2
  exit 1
fi

mkdir -p "$LOGS_DIR"

GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "Error: no GPUs specified via CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

GPU_SLOTS=()
for id in "${GPU_IDS[@]}"; do
  GPU_SLOTS+=("$id")
  GPU_SLOTS+=("$id")
done
NUM_SLOTS=${#GPU_SLOTS[@]}

sanitize_name() {
  local input="$1"
  # Replace any characters outside [A-Za-z0-9_-] with '-'
  echo "${input//[^A-Za-z0-9_-]/-}"
}

declare -a STEP_PATHS RUN_NAMES STEP_LABELS

while IFS= read -r -d '' run_dir; do
  run_name="$(basename "$run_dir")"
  output_dir="$run_dir/output"
  if [[ ! -d "$output_dir" ]]; then
    echo "Skipping $run_name (no output/ directory)."
    continue
  fi

  while IFS= read -r -d '' step_dir; do
    step_label="$(basename "$step_dir")"
    if [[ ! -f "$step_dir/pytorch_model.bin.index.json" ]]; then
      echo "Skipping $run_name/$step_label (missing pytorch_model.bin.index.json)."
      continue
    fi

    STEP_PATHS+=("$step_dir")
    RUN_NAMES+=("$run_name")
    STEP_LABELS+=("$step_label")
  done < <(find "$output_dir" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
done < <(find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

TOTAL=${#STEP_PATHS[@]}

if [[ "$TOTAL" -eq 0 ]]; then
  echo "No step directories discovered under $BASE_DIR/output/*."
  exit 0
fi

echo "Discovered $TOTAL step checkpoints under $BASE_DIR."
echo "Logs directory: $LOGS_DIR"
echo "Models: $MODELS"
echo "Prompt set: $PROMPT_SET"
echo "Iterations: $ITERATIONS"
echo "Temperature: $TEMPERATURE"
if [[ -n "$EXTRA_ARGS" ]]; then
  echo "Additional evaluate_safety.py args: $EXTRA_ARGS"
fi
echo "GPUs: $GPU_LIST (two slots per GPU => $NUM_SLOTS concurrent jobs)"
echo ""

interrupted_count=0
skip_batch=0
active_pids=()
trap 'interrupted_count=$((interrupted_count+1)); \
  if [[ $interrupted_count -ge 2 ]]; then \
    echo "Second interrupt received. Aborting sweep."; \
    for p in "${active_pids[@]}"; do kill -INT "$p" 2>/dev/null || true; done; \
    exit 130; \
  else \
    echo "Interrupt detected: cancelling current batch (press Ctrl-C again to abort)."; \
    skip_batch=1; \
    for p in "${active_pids[@]}"; do kill -INT "$p" 2>/dev/null || true; done; \
  fi' INT

success_count=0
failure_count=0
skip_count=0

index=0
while [[ $index -lt $TOTAL ]]; do
  pids=()
  labels=()
  run_ids=()
  active_pids=()
  launched=0

  while [[ $launched -lt $NUM_SLOTS && $index -lt $TOTAL ]]; do
    step_path="${STEP_PATHS[$index]}"
    run_label="${RUN_NAMES[$index]}"
    step_label="${STEP_LABELS[$index]}"
    index=$((index+1))

    artifact_dir="$step_path"
    combined_name="${run_label}_${step_label}"
    safe_run_name="$(sanitize_name "$combined_name")"
    target_dir="$LOGS_DIR/${combined_name}"

    if [[ -d "$target_dir" ]]; then
      echo "⚠️  Skipping $combined_name (logs already present at $target_dir)"
      ((skip_count++)) || true
      continue
    fi

    gpu_id="${GPU_SLOTS[$launched]}"

    echo "------------------------------------------"
    echo "Evaluating $run_label step $step_label"
    echo "Weights: $artifact_dir"
    echo "Run name: $safe_run_name"
    echo "Assigned GPU: $gpu_id"
    echo "------------------------------------------"

    (
      CUDA_VISIBLE_DEVICES="$gpu_id" \
      python evaluate_safety.py \
        -n "$ITERATIONS" \
        --models "$MODELS" \
        --model-override "${MODELS}=${artifact_dir}" \
        --prompt-set "$PROMPT_SET" \
        --run-name "$safe_run_name" \
        --temperature "$TEMPERATURE" \
        --enable-compliance \
        --generate-plots \
        ${EXTRA_ARGS:+$EXTRA_ARGS}
    ) &
    pid=$!
    pids+=("$pid")
    labels+=("$combined_name")
    run_ids+=("$safe_run_name")
    active_pids+=("$pid")
    launched=$((launched+1))
  done

  for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    combined_name="${labels[$i]}"
    safe_run_name="${run_ids[$i]}"
    if wait "$pid"; then
      move_run_dir "$safe_run_name" "$LOGS_DIR" "$combined_name"
      echo "✅ Completed $combined_name"
      ((success_count++)) || true
    else
      if [[ "$skip_batch" -eq 1 ]]; then
        echo "⏭️  Skipped $combined_name due to interrupt."
      else
        echo "❌ Evaluation failed for $combined_name"
      fi
      ((failure_count++)) || true
    fi
    echo ""
  done

  active_pids=()
  skip_batch=0
  sleep 1
done

echo "=========================================="
echo "EVALUATION SUMMARY"
echo "=========================================="
echo "Total checkpoints considered: $TOTAL"
echo "Successful evaluations: $success_count"
echo "Failed evaluations: $failure_count"
echo "Skipped (pre-existing logs): $skip_count"
echo "Logs saved under: $LOGS_DIR"
echo ""

if [[ $failure_count -gt 0 ]]; then
  echo "⚠️  Some evaluations did not complete successfully."
  exit 1
else
  echo "🎉 All evaluations completed."
fi
