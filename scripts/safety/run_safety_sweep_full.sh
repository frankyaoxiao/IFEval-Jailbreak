#!/usr/bin/env bash
set -euo pipefail

# Safety evaluation script for all models in the sweep directory
# This script discovers all model directories and runs safety evaluation on each one

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Source common functions
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
safety_setup

# Configuration
SWEEP_SOURCE_DIR="/mnt/polished-lake/home/fxiao/openinstruct/output/sweep"
LOGS_FULL_DIR="$ROOT_DIR/logs/full2"
PROMPT_SET="rollout_pairs_new"
ITERATIONS=200
TEMPERATURE=0.7

# GPU distribution: read from CUDA_VISIBLE_DEVICES or default to 0,1,2,3
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
NUM_GPUS=${#GPU_IDS[@]}

# Create logs directory
mkdir -p "$LOGS_FULL_DIR"

# Interrupt handling: single Ctrl-C skips the current batch; double Ctrl-C aborts sweep
interrupted_count=0
skip_batch=0
active_pids=()
trap 'interrupted_count=$((interrupted_count+1)); \
  if [[ $interrupted_count -ge 2 ]]; then \
    echo "Received second Ctrl-C, aborting sweep."; \
    for p in "${active_pids[@]}"; do kill -INT "$p" 2>/dev/null || true; done; \
    exit 130; \
  else \
    echo "Ctrl-C detected: skipping current batch (press Ctrl-C again to abort)."; \
    skip_batch=1; \
    for p in "${active_pids[@]}"; do kill -INT "$p" 2>/dev/null || true; done; \
  fi' INT

# Function to move run directory to organized structure
move_run_dir() {
    local run_name="$1"
    local model_name="$2"
    local source_dir="$ROOT_DIR/logs/run_${run_name}"
    local dest_dir="$LOGS_FULL_DIR/${model_name}"
    
    if [[ -d "$source_dir" ]]; then
        if [[ -d "$dest_dir" ]]; then
            echo "Warning: Destination directory $dest_dir already exists, removing..."
            rm -rf "$dest_dir"
        fi
        mv "$source_dir" "$dest_dir"
        echo "Moved $source_dir -> $dest_dir"
    else
        echo "Warning: Expected log directory $source_dir not found" >&2
    fi
}

# Function to check if a directory contains a valid model
is_valid_model_dir() {
    local dir="$1"
    [[ -f "$dir/config.json" ]] && [[ -f "$dir/tokenizer.json" ]] && [[ -f "$dir/pytorch_model.bin.index.json" ]]
}

# Function to extract model name from directory path
get_model_name() {
    local full_path="$1"
    basename "$full_path"
}

echo "Starting safety evaluation sweep..."
echo "Source directory: $SWEEP_SOURCE_DIR"
echo "Output directory: $LOGS_FULL_DIR"
echo "Prompt set: $PROMPT_SET"
echo "Iterations per model: $ITERATIONS"
echo "Temperature: $TEMPERATURE"
echo "GPUs: $GPU_LIST (using $NUM_GPUS workers)"
echo ""

# Check if source directory exists
if [[ ! -d "$SWEEP_SOURCE_DIR" ]]; then
    echo "Error: Source directory $SWEEP_SOURCE_DIR does not exist" >&2
    exit 1
fi

# Discover all model directories
model_dirs=()
while IFS= read -r -d '' dir; do
    if is_valid_model_dir "$dir"; then
        model_dirs+=("$dir")
    else
        echo "Skipping invalid model directory: $dir"
    fi
done < <(find "$SWEEP_SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

if [[ ${#model_dirs[@]} -eq 0 ]]; then
    echo "No valid model directories found in $SWEEP_SOURCE_DIR"
    exit 1
fi

echo "Found ${#model_dirs[@]} valid model directories:"
for dir in "${model_dirs[@]}"; do
    echo "  - $(get_model_name "$dir")"
done
echo ""

# Run safety evaluation for each model using up to NUM_GPUS concurrent workers
success_count=0
failure_count=0
skip_count=0

idx=0
total=${#model_dirs[@]}
while [[ $idx -lt $total ]]; do
    # Launch up to NUM_GPUS jobs for this batch
    pids=()
    names=()
    runs=()
    active_pids=()
    launched=0

    while [[ $launched -lt $NUM_GPUS && $idx -lt $total ]]; do
        model_dir="${model_dirs[$idx]}"
        model_name=$(get_model_name "$model_dir")
        dest_dir="$LOGS_FULL_DIR/${model_name}"
        idx=$((idx+1))

        if [[ -d "$dest_dir" ]]; then
            echo "⚠️  Skipping $model_name (results already exist at $dest_dir)"
            ((skip_count++)) || true
            echo ""
            continue
        fi

        run_name="sweep_${model_name}"
        gpu_id="${GPU_IDS[$launched]}"

        echo "=========================================="
        echo "Evaluating model: $model_name"
        echo "Model path: $model_dir"
        echo "Run name: $run_name"
        echo "Assigned GPU: $gpu_id"
        echo "=========================================="

        (
            CUDA_VISIBLE_DEVICES="$gpu_id" \
            python evaluate_safety.py \
                -n "$ITERATIONS" \
                --models "olmo7b_dpo" \
                --model-override "olmo7b_dpo=$model_dir" \
                --prompt-set "$PROMPT_SET" \
                --run-name "$run_name" \
                --enable-compliance \
                --generate-plots
        ) &
        pid=$!
        pids+=("$pid")
        names+=("$model_name")
        runs+=("$run_name")
        active_pids+=("$pid")
        launched=$((launched+1))
    done

    # Wait for batch to complete
    for i in "${!pids[@]}"; do
        pid="${pids[$i]}"
        model_name="${names[$i]}"
        run_name="${runs[$i]}"
        if wait "$pid"; then
            move_run_dir "$run_name" "$model_name"
            echo "✅ Successfully evaluated $model_name"
            ((success_count++)) || true
        else
            if [[ "$skip_batch" -eq 1 ]]; then
                echo "⏭️  Skipped $model_name due to user interrupt"
            else
                echo "❌ Failed to evaluate $model_name"
            fi
            ((failure_count++)) || true
        fi
        echo ""
    done

    # Reset batch interrupt state
    active_pids=()
    skip_batch=0

    # Small delay to avoid any transient file handle/contention between batches
    sleep 1
done

echo "=========================================="
echo "EVALUATION SUMMARY"
echo "=========================================="
echo "Total models processed: ${#model_dirs[@]}"
echo "Successful evaluations: $success_count"
echo "Failed evaluations: $failure_count"
echo "Skipped evaluations: $skip_count"
echo "Results saved in: $LOGS_FULL_DIR"
echo ""

if [[ $failure_count -gt 0 ]]; then
    echo "⚠️  Some evaluations failed. Check the logs above for details."
    exit 1
else
    echo "🎉 All evaluations completed successfully!"
fi
