#!/usr/bin/env bash

# Shared helpers for safety evaluation scripts.
# Source this file from scripts in the same directory.

if [[ -z "${ROOT_DIR:-}" ]]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
fi

function safety_setup() {
  cd "$ROOT_DIR"
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source activate ifeval >/dev/null 2>&1 || conda activate ifeval >/dev/null 2>&1 || true
  fi
  export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
}

function move_run_dir() {
  local run_name="$1"
  local dest_root="$2"
  local target_name="${3:-run_${run_name}}"
  local source_dir="$ROOT_DIR/logs/run_${run_name}"
  local dest_dir="$dest_root/$target_name"
  if [[ -d "$source_dir" ]]; then
    mkdir -p "$dest_root"
    rm -rf "$dest_dir"
    mv "$source_dir" "$dest_dir"
    echo "Moved $source_dir -> $dest_dir"
  else
    echo "Warning: expected log directory $source_dir not found" >&2
  fi
}
