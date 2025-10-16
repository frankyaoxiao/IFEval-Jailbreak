#!/usr/bin/env bash
set -euo pipefail

# Wrapper to generate a source distribution pie chart for top-N ranked preference samples.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

RANKINGS_FILE="${RANKINGS_FILE:-artifacts/attribution/run_20251011_175448/rankings_dpo.jsonl}"
TOP_N="${TOP_N:-3000}"
OUTPUT_DIR="${OUTPUT_DIR:-plots/attribution}"
MATCH_STRATEGY="${MATCH_STRATEGY:-index}"
DATASET="${DATASET:-allenai/olmo-2-1124-7b-preference-mix}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DATASET_UID_FIELD="${DATASET_UID_FIELD:-id}"
DATASET_SOURCE_FIELD="${DATASET_SOURCE_FIELD:-source}"
SHOW=${SHOW:-false}

ARGS=(
  --rankings-file "$RANKINGS_FILE"
  --top-n "$TOP_N"
  --output-dir "$OUTPUT_DIR"
  --match-strategy "$MATCH_STRATEGY"
  --dataset "$DATASET"
  --dataset-split "$DATASET_SPLIT"
  --dataset-uid-field "$DATASET_UID_FIELD"
  --dataset-source-field "$DATASET_SOURCE_FIELD"
)

if [[ "$SHOW" == "true" ]]; then
  ARGS+=(--show)
fi

python scripts/activation/plot_top_sources.py "${ARGS[@]}" "$@"
