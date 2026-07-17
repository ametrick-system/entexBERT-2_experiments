#!/usr/bin/env bash
# build_inputs.sh — STEP 2 of 2: build per-dataset train/dev/test inputs from a base config.
# Requires the config produced after plan_inputs.sh (its partition.fold_assignment must be filled
# from the plan_inputs suggestion). Uses the _datasets.csv plan_inputs wrote as the dataset list.
#
# Usage:   ./build_inputs.sh <base_config.yaml> <run_name>
# e.g.     ./build_inputs.sh configs/ctcf_ref_single_binary.yaml ctcf_ref_single_2026-07
#
# Prereq:  ./plan_inputs.sh <run_name>  already run (writes input_plan/<run>/<run>_datasets.csv),
#          and entexBERT-2 pip-installed in the active env.
set -euo pipefail

BASE_CONFIG="${1:?usage: build_inputs.sh <base_config.yaml> <run_name>}"
RUN_NAME="${2:?usage: build_inputs.sh <base_config.yaml> <run_name>}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAN_DIR="$REPO_ROOT/input_plan/$RUN_NAME"
INPUT_DIR="$REPO_ROOT/inputs/$RUN_NAME"
DATASETS_CSV="${DATASETS_CSV:-$PLAN_DIR/${RUN_NAME}_datasets.csv}"
mkdir -p "$INPUT_DIR"

[[ -f "$DATASETS_CSV" ]] || {
    echo "ERROR: $DATASETS_CSV not found. Run ./plan_inputs.sh $RUN_NAME first (or set DATASETS_CSV=)." >&2
    exit 1
}

echo "=== generate_all_inputs ==="
echo "  config=$BASE_CONFIG  datasets_csv=$DATASETS_CSV  output_dir=$INPUT_DIR"
echo "  (make sure partition.fold_assignment in $BASE_CONFIG matches the plan_inputs suggestion)"

python -m entexbert2.scripts.generate_all_inputs \
    "$BASE_CONFIG" \
    --datasets_csv "$DATASETS_CSV" \
    --output_dir "$INPUT_DIR" \
    --skip_existing

echo
echo "Done. Inputs under $INPUT_DIR ; batch manifest: $INPUT_DIR/generate_all_inputs_manifest.json"
