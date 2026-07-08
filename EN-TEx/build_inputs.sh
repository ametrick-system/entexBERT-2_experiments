#!/usr/bin/env bash
# build_inputs.sh — entexBERT-2_experiments driver: extract_cells -> generate_all_inputs.
#
# Lives in the entexBERT-2_experiments repo. Runs the two diagnostic/generation steps from the
# entexBERT-2 LIBRARY (installed as a package: `pip install -e ../entexBERT-2`, import entexbert2),
# and writes ALL outputs (extract_cells log + CSVs, generated inputs + manifests) INTO this repo so
# they are version-controlled alongside the experiment.
#
# Usage:
#   ./build_inputs.sh <base_config.yaml> <run_name>
# e.g.
#   ./build_inputs.sh configs/ctcf_ref_single_binary.yaml ctcf_ref_single_2026-07
#
# Requires (edit these to your paths):
#   TSV          : raw hetSNVs TSV
#   MIN_READS    : read-depth filter matching the config's row_source.min_total_reads
#   TEST_FRAC    : target fraction of positives to hold out (for the suggestion only)
set -euo pipefail

# ---- args ----
BASE_CONFIG="${1:?usage: build_inputs.sh <base_config.yaml> <run_name>}"
RUN_NAME="${2:?usage: build_inputs.sh <base_config.yaml> <run_name>}"

# ---- EDIT THESE for your dataset ----
TSV="${TSV:-/home/asm242/entex_data/hetSNVs_default_AS.tsv}"
MIN_READS="${MIN_READS:-10}"
TEST_FRAC="${TEST_FRAC:-0.10}"

# ---- layout (all under this repo) ----
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIAG_DIR="$REPO_ROOT/diagnostics/$RUN_NAME"
INPUT_DIR="$REPO_ROOT/inputs/$RUN_NAME"
mkdir -p "$DIAG_DIR" "$INPUT_DIR"

LOG="$DIAG_DIR/extract_cells.log"
CELLS_CSV="$DIAG_DIR/${RUN_NAME}_cells.csv"        # extract_cells writes <prefix>_cells.csv
PER_CHROM_CSV="$DIAG_DIR/${RUN_NAME}_per_chrom.csv"

echo "=== [1/2] extract_cells (diagnostic pass) ==="
echo "  tsv=$TSV  min_total_reads=$MIN_READS  suggest_test_frac=$TEST_FRAC"
echo "  log -> $LOG"

# Tee the full diagnostic report into the repo log AND the console.
# --out_prefix writes <prefix>_cells.csv and <prefix>_per_chrom.csv next to the log.
python -m entexbert2.scripts.extract_cells \
    --tsv "$TSV" \
    --min_total_reads "$MIN_READS" \
    --suggest_test_frac "$TEST_FRAC" \
    --out_prefix "$DIAG_DIR/$RUN_NAME" \
    2>&1 | tee "$LOG"

echo
echo "extract_cells done. Review $LOG for the suggested fold_assignment, then make sure the"
echo "partition.fold_assignment in $BASE_CONFIG matches the chromosomes you chose."
echo

# ---- optional gate: pause so you can set fold_assignment before generating ----
# Comment out this block to run straight through (e.g. when fold_assignment is already set).
if [[ "${CONFIRM:-1}" == "1" ]]; then
    read -r -p "fold_assignment set in $BASE_CONFIG? Proceed to generate inputs? [y/N] " ans
    [[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "Stopped before generate_all_inputs."; exit 0; }
fi

echo "=== [2/2] generate_all_inputs (build training CSVs for all cells) ==="
echo "  cells_csv=$CELLS_CSV  output_dir=$INPUT_DIR"
python -m entexbert2.scripts.generate_all_inputs \
    "$BASE_CONFIG" \
    --cells_csv "$CELLS_CSV" \
    --output_dir "$INPUT_DIR" \
    --skip_existing

echo
echo "Done. Inputs under $INPUT_DIR ; batch manifest: $INPUT_DIR/generate_all_inputs_manifest.json"
echo "Diagnostics + log committed under $DIAG_DIR"
