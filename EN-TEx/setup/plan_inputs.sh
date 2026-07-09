#!/usr/bin/env bash
# plan_inputs.sh — STEP 1 of 2: input-planning pass over the raw hetSNVs TSV.
# Produces everything you need to WRITE the base config, then STOPS. No config required.
#
# Outputs (all under this repo, version-controlled):
#   input_plan/<run>/plan_inputs.log   full report (tee'd)
#   input_plan/<run>/<run>_datasets.csv     per-(donor,assay) dataset table -> feed to build_inputs.sh --datasets_csv
#   input_plan/<run>/<run>_per_chrom.csv per-chromosome positive counts
# The log prints a suggested fold_assignment (targeting ~TEST_FRAC of positives) to paste into
# the config's partition: block. AFTER that, run build_inputs.sh.
#
# Usage:   ./plan_inputs.sh <run_name>
# e.g.     ./plan_inputs.sh ctcf_ref_single_2026-07
set -euo pipefail

RUN_NAME="${1:?usage: plan_inputs.sh <run_name>}"

# ---- EDIT THESE for your dataset (env-overridable) ----
TSV="${TSV:-/home/asm242/entex_data/hetSNVs_default_AS.tsv}"
MIN_READS="${MIN_READS:-10}"      # must match the config's row_source.min_total_reads
TEST_FRAC="${TEST_FRAC:-0.10}"    # target fraction of positives to hold out (suggestion only)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAN_DIR="$REPO_ROOT/input_plan/$RUN_NAME"
mkdir -p "$PLAN_DIR"
LOG="$PLAN_DIR/plan_inputs.log"

echo "=== plan_inputs (input-planning pass) ==="
echo "  tsv=$TSV  min_total_reads=$MIN_READS  suggest_test_frac=$TEST_FRAC"
echo "  log -> $LOG"

python "$REPO_ROOT/plan_inputs.py" \
    --tsv "$TSV" \
    --min_total_reads "$MIN_READS" \
    --suggest_test_frac "$TEST_FRAC" \
    --out_prefix "$PLAN_DIR/$RUN_NAME" \
    2>&1 | tee "$LOG"

echo
echo "NEXT: paste the suggested fold_assignment above into your base config's partition: block"
echo "      (the SAME dict for every dataset), then run:"
echo "      ./build_inputs.sh <base_config.yaml> $RUN_NAME"
