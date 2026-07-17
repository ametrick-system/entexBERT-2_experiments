#!/usr/bin/env bash
#SBATCH --job-name=eb2_ralt_hetero
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

# ============================================================
# ref_alt_pair SIGNED-EFFECT REGRESSION + DEPTH-SUPERVISED loss — ENC-002 CTCF (Wave B)
# Predicts signed_log_alt_ref on paired ref/alt windows; down-weights low-depth (noisy) loci.
# REQUIRES the heteroscedastic-head diff applied to the library (see heteroscedastic_head_diff_plan.md)
# and the hetero config (depth_col: total_reads). RUN test_heteroscedastic_head.py FIRST.
# ============================================================

# --- config / paths (override via env if needed) ---
CONFIG="${CONFIG:-configs/ctcf_ref_alt_pair_regression_hetero.yaml}"
DONOR="${DONOR:-ENC-002}"
DATASET="${DATASET:-${DONOR}__TF-ChIP-seq_CTCF}"
EXP_DIR="${EXP_DIR:-experiments/refalt_${DONOR}_CTCF_hetero}"
DATA_DIR="$EXP_DIR/inputs/$DATASET/fold0"
RUNS_DIR="$EXP_DIR/runs"
OUT="$RUNS_DIR/hetero"
MODEL="${MODEL:-$HOME/entexBERT-2/DNABERT-2-117M-attention}"
CONDA_ENV="${CONDA_ENV:-eb2}"
HETERO_LOSS="${HETERO_LOSS:-weighted_mse}"   # weighted_mse (simpler GLS) | nll

REPO_ROOT="/home/asm242/palmer_scratch/entexBERT-2_experiments/EN-TEx"

module load miniconda 2>/dev/null || true
# shellcheck disable=SC1091
source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

# ============================================================
# STEP 3 — stratified-by-depth evaluation (honest read: real signal vs noise-suppression)
# ============================================================
echo "=== [eval] dev Spearman within depth terciles ==="
python "$REPO_ROOT/eval_stratified_by_depth.py" \
    --checkpoint "$OUT" \
    --data_dir   "$DATA_DIR" \
    --split dev --bins 3 --device cuda 2>&1 | tee "$OUT/stratified_eval.log" || \
    echo "[eval] stratified eval skipped (place eval_stratified_by_depth.py at repo root)"
