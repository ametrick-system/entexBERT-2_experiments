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

# --- repo root: SLURM_SUBMIT_DIR under batch (BASH_SOURCE points at the spooled copy
#     in a non-writable system dir under sbatch); fall back to BASH_SOURCE for direct bash runs.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_ROOT"

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
mkdir -p "$EXP_DIR" logs "$OUT"

module load miniconda 2>/dev/null || true
# shellcheck disable=SC1091
source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

# ============================================================
# STEP 1 — build PAIRED inputs (sequence1=ref, sequence2=alt written into CSVs)
# ============================================================
if [[ -f "$DATA_DIR/train.csv" ]]; then
    echo "[build] $DATA_DIR/train.csv exists — skipping build (delete to force rebuild)"
else
    echo "[build] generating ref_alt_pair inputs for $DATASET into $EXP_DIR/inputs"
    python -m entexbert2.scripts.generate_all_inputs \
        "$CONFIG" \
        --output_dir "$EXP_DIR/inputs"
fi

# sanity: paired build MUST produce sequence1 + sequence2 (else it silently trained single-seq)
HEADER="$(head -1 "$DATA_DIR/train.csv")"
for col in sequence1 sequence2 label depth; do
    grep -q "$col" <<< "$HEADER" || { echo "FATAL: column '$col' missing from train.csv header (need paired seqs + depth; check input_mode + depth_col)"; exit 1; }
done
grep -q "sequence," <<< "$HEADER,"  && { echo "WARNING: a bare 'sequence' column is present — check input_mode"; }
echo "[build] paired columns present (sequence1, sequence2, label, depth)"

# ============================================================
# STEP 2 — finetune (signed-effect regression on the ref/alt contrast)
# ============================================================
echo "=== [hetero] $DATASET | signed_log_alt_ref | contrast=signed | loss=$HETERO_LOSS ==="
python -m entexbert2.finetune_entexbert2 \
    --model_name_or_path "$MODEL" \
    --data_path "$DATA_DIR" \
    --input_mode ref_alt_pair \
    --task regression \
    --main_num_labels 1 \
    --contrast_mode signed \
    --heteroscedastic True \
    --hetero_loss "$HETERO_LOSS" \
    --pooling_mode center_mean \
    --center_pool_width 5 \
    --head_num_layers 2 \
    --model_max_length 512 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 3 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --warmup_steps 50 \
    --logging_steps 50 \
    --eval_steps 100 \
    --save_steps 100 \
    --evaluation_strategy steps \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --metric_for_best_model spearman \
    --greater_is_better True \
    --bf16 True \
    --run_name "${DATASET}__hetero_${HETERO_LOSS}" \
    --output_dir "$OUT" \
    --save_model True \
    --eval_and_save_results True \
    2>&1 | tee "$OUT/train.log"

echo "=== training DONE. dev Spearman/MSE in $OUT/train.log ==="

# ============================================================
# STEP 3 — stratified-by-depth evaluation (honest read: real signal vs noise-suppression)
# ============================================================
echo "=== [eval] dev Spearman within depth terciles ==="
python "$REPO_ROOT/eval_stratified_by_depth.py" \
    --checkpoint "$OUT" \
    --data_dir   "$DATA_DIR" \
    --split dev --bins 3 2>&1 | tee "$OUT/stratified_eval.log" || \
    echo "[eval] stratified eval skipped (place eval_stratified_by_depth.py at repo root)"
