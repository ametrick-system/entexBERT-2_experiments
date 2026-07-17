#!/usr/bin/env bash
#SBATCH --job-name=eb2_ralt_reg
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

# ============================================================
# ref_alt_pair SIGNED-EFFECT REGRESSION — ENC-002 CTCF (Wave B, single arm)
# Predicts signed_log_alt_ref on paired ref/alt windows.
# ============================================================

# --- repo root: SLURM_SUBMIT_DIR under batch (BASH_SOURCE points at the spooled copy
#     in a non-writable system dir under sbatch); fall back to BASH_SOURCE for direct bash runs.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_ROOT"

# --- config / paths (override via env if needed) ---
CONFIG="${CONFIG:-configs/ctcf_ref_alt_pair_regression_depth100.yaml}"
DONOR="${DONOR:-ENC-002}"
DATASET="${DATASET:-${DONOR}__TF-ChIP-seq_CTCF}"
EXP_DIR="${EXP_DIR:-experiments/refalt_${DONOR}_CTCF_reg_100}"
DATA_DIR="$EXP_DIR/inputs/$DATASET/fold0"
RUNS_DIR="$EXP_DIR/runs"
OUT="$RUNS_DIR/ref_alt_reg"
MODEL="${MODEL:-$HOME/entexBERT-2/DNABERT-2-117M-attention}"
CONDA_ENV="${CONDA_ENV:-eb2}"
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
for col in sequence1 sequence2 label; do
    grep -q "$col" <<< "$HEADER" || { echo "FATAL: column '$col' missing from train.csv header — build did not run in ref_alt_pair mode"; exit 1; }
done
grep -q "sequence," <<< "$HEADER,"  && { echo "WARNING: a bare 'sequence' column is present — check input_mode"; }
echo "[build] paired columns present (sequence1, sequence2, label)"

# ============================================================
# STEP 2 — finetune (signed-effect regression on the ref/alt contrast)
# ============================================================
echo "=== [ref_alt_reg] $DATASET | signed_log_alt_ref | contrast=signed ==="
python -m entexbert2.finetune_entexbert2 \
    --model_name_or_path "$MODEL" \
    --data_path "$DATA_DIR" \
    --input_mode ref_alt_pair \
    --task regression \
    --main_num_labels 1 \
    --contrast_mode signed \
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
    --eval_steps 200 \
    --save_steps 200 \
    --evaluation_strategy steps \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --metric_for_best_model spearman \
    --greater_is_better True \
    --bf16 True \
    --run_name "${DATASET}__ref_alt_reg" \
    --output_dir "$OUT" \
    --save_model True \
    --eval_and_save_results True \
    2>&1 | tee "$OUT/train.log"

echo "=== DONE. dev Spearman/MSE in $OUT/train.log ==="
