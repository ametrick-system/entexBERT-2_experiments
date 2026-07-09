#!/usr/bin/env bash
#SBATCH --job-name=eb2_hpsearch
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu                   # <-- CONFIRM your GPU partition
#SBATCH --account=gerstein                # <-- CONFIRM your Slurm account
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00                   # <-- per-task cap; tune runs are short (2-4 epochs)
#SBATCH --array=0-0                       # <-- set to 0-(N-1); N printed by make_hp_task_manifest.py
#
# run_hp_search.sbatch — SLURM array for the ref_single BASELINE hyperparameter search.
# One array task = one (config x dataset x seed) baseline fine-tune. Selection is on the DEV split;
# --eval_and_save_results False means TEST is never evaluated during tuning (held-out chroms stay
# untouched). No final model is dumped (save_model False) — only the Trainer's own checkpoint
# (with trainer_state.json -> best dev auprc) is kept, so disk stays small.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root (sbatch lives in hp_tune/)
MANIFEST="${MANIFEST:-$REPO_ROOT/hp_tune/tasks_hp.tsv}"
MODEL="${MODEL:-$HOME/entexBERT-2/DNABERT-2-117M-attention}"
OUT_BASE="${OUT_BASE:-$REPO_ROOT/hp_tune/runs}"
CONDA_ENV="${CONDA_ENV:-eb2}"

mkdir -p logs "$OUT_BASE"
module load miniconda 2>/dev/null || true
# shellcheck disable=SC1091
source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

# manifest is header + rows; array id 0 -> first data row = line 2
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")
[[ -n "$LINE" ]] || { echo "No manifest line for array id $SLURM_ARRAY_TASK_ID in $MANIFEST"; exit 1; }
IFS=$'\t' read -r RUN_TAG CONFIG_ID DATA_DIR SEED LR BATCH EPOCHS WD WARMUP HEAD_LAYERS <<< "$LINE"
echo "[hp $SLURM_ARRAY_TASK_ID] $RUN_TAG | lr=$LR batch=$BATCH ep=$EPOCHS wd=$WD warmup=$WARMUP head=$HEAD_LAYERS seed=$SEED"

OUT_DIR="$OUT_BASE/$RUN_TAG"
mkdir -p "$OUT_DIR"

# Baseline only (num_aux_tasks=0). ref_single binary classification, center_mean pooling
# (Feng 2025, doi:10.1038/s41467-025-65823-8), class_weights=balanced. Select best on DEV auprc.
srun python -m entexbert2.finetune_entexbert2 \
    --model_name_or_path "$MODEL" \
    --data_path "$DATA_DIR" \
    --run_name "$RUN_TAG" \
    --output_dir "$OUT_DIR" \
    --seed "$SEED" \
    --task classification \
    --main_num_labels 2 \
    --pooling_mode center_mean \
    --center_pool_width 5 \
    --head_num_layers "$HEAD_LAYERS" \
    --class_weights balanced \
    --model_max_length 512 \
    --learning_rate "$LR" \
    --per_device_train_batch_size "$BATCH" \
    --per_device_eval_batch_size 32 \
    --num_train_epochs "$EPOCHS" \
    --weight_decay "$WD" \
    --warmup_steps "$WARMUP" \
    --logging_steps 50 \
    --eval_steps 200 \
    --save_steps 200 \
    --evaluation_strategy steps \
    --save_total_limit 1 \
    --load_best_model_at_end True \
    --metric_for_best_model auprc \
    --greater_is_better True \
    --fp16 True \
    --save_model False \
    --eval_and_save_results False

echo "[hp $SLURM_ARRAY_TASK_ID] done -> $OUT_DIR"
