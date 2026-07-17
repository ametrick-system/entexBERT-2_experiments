#!/usr/bin/env bash
# run_hp_job.sh — ONE baseline HP-search fine-tune. Called once per dSQ job-file line:
#   bash run_hp_job.sh <run_tag> <data_dir> <seed> <lr> <batch> <epochs> <wd> <warmup> <head_layers>
# Holds every FIXED flag; the 9 positional args are the only per-job varying values.
# Env-overridable: MODEL, OUT_BASE (default <this dir>/runs), CONDA_ENV.
# Baseline only (num_aux_tasks=0). Selection on DEV auprc; TEST never touched
# (--eval_and_save_results False), no model dumped (--save_model False).
set -euo pipefail

RUN_TAG="${1:?run_tag}"; DATA_DIR="${2:?data_dir}"; SEED="${3:?seed}"
LR="${4:?lr}"; BATCH="${5:?batch}"; EPOCHS="${6:?epochs}"
WD="${7:?weight_decay}"; WARMUP="${8:?warmup}"; HEAD_LAYERS="${9:?head_num_layers}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-$HOME/entexBERT-2/DNABERT-2-117M-attention}"
OUT_BASE="${OUT_BASE:-$SCRIPT_DIR/runs}"
CONDA_ENV="${CONDA_ENV:-eb2}"

module load miniconda 2>/dev/null || true
# shellcheck disable=SC1091
source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

OUT_DIR="$OUT_BASE/$RUN_TAG"
mkdir -p "$OUT_DIR"
echo "[hp] $RUN_TAG | lr=$LR batch=$BATCH ep=$EPOCHS wd=$WD warmup=$WARMUP head=$HEAD_LAYERS seed=$SEED"

python -m entexbert2.finetune_entexbert2 \
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
    --eval_and_save_results False \
    2>&1 | tee "$OUT_DIR/train.log"

echo "[hp] done -> $OUT_DIR"
