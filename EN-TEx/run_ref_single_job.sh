#!/usr/bin/env bash
# run_ref_single_job.sh — ONE ref_single fine-tune (baseline or lupi arm). Called per dSQ line:
#   bash run_ref_single_job.sh <run_tag> <data_dir> <aux_arm>
# aux_arm = baseline (num_aux_tasks=0) | lupi (privileged aux heads on).
# Holds every FIXED flag + the LUPI aux config; 3 positional args vary per job.
# Env-overridable: MODEL, OUT_BASE (default <this dir>/runs), CONDA_ENV.
set -euo pipefail

RUN_TAG="${1:?run_tag}"; DATA_DIR="${2:?data_dir}"; AUX_ARM="${3:?aux_arm}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-$HOME/entexBERT-2/DNABERT-2-117M-attention}"
OUT_BASE="${OUT_BASE:-$SCRIPT_DIR/runs}"
CONDA_ENV="${CONDA_ENV:-eb2}"

# LUPI aux config — privileged per-locus regression targets (Wave-B plan). Apply the
# HP-search winner's lr/batch/epochs/wd/warmup/head_num_layers here (same for both arms).
AUX_NAMES="log_total_count,neg_log10_p_betabinom,abs_log_count_ratio"
AUX_TYPES="regression,regression,regression"
AUX_NUMLABELS="1,1,1"
AUX_LAMBDAS="0.3,0.3,0.3"

module load miniconda 2>/dev/null || true
# shellcheck disable=SC1091
source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

OUT_DIR="$OUT_BASE/$RUN_TAG"
mkdir -p "$OUT_DIR"
echo "[ref_single] $RUN_TAG arm=$AUX_ARM data_dir=$DATA_DIR"

AUX_FLAGS=()
if [[ "$AUX_ARM" == "lupi" ]]; then
    NAUX=$(awk -F, '{print NF}' <<< "$AUX_NAMES")
    AUX_FLAGS=(
        --num_aux_tasks "$NAUX"
        --aux_task_names "$AUX_NAMES"
        --aux_task_types "$AUX_TYPES"
        --aux_num_labels "$AUX_NUMLABELS"
        --lambda_aux "$AUX_LAMBDAS"
    )
elif [[ "$AUX_ARM" != "baseline" ]]; then
    echo "Unknown aux_arm '$AUX_ARM' (expected baseline|lupi)"; exit 1
fi

python -m entexbert2.finetune_entexbert2 \
    --model_name_or_path "$MODEL" \
    --data_path "$DATA_DIR" \
    --run_name "$RUN_TAG" \
    --output_dir "$OUT_DIR" \
    --task classification \
    --main_num_labels 2 \
    --pooling_mode center_mean \
    --center_pool_width 5 \
    --head_num_layers 1 \
    --class_weights balanced \
    --model_max_length 512 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --learning_rate 3e-5 \
    --warmup_steps 50 \
    --weight_decay 0.01 \
    --logging_steps 50 \
    --eval_steps 200 \
    --save_steps 200 \
    --evaluation_strategy steps \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --fp16 True \
    --save_model True \
    --eval_and_save_results True \
    ${AUX_FLAGS[@]+"${AUX_FLAGS[@]}"} \
    2>&1 | tee "$OUT_DIR/train.log"

echo "[ref_single] done -> $OUT_DIR"
