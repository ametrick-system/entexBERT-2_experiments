#!/usr/bin/env bash
#SBATCH --job-name=eb2_refsingle
#SBATCH --output=logs/%x_%A_%a.out       # %x=jobname %A=arrayjobid %a=arraytaskid
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu                   # <-- CONFIRM: your GPU partition on McCleary
#SBATCH --account=gerstein                # <-- CONFIRM: your Slurm account/allocation
#SBATCH --gpus=1                          # 1 GPU per DNABERT-2 fine-tune
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00                   # <-- tune to your dataset size
#SBATCH --array=0-0                       # <-- set to 0-(N-1); N = number of lines in the task manifest
#
# run_ref_single.sbatch — SLURM array driver for the entexBERT-2 ref_single experiments.
# Lives in entexBERT-2_experiments. One array task = one (cell x LUPI-arm) fine-tune.
#
# It reads a TASK MANIFEST (one job per line) so the array is fully data-driven and reproducible.
# Build the manifest first (see make_task_manifest below), set --array=0-(N-1), then:
#   sbatch run_ref_single.sbatch
#
# Each manifest line is TAB-separated:
#   <run_tag> <data_dir> <aux_arm>
# where:
#   run_tag  : unique name for this run (used for output_dir + run_name), e.g. ENC-002__CTCF__lupi
#   data_dir : the per-cell/fold input dir generate_all_inputs produced
#              (…/inputs/<run>/<donor>__<assay>/fold0)
#   aux_arm  : "baseline" (num_aux_tasks=0) or "lupi" (privileged aux heads on)

set -euo pipefail

# ---- EDIT THESE ----
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${MANIFEST:-$REPO_ROOT/tasks_ref_single.tsv}"
MODEL="${MODEL:-$HOME/entexBERT-2/DNABERT-2-117M-attention}"   # local backbone dir (has bert_layers.py)
OUT_BASE="${OUT_BASE:-$REPO_ROOT/runs}"
CONDA_ENV="${CONDA_ENV:-eb2}"

# LUPI aux config — the privileged signals for the "lupi" arm (per your Wave-B plan).
# All three are per-locus regression targets already registered in utils.py target registry.
AUX_NAMES="log_total_count,neg_log10_p_betabinom,abs_log_count_ratio"
AUX_TYPES="regression,regression,regression"
AUX_NUMLABELS="1,1,1"
AUX_LAMBDAS="0.3,0.3,0.3"

mkdir -p logs "$OUT_BASE"

# ---- activate environment ----
module load miniconda 2>/dev/null || true
# shellcheck disable=SC1091
source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

# ---- pick this task's line from the manifest ----
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
[[ -n "$LINE" ]] || { echo "No manifest line for array id $SLURM_ARRAY_TASK_ID in $MANIFEST"; exit 1; }
IFS=$'\t' read -r RUN_TAG DATA_DIR AUX_ARM <<< "$LINE"
echo "[task $SLURM_ARRAY_TASK_ID] run_tag=$RUN_TAG arm=$AUX_ARM data_dir=$DATA_DIR"

OUT_DIR="$OUT_BASE/$RUN_TAG"
mkdir -p "$OUT_DIR"

# ---- assemble the LUPI (aux) flags only for the lupi arm ----
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

# ---- fine-tune ----
# ref_single binary AS classification. pooling_mode=center_mean (Feng 2025: mean-token beats CLS),
# class_weights=balanced (natural prevalence without majority collapse). --input_mode ref_single is
# harmless if the field is added later; drop it if your finetune build predates that field.
srun python -m entexbert2.finetune_entexbert2 \
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
    "${AUX_FLAGS[@]}"

echo "[task $SLURM_ARRAY_TASK_ID] done -> $OUT_DIR"
