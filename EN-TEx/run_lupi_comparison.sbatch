#!/usr/bin/env bash
#SBATCH --job-name=eb2_lupi_cmp
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G                         # tokenizer materializes the WHOLE train set in host RAM up front
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
#
# run_lupi_comparison.sbatch — batch version of the baseline-vs-LUPI comparison (default ENC-002 CTCF).
# Two sequential fine-tunes (baseline then lupi) on one GPU, same data, identical HPs. Submit from
# entexBERT-2_experiments/EN-TEx:
#   sbatch run_lupi_comparison.sbatch
#   DATASET=ENC-003__TF-ChIP-seq_CTCF sbatch run_lupi_comparison.sbatch
# NOTE on memory: the finetune tokenizes the entire train.csv into one in-RAM tensor before training
# (~489k rows for ENC-002 => several GB transient). --mem 64G covers it; an interactive code-server
# session with a small RAM cap gets OOM-killed at "Perform single-sequence prediction...".

set -euo pipefail

# ---- what to run ----
DATASET="${DATASET:-ENC-002__TF-ChIP-seq_CTCF}"
DONOR="${DATASET%%__*}"                            # ENC-002
ASSAY="${DATASET#*__}"                             # TF-ChIP-seq_CTCF
CONFIG="${CONFIG:-configs/ctcf_ref_single_lupi.yaml}"

# ---- fixed HPs for BOTH arms (edit here to pin the comparison's hyperparameters) ----
LR="${LR:-3e-5}"; BATCH="${BATCH:-16}"; EPOCHS="${EPOCHS:-3}"
WD="${WD:-0.01}"; WARMUP="${WARMUP:-50}"; HEAD_LAYERS="${HEAD_LAYERS:-1}"

# ---- LUPI aux config (must match the aux_labels in the config, in order) ----
# NOTE: finetune_entexbert2.py parses these four as LIST args (HfArgumentParser nargs="+"),
# so each value must be a SEPARATE space-separated token — NOT one comma-joined string.
# Passing "1,1,1" makes argparse do int("1,1,1") -> error. Hence bash arrays, expanded unquoted.
AUX_NAMES=(log_total_count neg_log10_p_betabinom abs_log_count_ratio)
AUX_TYPES=(regression regression regression)
AUX_NUMLABELS=(1 1 1)
# lambda override: accept a comma- or space-separated AUX_LAMBDAS env var, else default 0.3 x3
IFS=', ' read -r -a AUX_LAMBDAS <<< "${AUX_LAMBDAS:-0.3 0.3 0.3}"

# ---- paths ----
# Under sbatch, ${BASH_SOURCE[0]} points at SLURM's spooled copy of this script (a non-writable
# system dir), so cd-ing there makes every mkdir fail with "Permission denied". SLURM sets
# SLURM_SUBMIT_DIR to the dir you ran sbatch from (and cd's there by default) — use it. Fall
# back to BASH_SOURCE so the file still works if run directly with bash.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_ROOT"
EXP_DIR="${EXP_DIR:-experiments/lupi_${DONOR}_CTCF}"    # clearly-LUPI experiment folder
DATA_DIR="$EXP_DIR/inputs/$DATASET/fold0"
RUNS_DIR="$EXP_DIR/runs"
MODEL="${MODEL:-$HOME/entexBERT-2/DNABERT-2-117M-attention}"
CONDA_ENV="${CONDA_ENV:-eb2}"
mkdir -p "$EXP_DIR" logs "$RUNS_DIR"

module load miniconda 2>/dev/null || true
# shellcheck disable=SC1091
source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

# ============================================================
# STEP 1 — build inputs ONCE (aux targets written into the CSVs)
# ============================================================
if [[ -f "$DATA_DIR/train.csv" ]]; then
    echo "[build] $DATA_DIR/train.csv exists — skipping build (delete to force rebuild)"
else
    echo "[build] generating inputs for $DATASET into $EXP_DIR/inputs"
    python -m entexbert2.scripts.generate_all_inputs \
        "$CONFIG" \
        --output_dir "$EXP_DIR/inputs"
fi
# sanity: the three aux columns must be present, or the LUPI arm will KeyError
HEADER="$(head -1 "$DATA_DIR/train.csv")"
for col in log_total_count neg_log10_p_betabinom abs_log_count_ratio; do
    grep -q "$col" <<< "$HEADER" || { echo "FATAL: aux column '$col' missing from train.csv header"; exit 1; }
done
echo "[build] aux columns present in train.csv"

# ============================================================
# shared finetune flags (identical across arms)
# ============================================================
common_flags=(
    --model_name_or_path "$MODEL"
    --data_path "$DATA_DIR"
    --task classification
    --main_num_labels 2
    --pooling_mode center_mean
    --center_pool_width 5
    --head_num_layers "$HEAD_LAYERS"
    --class_weights balanced
    --model_max_length 512
    --per_device_train_batch_size "$BATCH"
    --per_device_eval_batch_size 32
    --num_train_epochs "$EPOCHS"
    --learning_rate "$LR"
    --weight_decay "$WD"
    --warmup_steps "$WARMUP"
    --logging_steps 50
    --eval_steps 200
    --save_steps 200
    --evaluation_strategy steps
    --save_total_limit 2
    --load_best_model_at_end True
    --metric_for_best_model auprc
    --greater_is_better True
    --fp16 True
    --save_model True
    --eval_and_save_results True
)

run_arm () {   # $1 = arm name, $2.. = extra flags
    local arm="$1"; shift
    local out="$RUNS_DIR/$arm"
    mkdir -p "$out"
    echo "=== [$arm] $DATASET | lr=$LR batch=$BATCH ep=$EPOCHS wd=$WD warmup=$WARMUP head=$HEAD_LAYERS ==="
    python -m entexbert2.finetune_entexbert2 \
        "${common_flags[@]}" \
        --run_name "${DATASET}__${arm}" \
        --output_dir "$out" \
        "$@" \
        2>&1 | tee "$out/train.log"
    echo "=== [$arm] done -> $out ==="
}

# ============================================================
# STEP 2 — lupi arm (privileged aux heads) — RUN FIRST
# ============================================================
run_arm lupi \
    --num_aux_tasks 3 \
    --aux_task_names  "${AUX_NAMES[@]}" \
    --aux_task_types  "${AUX_TYPES[@]}" \
    --aux_num_labels  "${AUX_NUMLABELS[@]}" \
    --lambda_aux      "${AUX_LAMBDAS[@]}"

# ============================================================
# STEP 3 — baseline arm (no aux)
# ============================================================
run_arm baseline --num_aux_tasks 0

echo "ALL DONE. Compare dev AUPRC:"
echo "  baseline: $RUNS_DIR/baseline/train.log"
echo "  lupi:     $RUNS_DIR/lupi/train.log"
