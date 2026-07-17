#!/usr/bin/env bash
#SBATCH --job-name=eb2_ralt_clf
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

# ============================================================
# ref_alt_pair BINARY ASB-SIGNIFICANCE CLASSIFICATION — ENC-002 CTCF (hurdle STAGE 1)
# Predicts imbalance_significance (1=ASB, 0=not) on paired ref/alt windows.
# Contrast is symmetric_abs: significance is DIRECTION-AGNOSTIC (ASB whether ALT or REF binds more).
# Reuses the exact twin/ref_alt_pair plumbing verified for the regression run. No depth/hetero here.
# ============================================================

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/ctcf_ref_alt_pair_classification.yaml}"
DONOR="${DONOR:-ENC-002}"
DATASET="${DATASET:-${DONOR}__TF-ChIP-seq_CTCF}"
EXP_DIR="${EXP_DIR:-experiments/refalt_${DONOR}_CTCF_clf}"
DATA_DIR="$EXP_DIR/inputs/$DATASET/fold0"
RUNS_DIR="$EXP_DIR/runs"
OUT="$RUNS_DIR/clf"
MODEL="${MODEL:-$HOME/entexBERT-2/DNABERT-2-117M-attention}"
CONDA_ENV="${CONDA_ENV:-eb2}"
mkdir -p "$EXP_DIR" logs "$OUT"

module load miniconda 2>/dev/null || true
# shellcheck disable=SC1091
source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

# ============================================================
# STEP 1 — build PAIRED inputs (sequence1=ref, sequence2=alt); label = imbalance_significance
# ============================================================
if [[ -f "$DATA_DIR/train.csv" ]]; then
    echo "[build] $DATA_DIR/train.csv exists — skipping build (delete to force rebuild)"
else
    echo "[build] generating ref_alt_pair CLASSIFICATION inputs for $DATASET into $EXP_DIR/inputs"
    python -m entexbert2.scripts.generate_all_inputs \
        "$CONFIG" \
        --output_dir "$EXP_DIR/inputs"
fi

# sanity: paired build MUST produce sequence1 + sequence2 + label (no depth needed for classification)
HEADER="$(head -1 "$DATA_DIR/train.csv")"
for col in sequence1 sequence2 label; do
    grep -q "$col" <<< "$HEADER" || { echo "FATAL: column '$col' missing from train.csv header (check input_mode)"; exit 1; }
done
echo "[build] paired columns present (sequence1, sequence2, label)"

# STEP 1b — report the POSITIVE COUNT (decides whether hurdle STAGE 2 regression is trainable).
echo "=== [build] label balance (positives = ASB-significant loci) ==="
for split in train dev test; do
    f="$DATA_DIR/${split}.csv"
    [[ -f "$f" ]] || continue
    python - "$f" "$split" <<'PY'
import sys, pandas as pd
f, split = sys.argv[1], sys.argv[2]
df = pd.read_csv(f)
n = len(df); pos = int((df["label"] == 1).sum())
print(f"  {split:5s} n={n:7d}  positives={pos:6d}  frac={pos/max(n,1):.4f}")
PY
done

# ============================================================
# STEP 2 — finetune (binary significance classification on the symmetric ref/alt contrast)
# ============================================================
echo "=== [clf] $DATASET | imbalance_significance | contrast=symmetric_abs | class_weights=balanced ==="
python -m entexbert2.finetune_entexbert2 \
    --model_name_or_path "$MODEL" \
    --data_path "$DATA_DIR" \
    --input_mode ref_alt_pair \
    --task classification \
    --main_num_labels 2 \
    --contrast_mode symmetric_abs \
    --class_weights balanced \
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
    --metric_for_best_model auprc \
    --greater_is_better True \
    --bf16 True \
    --run_name "${DATASET}__clf" \
    --output_dir "$OUT" \
    --save_model True \
    --eval_and_save_results True \
    2>&1 | tee "$OUT/train.log"

echo "=== training DONE. dev AUPRC/AUROC in $OUT/train.log ==="
echo "    baseline AUPRC floor ~ positive fraction (see STEP 1b); AUPRC >> that = real signal."
