#!/usr/bin/env bash
#SBATCH --job-name=eb2_ralt_clf_baltrain
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

# ============================================================
# ref_alt_pair ASB-significance classification — TRAIN-BALANCED variant (ENC-002 CTCF)
# A/B vs the class_weights run: this DOWNSAMPLES negatives in the TRAIN split only
# (balance.apply_to: train) and uses --class_weights none, so imbalance is handled by DATA,
# not by loss weighting. dev/test stay natural-prevalence + locus-grouped => AUROC is directly
# comparable to the class_weights run (~0.61). Settles the old 'balanced ~0.75-0.8' question.
# ============================================================

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/ctcf_ref_alt_pair_classification_baltrain.yaml}"
DONOR="${DONOR:-ENC-002}"
DATASET="${DATASET:-${DONOR}__TF-ChIP-seq_CTCF}"
EXP_DIR="${EXP_DIR:-experiments/refalt_${DONOR}_CTCF_clf_baltrain}"
DATA_DIR="$EXP_DIR/inputs/$DATASET/fold0"
RUNS_DIR="$EXP_DIR/runs"
OUT="$RUNS_DIR/clf_baltrain"
MODEL="${MODEL:-$HOME/entexBERT-2/DNABERT-2-117M-attention}"
CONDA_ENV="${CONDA_ENV:-eb2}"
mkdir -p "$EXP_DIR" logs "$OUT"

module load miniconda 2>/dev/null || true
# shellcheck disable=SC1091
source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

# ============================================================
# STEP 1 — build PAIRED inputs; TRAIN split downsampled to balanced, dev/test natural-prevalence
# ============================================================
if [[ -f "$DATA_DIR/train.csv" ]]; then
    echo "[build] $DATA_DIR/train.csv exists — skipping build (delete to force rebuild)"
else
    echo "[build] generating TRAIN-BALANCED ref_alt_pair classification inputs for $DATASET"
    python -m entexbert2.scripts.generate_all_inputs \
        "$CONFIG" \
        --output_dir "$EXP_DIR/inputs"
fi

# sanity: paired build MUST produce sequence1 + sequence2 + label
HEADER="$(head -1 "$DATA_DIR/train.csv")"
for col in sequence1 sequence2 label; do
    grep -q "$col" <<< "$HEADER" || { echo "FATAL: column '$col' missing from train.csv header (check input_mode)"; exit 1; }
done
echo "[build] paired columns present (sequence1, sequence2, label)"

# STEP 1b — CONFIRM the balance landed where intended: train ~50/50, dev/test still ~3-4% positive.
echo "=== [build] label balance (train should be ~50/50; dev/test NATURAL ~3-4%) ==="
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
echo "    ^ if dev/test frac is ~0.5 the balance LEAKED into eval (apply_to should be 'train'); STOP and check."

# ============================================================
# STEP 2 — finetune: classification, symmetric_abs, NO class weights (data is already balanced)
# ============================================================
echo "=== [clf-baltrain] $DATASET | imbalance_significance | contrast=symmetric_abs | class_weights=NONE (data balanced) ==="
python -m entexbert2.finetune_entexbert2 \
    --model_name_or_path "$MODEL" \
    --data_path "$DATA_DIR" \
    --input_mode ref_alt_pair \
    --task classification \
    --main_num_labels 2 \
    --contrast_mode symmetric_abs \
    --class_weights none \
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
    --run_name "${DATASET}__clf_baltrain" \
    --output_dir "$OUT" \
    --save_model True \
    --eval_and_save_results True \
    2>&1 | tee "$OUT/train.log"

echo "=== training DONE. dev AUPRC/AUROC in $OUT/train.log ==="
echo "    COMPARE dev AUROC to the class_weights run (~0.61):"
echo "      ~0.75-0.8  => train-balancing is a REAL win (adopt hurdle-wide)"
echo "      ~0.61      => the old high number was an eval artifact (leaky/easy eval)"
