#!/usr/bin/env bash
#SBATCH --job-name=eb2_rsingle_clf_binsplit2_cw_enc001
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

# ============================================================
# ref_SINGLE ASB classification — bin-split, CLASS-WEIGHTED (no downsampling).
# Full-data counterpart to run_ref_single_classification_binsplit2.sbatch: keep ALL negatives,
# reweight the loss via --class_weights balanced. Same 100kb bin-split (salt=entexbert2_v1), so the
# split is identical; the ONLY change vs the downsample run is balance:none + class_weights balanced.
# ============================================================
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/ctcf_ref_single_classification_binsplit2_cw_enc001.yaml}"
TAG="$(basename "$CONFIG" .yaml)"
DONOR="${DONOR:-ENC-001}"
DATASET="${DATASET:-${DONOR}__TF-ChIP-seq_CTCF}"
EXP_DIR="${EXP_DIR:-experiments/refsingle_${DONOR}_CTCF_${TAG}}"
DATA_DIR="$EXP_DIR/inputs/$DATASET/fold0"
RUNS_DIR="$EXP_DIR/runs"
OUT="$RUNS_DIR/clf"
MODEL="${MODEL:-$HOME/entexBERT-2/DNABERT-2-117M-attention}"
CONDA_ENV="${CONDA_ENV:-eb2}"
mkdir -p "$EXP_DIR" logs "$OUT"

module load miniconda 2>/dev/null || true
# shellcheck disable=SC1091
source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

# STEP 1 — build SINGLE-WINDOW inputs, NO downsampling (balance:none).
if [[ -f "$DATA_DIR/train.csv" ]]; then
    echo "[build] $DATA_DIR/train.csv exists — skipping build (delete to force rebuild)"
else
    echo "[build] generating CLASS-WEIGHTED (no-downsample) ref_single inputs ($TAG) for $DATASET"
    python -m entexbert2.scripts.generate_all_inputs "$CONFIG" --output_dir "$EXP_DIR/inputs"
fi

HEADER="$(head -1 "$DATA_DIR/train.csv")"
grep -q "sequence" <<< "$HEADER" || { echo "FATAL: 'sequence' column missing from train.csv (check input_mode)"; exit 1; }
if grep -q "sequence1" <<< "$HEADER"; then
    echo "FATAL: found 'sequence1' — this is a PAIRED build; ref_single config expected"; exit 1
fi
echo "[build] single-window column present (sequence, label)"

# STEP 1b — with balance:none, ALL splits stay at NATURAL ~3-4% (train NOT 50/50).
echo "=== [build] label balance (ALL splits NATURAL ~3-4% — train is NOT balanced here) ==="
for split in train dev test; do
    f="$DATA_DIR/${split}.csv"; [[ -f "$f" ]] || continue
    python - "$f" "$split" <<'PY'
import sys, pandas as pd
f, split = sys.argv[1], sys.argv[2]
df = pd.read_csv(f); n=len(df); pos=int((df["label"]==1).sum())
print(f"  {split:5s} n={n:7d}  positives={pos:6d}  frac={pos/max(n,1):.4f}")
PY
done
echo "    ^ train frac should be ~0.03-0.04 (NOT 0.5) — if it's 0.5 the config still has balancing; STOP."

# STEP 2 — finetune with CLASS-WEIGHTED loss (num_labels 2 -> weighted cross-entropy).
echo "=== [clf-binsplit2-CW $TAG] $DATASET | imbalance_significance | ref_single | class_weights=BALANCED ==="
python -m entexbert2.finetune_entexbert2 \
    --model_name_or_path "$MODEL" \
    --data_path "$DATA_DIR" \
    --input_mode ref_single \
    --task classification \
    --main_num_labels 2 \
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
    --run_name "${DATASET}__${TAG}" \
    --output_dir "$OUT" \
    --save_model True \
    --eval_and_save_results True \
    2>&1 | tee "$OUT/train.log"

echo "=== training DONE. dev AUROC/AUPRC in $OUT/train.log ==="
echo "    COMPARE to the DOWNSAMPLE pair run (AUROC 0.672 / AUPRC 0.130):"
echo "      class-weight AUPRC > 0.130 => keeping all negatives + reweighting helped"
echo "      class-weight AUPRC ~ 0.130 => downsample-balance was already sufficient"
