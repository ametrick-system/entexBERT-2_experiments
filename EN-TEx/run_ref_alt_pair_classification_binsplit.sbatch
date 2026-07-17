#!/usr/bin/env bash
#SBATCH --job-name=eb2_ralt_clf_binsplit2
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

# ============================================================
# ref_alt_pair ASB-significance classification — PURE GENOMIC-BIN SPLIT (ENC-002 CTCF)
# All chromosomes contribute to training; 100kb bins hashed 3-way (train/dev/test) so co-located
# loci stay in one split. Train balanced (global_binary, apply_to: train) => dev/test keep NATURAL
# ~3% prevalence. Compare dev AUROC to chromosome-holdout 0.61.
#
# SCHEME SELECT (env override):
#   default            -> configs/ctcf_ref_alt_pair_classification_binsplit2.yaml         (#1 plain, ~0.5% boundary leak)
#   CONFIG=..._strict  -> configs/ctcf_ref_alt_pair_classification_binsplit2_strict.yaml  (#2 boundary-excluded, ~0 leak)
# ============================================================

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/ctcf_ref_alt_pair_classification_binsplit2.yaml}"
TAG="$(basename "$CONFIG" .yaml)"           # e.g. ctcf_ref_alt_pair_classification_binsplit2[_strict]
DONOR="${DONOR:-ENC-002}"
DATASET="${DATASET:-${DONOR}__TF-ChIP-seq_CTCF}"
EXP_DIR="${EXP_DIR:-experiments/refalt_${DONOR}_CTCF_${TAG}}"
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
# STEP 1 — build PAIRED inputs. Pure-bin split (fold_assignment {}) + train-only balance.
# ============================================================
if [[ -f "$DATA_DIR/train.csv" ]]; then
    echo "[build] $DATA_DIR/train.csv exists — skipping build (delete to force rebuild)"
else
    echo "[build] generating PURE-BIN-SPLIT ref_alt_pair classification inputs ($TAG) for $DATASET"
    python -m entexbert2.scripts.generate_all_inputs \
        "$CONFIG" \
        --output_dir "$EXP_DIR/inputs"
fi

# paired build MUST produce sequence1 + sequence2 + label
HEADER="$(head -1 "$DATA_DIR/train.csv")"
for col in sequence1 sequence2 label; do
    grep -q "$col" <<< "$HEADER" || { echo "FATAL: column '$col' missing from train.csv (check input_mode)"; exit 1; }
done
echo "[build] paired columns present (sequence1, sequence2, label)"

# STEP 1b — CONFIRM the split is honest: train ~50/50 (balanced), dev/test NATURAL ~3-4%.
# With apply_to:train fixed, the build log should also show a 'train-only balance ...' line.
echo "=== [build] label balance (train ~50/50 after balancing; dev/test NATURAL ~3-4%) ==="
for split in train dev test; do
    f="$DATA_DIR/${split}.csv"; [[ -f "$f" ]] || continue
    python - "$f" "$split" <<'PY'
import sys, pandas as pd
f, split = sys.argv[1], sys.argv[2]
df = pd.read_csv(f); n=len(df); pos=int((df["label"]==1).sum())
print(f"  {split:5s} n={n:7d}  positives={pos:6d}  frac={pos/max(n,1):.4f}")
PY
done
echo "    ^ if dev/test frac is ~0.5 the balance LEAKED into eval (config key must be 'apply_to: train',"
echo "      NOT 'balance_split'); STOP and fix the config before trusting AUPRC."

# ============================================================
# STEP 2 — finetune: classification, symmetric_abs, NO class weights (train already balanced)
# ============================================================
echo "=== [clf-binsplit2 $TAG] $DATASET | imbalance_significance | contrast=symmetric_abs | class_weights=NONE ==="
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
    --run_name "${DATASET}__${TAG}" \
    --output_dir "$OUT" \
    --save_model True \
    --eval_and_save_results True \
    2>&1 | tee "$OUT/train.log"

echo "=== training DONE. dev AUROC/AUPRC in $OUT/train.log ==="
echo "    COMPARE dev AUROC to chromosome-holdout 0.61:"
echo "      >0.61  => all-chromosome coverage / dropped chr-confound helped (but check regional leakage)"
echo "      ~0.61  => 0.61 is robust across split schemes"
echo "    AUPRC is now MEANINGFUL again (dev at natural ~3% prevalence)."
