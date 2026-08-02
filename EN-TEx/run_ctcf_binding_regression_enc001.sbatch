#!/usr/bin/env bash
#SBATCH --job-name=eb2_ctcf_binding_reg_enc001
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

# ============================================================
# CTCF BINDING-SIGNAL regression (ENC-001) — for the ASB benchmark (Han et al. 2024 Fig 3A).
# Trains entexBERT-2 to regress BigWig fold-change over peak + matched-background windows.
# The model never sees ASB labels; ref-vs-alt Delta scoring on the ADASTRA benchmark is a
# SEPARATE step after this (score_ctcf_adastra.py) that puts entexBERT-2 on the same axis
# as DeepSEA/Sei/Enformer.
# NOTE: requires the peak_bed RowSource (DIFF 1 + DIFF 2 in regression_experiment_diffs.md)
#       applied to src/entexbert2/{utils.py,run_experiment.py}.
# ============================================================

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/ctcf_binding_regression_enc001.yaml}"
TAG="$(basename "$CONFIG" .yaml)"
DONOR="${DONOR:-ENC-001}"
DATASET="${DATASET:-${DONOR}__TF-ChIP-seq_CTCF}"
EXP_DIR="${EXP_DIR:-experiments/binding_reg_${DONOR}_CTCF_${TAG}}"
DATA_DIR="$EXP_DIR/inputs/$DATASET/fold0"
RUNS_DIR="$EXP_DIR/runs"
OUT="$RUNS_DIR/reg"
MODEL="${MODEL:-$HOME/entexBERT-2/DNABERT-2-117M-attention}"
CONDA_ENV="${CONDA_ENV:-eb2}"
mkdir -p "$EXP_DIR" logs "$OUT"

module load miniconda 2>/dev/null || true
# shellcheck disable=SC1091
source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

# ============================================================
# STEP 1 — build binding-regression inputs (peak_bed source + BigWig fold-change label).
# Pure-bin split (fold_assignment {}); no balancing (continuous target).
# ============================================================
if [[ -f "$DATA_DIR/train.csv" ]]; then
    echo "[build] $DATA_DIR/train.csv exists — skipping build (delete to force rebuild)"
else
    echo "[build] generating CTCF binding-regression inputs ($TAG) for $DATASET"
    python -m entexbert2.scripts.generate_all_inputs \
        "$CONFIG" \
        --output_dir "$EXP_DIR/inputs"
fi

# ref_single build MUST produce a single 'sequence' column + a continuous label.
HEADER="$(head -1 "$DATA_DIR/train.csv")"
grep -q "sequence" <<< "$HEADER" || { echo "FATAL: 'sequence' column missing from train.csv"; exit 1; }
if grep -q "sequence1" <<< "$HEADER"; then
    echo "FATAL: found 'sequence1' — paired build; ref_single expected"; exit 1
fi
echo "[build] single-window column present (sequence, label)"

# STEP 1b — REGRESSION sanity: the label is CONTINUOUS, and peak windows should carry
# clearly higher signal than background windows (else the target is broken).
echo "=== [build] regression target summary (peak vs background signal separation) ==="
for split in train dev test; do
    f="$DATA_DIR/${split}.csv"; m="$DATA_DIR/${split}.meta.csv"; [[ -f "$f" ]] || continue
    python - "$f" "$m" "$split" <<'PY'
import sys, pandas as pd, numpy as np
f, m, split = sys.argv[1], sys.argv[2], sys.argv[3]
d = pd.read_csv(f).reset_index(drop=True)
lab = d["label"].astype(float)
line = f"  {split:5s} n={len(d):7d}  label[mean={lab.mean():.3f} sd={lab.std():.3f} min={lab.min():.3f} max={lab.max():.3f}]"
try:
    meta = pd.read_csv(m).reset_index(drop=True)
    if "feature_type" in meta.columns:
        j = meta.assign(label=lab)
        g = j.groupby("feature_type")["label"].mean()
        pk, bg = g.get("peak", float("nan")), g.get("background", float("nan"))
        line += f"  peak_mean={pk:.3f} bg_mean={bg:.3f}  sep={'OK' if pk>bg else 'BROKEN(peak<=bg)'}"
except Exception as e:
    line += f"  (meta sep check skipped: {e})"
print(line)
PY
done
echo "    ^ peak_mean MUST exceed bg_mean; if not, the BigWig path/coords are wrong — STOP."

# ============================================================
# STEP 2 — finetune: REGRESSION, single reference window, linear head (matches DNABERT-1 depth).
# ============================================================
echo "=== [binding-reg $TAG] $DATASET | ctcf_foldchange | ref_single | regression ==="
python -m entexbert2.finetune_entexbert2 \
    --model_name_or_path "$MODEL" \
    --data_path "$DATA_DIR" \
    --input_mode ref_single \
    --task regression \
    --main_num_labels 1 \
    --pooling_mode center_mean \
    --center_pool_width 5 \
    --head_num_layers 1 \
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
    --metric_for_best_model eval_spearman \
    --greater_is_better True \
    --bf16 True \
    --run_name "${DATASET}__${TAG}" \
    --output_dir "$OUT" \
    --save_model True \
    --eval_and_save_results True \
    2>&1 | tee "$OUT/train.log"

echo "=== training DONE. Regression metrics (eval_mse / eval_r2 / eval_pearson / eval_spearman) in $OUT/train.log ==="
echo "    NEXT: score the ADASTRA CTCF eval set by ref-vs-alt Delta with the checkpoint in $OUT,"
echo "          then AUROC vs the 14-model CTCF numbers (ctcf_benchmark_reference_auroc.csv)."
echo "          The scoring script (score_ctcf_adastra.py) is the separate eval step."
