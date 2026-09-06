#!/usr/bin/env bash
#SBATCH --job-name=eb2_ctcf_staircase
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

# ============================================================
# CTCF DNABERT-2 "staircase" probe — ONE frozen-trunk protocol x 3 trunk states x 2 window sets.
# Produces the missing rungs BELOW the already-verified fine-tuned ASB numbers:
#   binding : peak-vs-background AUROC + peak-strength Spearman, trunk={random,pretrained,finetuned}
#   asb     : zero-shot embedding distance (pretrained) + frozen twin-probe {random,pretrained}
# The verified fine-tuned rungs (ref_single 0.5796, paired 0.7208, +stem 0.7271) are merged into
# the figure DOWNSTREAM from saved numbers — NOT recomputed here.
#
# Trunk is always FROZEN; only the small probe trains (linear/CNN/twin). Two GPU embedding passes;
# probes are CPU-fast. Everything self-contained (AutoModel), no dependency on the entexbert2 refactor.
# ============================================================

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-eb2}"
MODEL="${MODEL:-$HOME/entexBERT-2/DNABERT-2-117M-attention}"
REF_FASTA="${REF_FASTA:-$HOME/reference_genome/hg38.fa}"

# WORK dir for this experiment's fresh inputs (kept separate from the training experiments).
WORK="${WORK:-$HOME/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/ctcf_staircase}"
BINDING_DIR="$WORK/binding_windows"     # regenerated binding peak/bg windows (this run)
PAIRS_CSV="$WORK/adastra_pairs.csv"      # ADASTRA ref/alt pairs, leak-free (this run)

# The CTCF binding config (multi_tissue_peak, ref_single, feature_type peak/background) + its trunk.
BINDING_CONFIG="${BINDING_CONFIG:-configs/ctcf_binding_regression_multitissue_enc002.yaml}"
FINETUNED_TRUNK="${FINETUNED_TRUNK:-experiments/binding_reg_ENC-002_CTCF_ctcf_binding_regression_multitissue_enc002/runs/reg}"

# ASB twin-probe TRAINING pairs = the Stage-2 classification inputs the CTCF stem run used.
ASB_TRAIN_DIR="${ASB_TRAIN_DIR:-experiments/stage2_ctcf_clf/inputs_clf}"
# The ADASTRA benchmark eval set + the leak-coords of the twin probe's own training set.
ADASTRA_EVAL="/home/asm242/palmer_scratch/entexBERT-2_experiments/ADASTRA/ctcf_adastra_evalset.csv"
TRAIN_COORDS_1="$ASB_TRAIN_DIR/train.meta.csv"
TRAIN_COORDS_2="$ASB_TRAIN_DIR/dev.meta.csv"
REFERENCE_CSV="${REFERENCE_CSV:-ctcf_benchmark_reference_auroc.csv}"

OUT="${OUT:-$WORK/staircase_ctcf}"
mkdir -p "$WORK" "$BINDING_DIR" logs

module load miniconda 2>/dev/null || true
# shellcheck disable=SC1091
source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

# ============================================================
# STEP 1 — regenerate binding peak/background windows from the CTCF binding config.
# Uses the CURRENT contract: run_experiment writes FLAT train/dev/test(.meta).csv to --output_dir.
# ref_single -> single 'sequence' column; feature_type (peak|background) rides in *.meta.csv.
# ============================================================
if [[ -f "$BINDING_DIR/train.csv" && -f "$BINDING_DIR/test.meta.csv" ]]; then
    echo "[build] $BINDING_DIR already populated — skipping (delete to force rebuild)"
else
    echo "[build] regenerating CTCF binding peak/background windows -> $BINDING_DIR"
    python -m entexbert2.run_experiment "$BINDING_CONFIG" \
        --ref_fasta "$REF_FASTA" \
        --output_dir "$BINDING_DIR"
fi

# preflight: single 'sequence' column + feature_type present in meta
HDR="$(head -1 "$BINDING_DIR/train.csv")"
grep -q "sequence" <<< "$HDR" || { echo "FATAL: no 'sequence' column in binding train.csv"; exit 1; }
grep -q "sequence1" <<< "$HDR" && { echo "FATAL: found 'sequence1' — paired build; ref_single expected"; exit 1; }
head -1 "$BINDING_DIR/test.meta.csv" | grep -q "feature_type" \
    || { echo "FATAL: 'feature_type' missing from test.meta.csv (needed for peak/bg label)"; exit 1; }
echo "[build] OK: single-window binding build with feature_type"

# ============================================================
# STEP 2 — build ADASTRA ref/alt pairs, LEAK-FREE against the twin probe's own EN-TEx training bins.
# --drop_leaky pre-filters, so the emitted pairs are already leak-free (sequence1,sequence2,label).
# The probe scores every emitted row (no chr/pos needed downstream).
# ============================================================
if [[ -f "$PAIRS_CSV" ]]; then
    echo "[build] $PAIRS_CSV exists — skipping (delete to force rebuild)"
else
    echo "[build] building leak-free ADASTRA pairs -> $PAIRS_CSV"
    python /home/asm242/entexBERT-2/build_adastra_pairs.py \
        --eval_csv "$ADASTRA_EVAL" \
        --ref_fasta "$REF_FASTA" \
        --out "$PAIRS_CSV" \
        --left_bp 128 --right_bp 128 \
        --train_coords "$TRAIN_COORDS_1" "$TRAIN_COORDS_2" \
        --bin_size 100000 \
        --drop_leaky
fi

# ============================================================
# STEP 3 — the staircase probe. Trunk frozen; probes train on cached embeddings.
# ============================================================
echo "=== [staircase] 3 trunk states x {binding, asb} ==="
python staircase_probe.py \
    --backbone_path   "$MODEL" \
    --finetuned_trunk "$FINETUNED_TRUNK" \
    --binding_dir     "$BINDING_DIR" \
    --asb_train_dir   "$ASB_TRAIN_DIR" \
    --adastra_pairs   "$PAIRS_CSV" \
    --reference_csv   "$REFERENCE_CSV" \
    --bin_size 100000 \
    --batch_size 64 \
    --max_len 128 \
    --proj_dim 128 \
    --seed 0 \
    --device cuda \
    --out "$OUT" \
    2>&1 | tee "$OUT.log"

echo "=== staircase DONE -> ${OUT}_staircase_metrics.json ==="
echo "    Pull ${OUT}_staircase_metrics.json back to build the two-panel staircase figure."
