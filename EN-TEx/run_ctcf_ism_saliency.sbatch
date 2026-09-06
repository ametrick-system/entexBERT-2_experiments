#!/bin/bash
#SBATCH --job-name=eb2_ctcf_ism
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=1
#SBATCH --constraint=rtx3090
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# IMPORTANT: run `mkdir -p logs` in the submit dir BEFORE `sbatch` -- SLURM opens this --output
# path at job start (before the script body), so an in-script mkdir is too late and the job
# fails with no log if logs/ is missing.
set -euo pipefail

# ============================================================
# In-silico mutagenesis (ISM) saliency for the CTCF binding TRUNK (Stage-1, regression).
# Scores every single-base substitution over the top CTCF peak windows, re-tokenizing each
# mutant (tokenization-agnostic). Output .npz -> download -> plot_ism_logo.py locally.
#
#   Validates the lever-A conclusion visually: if the DAPT trunk carries the ASB signal,
#   its ISM logo should reproduce the CTCF motif (JASPAR MA0139.1).
# ============================================================

module purge
source ~/.bashrc
conda activate eb2

REPO=$HOME/entexBERT-2
export PYTHONPATH=$REPO/src:${PYTHONPATH:-}

# --- single-track DAPT binding trunk (the 0.7208-lineage trunk, num_labels=1) ---
TRUNK=$HOME/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/binding_reg_ENC-002_CTCF_ctcf_binding_regression_multitissue_enc002/runs/reg
# --- binding windows CSV: must have columns sequence, feature_type (peak/background), binding_label_raw ---
# Reuse the binding build the trunk was trained on (adjust if your path differs):
WINDOWS=$HOME/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/binding_reg_ENC-002_CTCF_ctcf_binding_regression_multitissue_enc002/inputs/ENC-002__TF-ChIP-seq_CTCF/fold0/test.csv
OUT=$HOME/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/ctcf_ism/trunk_ism_300.npz
mkdir -p "$(dirname "$OUT")"

echo "=== preflight ==="
python - <<PY
import torch, os
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
for p in ["$TRUNK/pytorch_model.bin", "$WINDOWS"]:
    print(("found " if os.path.exists(p) else "MISSING "), p)
PY

echo "=== ISM (top 30 CTCF peak windows, 3xL substitutions each) ==="
python "$REPO/src/entexbert2/ism_saliency.py" \
  --checkpoint_dir "$TRUNK" \
  --windows_csv "$WINDOWS" \
  --seq_col sequence --feature_col feature_type --rank_col binding_label_raw \
  --n_windows 300 --batch_size 256 --device cuda \
  --out "$OUT"

echo ""
echo "============================================================"
echo " ISM COMPLETE -> $OUT"
echo " Download it, then locally:"
echo "   python plot_ism_logo.py --ism trunk_ism.npz --pwm_json handoff/ctcf_pwm.json \\"
echo "     --matrix_id MA0139.1 --out ctcf_trunk_ism_logo.png"
echo "============================================================"
