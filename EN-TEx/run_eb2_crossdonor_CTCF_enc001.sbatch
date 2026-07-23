#!/usr/bin/env bash
#SBATCH --job-name=eb2_crossdonor_CTCF_enc001
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

# ---- which ENC-001 run to evaluate: pass the config TAG whose checkpoint you want to test ----
# default = the DNABERT-1-matched downsample run (0.711). Override e.g.:
#   BASECFG=configs/ctcf_ref_single_classification_binsplit2_balall_enc001.yaml \
#   ENC001_EXP=experiments/refsingle_ENC-001_CTCF_ctcf_ref_single_classification_binsplit2_balall_enc001 \
#   sbatch run_eb2_crossdonor_CTCF_enc001.sbatch
BASECFG="${BASECFG:-configs/ctcf_ref_single_classification_binsplit2_balall_enc001.yaml}"
ENC001_EXP="${ENC001_EXP:-experiments/refsingle_ENC-001_CTCF_ctcf_ref_single_classification_binsplit2_balall_enc001}"
CKPT="$ENC001_EXP/runs/clf"
ENC001_TRAIN_META="$ENC001_EXP/inputs/ENC-001__TF-ChIP-seq_CTCF/fold0/train.meta.csv"

XD_INPUTS="${XD_INPUTS:-experiments/crossdonor_CTCF_enc001/inputs}"
XD_POOL="${XD_POOL:-experiments/crossdonor_CTCF_enc001/pool}"
CONDA_ENV="${CONDA_ENV:-eb2}"

module load miniconda 2>/dev/null || true
source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

echo "=== preflight ==="
test -e "$CKPT/pytorch_model.bin" || { echo "MISSING checkpoint $CKPT/pytorch_model.bin -- train the ENC-001 run first"; exit 1; }
test -e "$CKPT/run_config.json"   || { echo "MISSING $CKPT/run_config.json"; exit 1; }
test -s "$ENC001_TRAIN_META"      || { echo "MISSING $ENC001_TRAIN_META"; exit 1; }
echo "  checkpoint: $CKPT"
echo "  base config (same salt as ENC-001!): $BASECFG"

# STEP 1 -- build ENC-002/3/4 inputs with the SAME base config+salt (=> same global bin->split map)
echo "=== STEP 1: build cross-donor inputs (same salt as ENC-001) ==="
python -m entexbert2.scripts.generate_all_inputs "$BASECFG" \
    --datasets_csv crossdonor_datasets_ctcf.csv \
    --output_dir "$XD_INPUTS"

# STEP 2 -- pool the three donors' fold0/test sets + belt-and-suspenders locus exclusion
echo "=== STEP 2: pool + leak-check ==="
python build_eb2_crossdonor_pool.py \
    --inputs_root "$XD_INPUTS" \
    --donors ENC-002,ENC-003,ENC-004 \
    --assay TF-ChIP-seq_CTCF \
    --enc001_train_meta "$ENC001_TRAIN_META" \
    --out_dir "$XD_POOL"

# STEP 3 -- evaluate the ENC-001 checkpoint on the pooled cross-donor test (no retrain)
echo "=== STEP 3: analyze ENC-001 checkpoint on cross-donor pool ==="
python -m entexbert2.analyze \
    --checkpoint_dir "$CKPT" \
    --data_csv "$XD_POOL/test.csv" \
    --output_dir "$XD_POOL/eval" \
    --threshold 0.5
echo "=== done: metrics -> $XD_POOL/eval ==="
cat "$XD_POOL/eval"/*.json 2>/dev/null | head -40
