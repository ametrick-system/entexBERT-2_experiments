#!/bin/bash
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=rtx3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=eb2_stage2_ctcf
#SBATCH --output=eb2_stage2_ctcf_%j.log
#SBATCH --mail-type=END,FAIL

# =============================================================================
# entexBERT-2 Stage-2 ASB experiment  (CTCF, ENC-002 trunk)
# -----------------------------------------------------------------------------
# Two-stage: reuse the ENC-002 binding-regression trunk (Stage 1, gave |Delta|~0.68),
# add the ASB twin head g_phi trained under privileged precision weighting w = n_eff(n).
#   STEP 0  estimate s-hat (beta-binomial dispersion) via method-of-moments on real counts
#   STEP 1  build the full Stage-2 dataset (all CTCF loci, chr5+chr12 held out to TEST)
#   STEP 2a Stage-2a: FROZEN trunk, linear head, sweep s in {s_hat, grid neighbors}
#   STEP 2b Stage-2b: gentle unfreeze from the best 2a checkpoint (low LR)
#   STEP 3  score ADASTRA + hetSNV, leak-free
# Head: linear (head_num_layers=1). Loss: precision-weighted MSE (beta-binom NLL collapsed).
# =============================================================================

set -euo pipefail

module purge
module load StdEnv
module load miniconda
conda activate eb2

# ---- paths (EDIT if yours differ) -------------------------------------------
REPO=$HOME/entexBERT-2
export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
MODEL=$REPO/DNABERT-2-117M-attention
REF=$HOME/reference_genome/hg38.fa
HETSNV=$HOME/palmer_scratch/entex_data/hetSNVs.tsv
ADASTRA=$HOME/palmer_scratch/entexBERT-2_experiments/ADASTRA/ctcf_adastra_evalset.csv
REFCSV=$HOME/palmer_scratch/entexBERT-2_experiments/ADASTRA/ctcf_benchmark_reference_auroc.csv
# Stage-1 binding trunk (the ~0.68 checkpoint). Must contain backbone.* keys (verified: it does).
STAGE1=$HOME/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/binding_reg_ENC-002_CTCF_ctcf_binding_regression_multitissue_enc002/runs/reg/pytorch_model.bin

WORK=$HOME/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/stage2_ctcf_asb
BUILD=$WORK/inputs
mkdir -p "$WORK"
cd "$REPO"

echo "=== preflight ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch,transformers; print('torch',torch.__version__,'| cuda',torch.cuda.is_available())"
for f in "$MODEL" "$REF" "$HETSNV" "$ADASTRA" "$STAGE1"; do
  [ -e "$f" ] && echo "  found: $f" || { echo "  MISSING: $f"; exit 1; }
done

# =============================================================================
# STEP 0 -- aggregate real CTCF counts, then estimate s_hat by method-of-moments.
# Beta-binomial: for a locus with counts (k, n) and mean p, the dispersion rho is
# estimated from the excess variance of the per-locus allelic fraction vs binomial.
# n_eff = n(1+s)/(n+s) with s = (1-rho)/rho, so we report s_hat and its grid neighbors.
# =============================================================================
echo ""; echo "=== STEP 0: aggregate counts + estimate s_hat ==="
COUNTS=$WORK/ctcf_betabinom_counts.csv
python build_betabinom_counts.py \
    --hetsnv_tsv "$HETSNV" --assay CTCF --min_total_reads 20 --out "$COUNTS"

SHAT=$(python - <<PY
import pandas as pd, numpy as np
df = pd.read_csv("$COUNTS")
k = df["k"].to_numpy(float); n = df["n"].to_numpy(float)
p = k / n
# Method-of-moments for beta-binomial intra-class correlation rho (Kleinman 1973):
# pooled mean, then rho from the ratio of observed to binomial (chi-square style) dispersion.
pbar = k.sum() / n.sum()
# per-locus squared standardized residual; S = sum w_i (p_i - pbar)^2 / (pbar(1-pbar))
num = np.sum((k - n*pbar)**2 / (n*pbar*(1-pbar)))
N = len(n); nbar = n.mean()
# Kleinman moment estimator of rho
rho = (num - (N - 1)) / (np.sum(n) - nbar)   # approx; clamp to (0,1)
rho = min(max(rho, 1e-3), 0.99)
s = (1 - rho) / rho
print(f"{s:.1f}")
import sys
print(f"[s_hat] rho={rho:.4f} -> s_hat={s:.1f}  (n loci={N}, mean n={nbar:.1f})", file=sys.stderr)
PY
)
echo "[STEP 0] s_hat = $SHAT"
# Sweep s_hat plus the two standard grid neighbors that bracket it.
SGRID="$SHAT 20 50 100"
echo "[STEP 0] s sweep = $SGRID"
echo "[STEP 0] OK"

# =============================================================================
# STEP 1 -- build the full Stage-2 dataset (all CTCF loci; chr5+chr12 -> TEST).
# =============================================================================
echo ""; echo "=== STEP 1: build Stage-2 dataset ==="
cat > "$WORK/stage2.yaml" <<YAML
experiment: stage2_ctcf_asb_enc002
ref_fasta: $REF
output_dir: $BUILD
row_source: {type: betabinom_counts, path: $COUNTS, assay: CTCF, min_total_reads: 20}
primary_label: {type: logit_ratio, name: logit_ratio}
sequence: {input_mode: hap_pair}
window: {left_bp: 128, right_bp: 128, snv_offset_mode: fixed, jitter_max_bp: 0}
head: {task: regression, num_labels: 1, head_num_layers: 1, head_hidden_size: -1}
depth_col: n
split: {mode: train_dev_test, ratio: [0.8, 0.1, 0.1], seed: 42, group: locus}
partition:
  enabled: true
  bin_size: 100000
  fold_id: 0
  fold_assignment: {chr5: 0, chr12: 0}
  exclude_boundary: true
  boundary_bp: 128
balance: {strategy: none}
YAML

python -m entexbert2.run_experiment "$WORK/stage2.yaml" --ref_fasta "$REF" --output_dir "$BUILD"
python -c "import pandas as pd; d=pd.read_csv('$BUILD/train.csv'); print('[STEP 1] train.csv cols:', list(d.columns), '| rows:', len(d))"
echo "[STEP 1] OK"

# =============================================================================
# STEP 2a -- Stage-2a: FROZEN trunk, LINEAR head, sweep s. Pick best by dev spearman.
# =============================================================================
echo ""; echo "=== STEP 2a: frozen-trunk head training (s sweep) ==="
BEST_2A=""; BEST_S=""; BEST_SP="-1"
for S in $SGRID; do
  RUN=$WORK/runs/asb_2a_s${S}
  echo "--- Stage-2a s=$S -> $RUN ---"
  python -m entexbert2.finetune_entexbert2 \
      --model_name_or_path "$MODEL" \
      --data_path "$BUILD" \
      --task regression --input_mode hap_pair \
      --pooling_mode center_mean --center_pool_width 5 \
      --head_num_layers 1 --neff_s "$S" \
      --init_backbone_from "$STAGE1" --freeze_backbone True \
      --model_max_length 512 \
      --per_device_train_batch_size 32 --per_device_eval_batch_size 64 \
      --num_train_epochs 5 --learning_rate 1e-3 --warmup_steps 50 --weight_decay 0.01 \
      --logging_steps 20 --eval_steps 100 --save_steps 100 --save_total_limit 1 \
      --evaluation_strategy steps --load_best_model_at_end True \
      --metric_for_best_model spearman --greater_is_better True \
      --bf16 --save_model True --eval_and_save_results True \
      --output_dir "$RUN" --run_name "asb_2a_s${S}"
  # track best dev spearman across the sweep
  SP=$(python - <<PY
import json,glob,os
# best_model_at_end restored; read the trainer state for best_metric, else eval_results
st=os.path.join("$RUN","trainer_state.json")
val=None
if os.path.exists(st):
    val=json.load(open(st)).get("best_metric")
if val is None:
    er=glob.glob("$RUN/results/*/eval_results.json")
    if er: val=json.load(open(er[0])).get("eval_spearman")
print(val if val is not None else -1)
PY
)
  echo "    dev spearman = $SP"
  awk "BEGIN{exit !($SP > $BEST_SP)}" && { BEST_SP="$SP"; BEST_2A="$RUN"; BEST_S="$S"; }
done
echo "[STEP 2a] BEST: s=$BEST_S  dev_spearman=$BEST_SP  ckpt=$BEST_2A"
echo "[STEP 2a] OK"

# =============================================================================
# STEP 2b -- Stage-2b: gentle unfreeze from the best 2a checkpoint (low LR, no freeze).
# =============================================================================
echo ""; echo "=== STEP 2b: gentle unfreeze from best 2a (s=$BEST_S) ==="
RUN2B=$WORK/runs/asb_2b_s${BEST_S}
python -m entexbert2.finetune_entexbert2 \
    --model_name_or_path "$MODEL" \
    --data_path "$BUILD" \
    --task regression --input_mode hap_pair \
    --pooling_mode center_mean --center_pool_width 5 \
    --head_num_layers 1 --neff_s "$BEST_S" \
    --init_backbone_from "$BEST_2A/pytorch_model.bin" \
    --model_max_length 512 \
    --per_device_train_batch_size 16 --per_device_eval_batch_size 64 \
    --num_train_epochs 2 --learning_rate 3e-6 --warmup_steps 50 --weight_decay 0.01 \
    --logging_steps 20 --eval_steps 100 --save_steps 100 --save_total_limit 1 \
    --evaluation_strategy steps --load_best_model_at_end True \
    --metric_for_best_model spearman --greater_is_better True \
    --bf16 --save_model True --eval_and_save_results True \
    --output_dir "$RUN2B" --run_name "asb_2b_s${BEST_S}"
echo "[STEP 2b] OK"

# =============================================================================
# STEP 3 -- score ADASTRA + hetSNV (leak-free) for both the best 2a and the 2b model.
# =============================================================================
echo ""; echo "=== STEP 3: score (leak-free) ==="
for TAG in "2a:$BEST_2A" "2b:$RUN2B"; do
  NAME=${TAG%%:*}; CKPT=${TAG#*:}
  echo "--- scoring $NAME ($CKPT) ---"
  python -m entexbert2.score_asb \
      --eval adastra --checkpoint_dir "$CKPT" --ref_fasta "$REF" \
      --eval_csv "$ADASTRA" ${REFCSV:+--reference_csv "$REFCSV"} \
      --train_coords "$BUILD/train.meta.csv" "$BUILD/dev.meta.csv" --drop_leaky \
      --left_bp 128 --right_bp 128 --batch_size 64 --device cuda \
      --out "$WORK/score_${NAME}_adastra"
  python -m entexbert2.score_asb \
      --eval hetsnv --checkpoint_dir "$CKPT" --ref_fasta "$REF" \
      --hetsnv_tsv "$HETSNV" --assay CTCF \
      --train_coords "$BUILD/train.meta.csv" "$BUILD/dev.meta.csv" --drop_leaky \
      --left_bp 128 --right_bp 128 --batch_size 64 --device cuda \
      --out "$WORK/score_${NAME}_hetsnv"
done
echo "[STEP 3] OK"
echo ""
echo "============================================================"
echo " STAGE-2 COMPLETE."
echo "   best s (dev spearman): $BEST_S"
echo "   2a ckpt: $BEST_2A"
echo "   2b ckpt: $RUN2B"
echo "   scores:  $WORK/score_2a_*  and  $WORK/score_2b_*"
echo " Compare leak-free |Delta| AUROC vs the ~0.68 probe baseline"
echo " and the benchmark table."
echo "============================================================"
