#!/bin/bash
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=rtx3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=eb2_ctcf_nodapt_arms
#SBATCH --output=logs/eb2_ctcf_nodapt_arms_%j.log
#SBATCH --error=logs/eb2_ctcf_nodapt_arms_%j.err
#SBATCH --mail-type=END,FAIL
# =============================================================================
# LEVER A — the two NO-DAPT control arms, run standalone (the a1 arm is already done).
# Both start from the RAW pretrained DNABERT-2 trunk (NO --init_backbone_from) and use the SAME
# baseline Stage-2 ASB split + contrast-head recipe. Single s (no sweep). Quick look.
#
#   arm "nodapt"     : trunk FROZEN, head-only, lr 1e-3.  <- apples-to-apples lever-A test:
#                      identical to a1 / single-track EXCEPT the trunk weights (raw pretrained vs
#                      DAPT). Isolates "does DAPT trunk geometry beat raw pretrained, head fixed?"
#   arm "nodapt_ft"  : trunk UNFROZEN, full fine-tune, lr 2e-5.  <- "can a full fine-tune from raw
#                      pretrained match DAPT?" (deployability). 2e-5 is the standard DNABERT-2
#                      full-finetune LR; the earlier 1e-3 unfrozen run collapsed to chance.
#
# Three-way for the writeup (all leak-free ADASTRA, s=20, same split/head/eval):
#   single-track DAPT -> Stage-2 : 0.7208  (verified baseline)
#   multi-track DAPT  -> Stage-2 : 0.7082  (a1, already scored)
#   raw pretrained (frozen head) -> Stage-2 : this run, arm nodapt
#   raw pretrained (full FT)     -> Stage-2 : this run, arm nodapt_ft
# =============================================================================
set -euo pipefail
mkdir -p logs

module purge
module load StdEnv
module load miniconda
conda activate eb2

# ---- paths (identical to run_ctcf_a1_multitrack.sbatch) ----------------------
REPO=$HOME/entexBERT-2
export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
MODEL=$REPO/DNABERT-2-117M-attention
REF=$HOME/reference_genome/hg38.fa
export HETSNV=$HOME/palmer_scratch/entex_data/hetSNVs.tsv
ASSAY=CTCF
export ADASTRA=$HOME/palmer_scratch/entexBERT-2_experiments/ADASTRA/ctcf_adastra_evalset.csv
export REFCSV=$HOME/palmer_scratch/entexBERT-2_experiments/ADASTRA/ctcf_benchmark_reference_auroc.csv

export A1ROOT=$HOME/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/ctcf_a1_multitrack
export BASE_WORK=$HOME/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/stage2_ctcf_clf
export BUILD=$BASE_WORK/inputs_clf

S=20   # single s (the value that won the single-track baseline -> 0.7208). No sweep.

# ---- preflight --------------------------------------------------------------
echo "=== preflight ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available())"
for f in "$MODEL" "$REF" "$ADASTRA" "$REFCSV" "$BUILD/train.csv" "$BUILD/test.csv" "$BUILD/train.meta.csv" "$BUILD/dev.meta.csv"; do
  [ -e "$f" ] && echo "  found: $f" || { echo "  FATAL missing: $f"; exit 1; }
done

# =============================================================================
# run_arm2 ARM  FREEZE_FLAG  LR   — single s, train, then score leak-free ADASTRA + hetSNV
# =============================================================================
run_arm2 () {
  local ARM="$1"; local FREEZE="$2"; local LR="$3"
  local RUN=$A1ROOT/stage2_$ARM/runs/clf_2a_s${S}
  mkdir -p "$RUN"
  echo ""; echo "=== [$ARM] Stage-2a (frozen-head A/B): s=$S lr=$LR $FREEZE -> $RUN ==="
  python -m entexbert2.finetune_entexbert2 \
      --model_name_or_path "$MODEL" \
      --data_path "$BUILD" \
      --task classification --input_mode hap_pair \
      --pooling_mode center_mean --center_pool_width 5 \
      --head_num_layers 2 --head_hidden_size 128 --proj_dim 128 \
      --balanced_sampler True --neff_s "$S" \
      $FREEZE \
      --model_max_length 512 \
      --per_device_train_batch_size 32 --per_device_eval_batch_size 64 \
      --num_train_epochs 3 \
      --learning_rate "$LR" --warmup_steps 50 --weight_decay 0.01 \
      --logging_steps 20 --eval_steps 250 --save_steps 250 --save_total_limit 2 \
      --evaluation_strategy steps --load_best_model_at_end True \
      --metric_for_best_model auroc --greater_is_better True \
      --bf16 --save_model True --eval_and_save_results True \
      --output_dir "$RUN" --run_name "clf_2a_${ARM}_s${S}"
  local AU=$(python - <<PY
import json,glob,os
st=os.path.join("$RUN","trainer_state.json")
val=json.load(open(st)).get("best_metric") if os.path.exists(st) else None
if val is None:
    er=glob.glob("$RUN/results/*/eval_results.json")
    if er: val=json.load(open(er[0])).get("eval_auroc")
print(val if val is not None else -1)
PY
)
  echo "  [$ARM] dev AUROC = $AU"
  echo "=== [$ARM] score leak-free ADASTRA + hetSNV ==="
  python -m entexbert2.score_asb \
      --eval adastra --checkpoint_dir "$RUN" --ref_fasta "$REF" \
      --eval_csv "$ADASTRA" --assay "$ASSAY" ${REFCSV:+--reference_csv "$REFCSV"} \
      --train_coords "$BUILD/train.meta.csv" "$BUILD/dev.meta.csv" --drop_leaky \
      --left_bp 128 --right_bp 128 --batch_size 64 --device cuda \
      --out "$A1ROOT/score_${ARM}_adastra"
  python -m entexbert2.score_asb \
      --eval hetsnv --checkpoint_dir "$RUN" --ref_fasta "$REF" \
      --hetsnv_tsv "$HETSNV" --assay "$ASSAY" \
      --train_coords "$BUILD/train.meta.csv" "$BUILD/dev.meta.csv" --drop_leaky \
      --left_bp 128 --right_bp 128 --batch_size 64 --device cuda \
      --out "$A1ROOT/score_${ARM}_hetsnv"
  echo "[$ARM] scored -> $A1ROOT/score_${ARM}_adastra*"
}

# arm 1: raw pretrained trunk, FROZEN, head-only @1e-3 (true apples-to-apples with a1)
run_arm2 "nodapt"    "--freeze_backbone True"  "1e-3"
# arm 2: raw pretrained trunk, UNFROZEN, full fine-tune @2e-5 (deployability)
run_arm2 "nodapt_ft" ""                        "2e-5"

echo ""
echo "============================================================"
echo " NO-DAPT ARMS COMPLETE (s=$S)."
echo "   nodapt    (frozen head)  -> $A1ROOT/score_nodapt_adastra*"
echo "   nodapt_ft (full FT)      -> $A1ROOT/score_nodapt_ft_adastra*"
echo " Compare to: single-track 0.7208 | a1 multi-track 0.7082 (leak-free ADASTRA)."
echo "============================================================"
