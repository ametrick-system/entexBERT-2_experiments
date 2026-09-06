#!/bin/bash
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=rtx3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=eb2_ctcf_lm_head
#SBATCH --output=logs/eb2_ctcf_lm_head_%j.log
#SBATCH --error=logs/eb2_ctcf_lm_head_%j.err
#SBATCH --mail-type=END,FAIL
# =============================================================================
# HEAD IMPROVEMENT #1 — learned-metric (Mahalanobis) contrast head.
# Replaces the Stage-2 contrast distance  s = ||z1 - z2||  with  s = ||L (z1 - z2)||,
# L a proj_dim x proj_dim linear map INITIALIZED TO IDENTITY (so step-0 == the plain head;
# it only departs from Euclidean as L trains, weighting the ASB-relevant directions).
# Enabled by --learned_metric True.
#
# Clean A/B against the verified single-track baseline (SAME trunk, split, head recipe, eval;
# the ONLY change is the metric map):
#   single-track DAPT trunk -> plain-distance head   : 0.7208  (verified baseline)
#   single-track DAPT trunk -> learned-metric head   : this run
#
# The trunk is the single-track binding DAPT checkpoint (frozen, Stage-2a only) -- the exact
# trunk that produced 0.7208, so any delta is attributable to the head change alone.
#
# PREREQS (fail-fast preflight): repo patched with the #1 support:
#   model.py (learned_metric ctor param + identity-init metric_map + forward),
#   model_io.py (build_model reads learned_metric from run_config),
#   finetune_entexbert2.py (--learned_metric arg -> ctor -> run_config).
#
# DIAG=0 (default): full run, s-sweep {s_hat,20,50,100}, 3 epochs.
# DIAG=1: capped steps + {s_hat,20} for a fast end-to-end check.
#   sbatch --export=ALL,DIAG=1 run_ctcf_stage2_learned_metric.sbatch
# =============================================================================
set -euo pipefail
mkdir -p logs

module purge
module load StdEnv
module load miniconda
conda activate eb2

# ---- paths ------------------------------------------------------------------
REPO=$HOME/entexBERT-2
export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
MODEL=$REPO/DNABERT-2-117M-attention
REF=$HOME/reference_genome/hg38.fa
export HETSNV=$HOME/palmer_scratch/entex_data/hetSNVs.tsv
ASSAY=CTCF
export ADASTRA=$HOME/palmer_scratch/entexBERT-2_experiments/ADASTRA/ctcf_adastra_evalset.csv
export REFCSV=$HOME/palmer_scratch/entexBERT-2_experiments/ADASTRA/ctcf_benchmark_reference_auroc.csv

# single-track binding DAPT trunk = the exact checkpoint behind the 0.7208 baseline.
export STAGE1_CKPT=$HOME/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/binding_reg_ENC-002_CTCF_ctcf_binding_regression_multitissue_enc002/runs/reg/pytorch_model.bin

# Stage-2 ASB build: REUSE the exact baseline split so the A/B differs ONLY in the head.
export BASE_WORK=$HOME/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/stage2_ctcf_clf
export COUNTS=$BASE_WORK/ctcf_betabinom_counts.csv
export BUILD=$BASE_WORK/inputs_clf

# separate work root so nothing from the baseline/A1 runs is clobbered
export WORK=$HOME/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/stage2_ctcf_lm_head
mkdir -p "$WORK"

# ---- DIAG toggle ------------------------------------------------------------
DIAG="${DIAG:-0}"
if [[ "$DIAG" == "1" ]]; then
  echo "*** DIAGNOSTIC MODE: capped steps + 2-value s-sweep. NOT a final result. ***"
  S2A_MAX_STEPS=400; EVAL_EVERY=100; DIAG_SGRID="s_hat 20"
else
  echo "*** FULL RUN: Stage-2a 3-epoch s-sweep {s_hat,20,50,100}, learned-metric head. ***"
  S2A_MAX_STEPS=0;   EVAL_EVERY=250; DIAG_SGRID=""
fi
maxsteps() { [[ "${1:-0}" -gt 0 ]] && echo "--max_steps $1" || true; }

# ---- preflight --------------------------------------------------------------
echo "=== preflight ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available())"
# repo must carry the #1 support
grep -q "learned_metric" "$REPO/src/entexbert2/model.py"                  && echo "  repo OK: model.py learned_metric" || { echo "  FATAL: model.py missing learned_metric"; exit 1; }
grep -q "learned_metric" "$REPO/src/entexbert2/model_io.py"               && echo "  repo OK: model_io.py learned_metric" || { echo "  FATAL: model_io.py missing learned_metric"; exit 1; }
grep -q "learned_metric" "$REPO/src/entexbert2/finetune_entexbert2.py"    && echo "  repo OK: finetune learned_metric" || { echo "  FATAL: finetune missing learned_metric"; exit 1; }
for f in "$MODEL" "$REF" "$ADASTRA" "$REFCSV" "$STAGE1_CKPT" "$BUILD/train.csv" "$BUILD/test.csv" "$BUILD/train.meta.csv" "$BUILD/dev.meta.csv"; do
  [ -e "$f" ] && echo "  found: $f" || { echo "  FATAL missing: $f"; exit 1; }
done

# =============================================================================
# STEP 1 — s_hat from the betabinom counts (reuse baseline counts)
# =============================================================================
echo ""; echo "=== STEP 1: s_hat ==="
[[ -f "$COUNTS" ]] || { echo "[STEP 1] counts absent -> building"; python -m entexbert2.build_betabinom_counts --hetsnv_tsv "$HETSNV" --assay "$ASSAY" --min_total_reads 20 --out "$COUNTS"; }
SHAT=$(python - <<'PY'
import os, pandas as pd, numpy as np
df=pd.read_csv(os.environ["COUNTS"]); k=df["k"].to_numpy(float); n=df["n"].to_numpy(float)
pbar=k.sum()/n.sum(); num=np.sum((k-n*pbar)**2/(n*pbar*(1-pbar)))
N=len(n); nbar=n.mean(); rho=(num-(N-1))/(np.sum(n)-nbar); rho=min(max(rho,1e-3),0.99); s=(1-rho)/rho
print(f"{s:.1f}")
PY
)
if [[ -n "$DIAG_SGRID" ]]; then SGRID=$(echo "$DIAG_SGRID" | sed "s/s_hat/$SHAT/"); else SGRID="$SHAT 20 50 100"; fi
echo "[STEP 1] s_hat=$SHAT | s-sweep = $SGRID"

# =============================================================================
# STEP 2 — Stage-2a: single-track DAPT trunk (frozen) + learned-metric head, s-sweep
# =============================================================================
BEST=""; BEST_S=""; BEST_AU="-1"
for S in $SGRID; do
  RUN=$WORK/runs/clf_2a_lm_s${S}
  mkdir -p "$RUN"
  echo ""; echo "--- Stage-2a (learned-metric) s=$S -> $RUN ---"
  python -m entexbert2.finetune_entexbert2 \
      --model_name_or_path "$MODEL" \
      --data_path "$BUILD" \
      --task classification --input_mode hap_pair \
      --init_backbone_from "$STAGE1_CKPT" --freeze_backbone True \
      --pooling_mode center_mean --center_pool_width 5 \
      --head_num_layers 2 --head_hidden_size 128 --proj_dim 128 \
      --learned_metric True \
      --balanced_sampler True --neff_s "$S" \
      --model_max_length 512 \
      --per_device_train_batch_size 32 --per_device_eval_batch_size 64 \
      --num_train_epochs 3 $(maxsteps "$S2A_MAX_STEPS") \
      --learning_rate 1e-3 --warmup_steps 50 --weight_decay 0.01 \
      --logging_steps 20 --eval_steps "$EVAL_EVERY" --save_steps "$EVAL_EVERY" --save_total_limit 2 \
      --evaluation_strategy steps --load_best_model_at_end True \
      --metric_for_best_model auroc --greater_is_better True \
      --bf16 --save_model True --eval_and_save_results True \
      --output_dir "$RUN" --run_name "clf_2a_lm_s${S}"
  AU=$(python - <<PY
import json,glob,os
st=os.path.join("$RUN","trainer_state.json")
val=json.load(open(st)).get("best_metric") if os.path.exists(st) else None
if val is None:
    er=glob.glob("$RUN/results/*/eval_results.json")
    if er: val=json.load(open(er[0])).get("eval_auroc")
print(val if val is not None else -1)
PY
)
  echo "  [lm] s=$S dev AUROC = $AU"
  awk "BEGIN{exit !($AU > $BEST_AU)}" && { BEST_AU="$AU"; BEST="$RUN"; BEST_S="$S"; }
done
echo ""; echo "[STEP 2] BEST 2a: s=$BEST_S dev_AUROC=$BEST_AU ckpt=$BEST"
[[ -n "$BEST" ]] || { echo "FATAL: no 2a run succeeded"; exit 1; }

# =============================================================================
# STEP 3 — score best checkpoint leak-free (ADASTRA + hetSNV)
# =============================================================================
echo ""; echo "=== STEP 3: score leak-free (best 2a) ==="
python -m entexbert2.score_asb \
    --eval adastra --checkpoint_dir "$BEST" --ref_fasta "$REF" \
    --eval_csv "$ADASTRA" --assay "$ASSAY" ${REFCSV:+--reference_csv "$REFCSV"} \
    --train_coords "$BUILD/train.meta.csv" "$BUILD/dev.meta.csv" --drop_leaky \
    --left_bp 128 --right_bp 128 --batch_size 64 --device cuda \
    --out "$WORK/score_lm_adastra"
python -m entexbert2.score_asb \
    --eval hetsnv --checkpoint_dir "$BEST" --ref_fasta "$REF" \
    --hetsnv_tsv "$HETSNV" --assay "$ASSAY" \
    --train_coords "$BUILD/train.meta.csv" "$BUILD/dev.meta.csv" --drop_leaky \
    --left_bp 128 --right_bp 128 --batch_size 64 --device cuda \
    --out "$WORK/score_lm_hetsnv"

echo ""
echo "============================================================"
echo " LEARNED-METRIC HEAD (#1) COMPLETE.  (DIAG=$DIAG)"
echo "   best 2a: s=$BEST_S dev_AUROC=$BEST_AU"
echo "   scored -> $WORK/score_lm_adastra*"
echo " A/B: plain-distance head was leak-free AUROC 0.7208 [0.7031,0.7380]."
echo "   lm > 0.7208 (non-overlapping CI) => learned metric helps."
[[ "$DIAG" == "1" ]] && echo " NOTE: DIAG mode -> UNDER-TRAINED, not final. Rerun DIAG=0 for the real number."
echo "============================================================"
