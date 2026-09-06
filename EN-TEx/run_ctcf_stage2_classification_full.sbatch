#!/bin/bash
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=rtx3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=eb2_ctcf_clf
#SBATCH --output=logs/eb2_ctcf_clf_%j.log
#SBATCH --error=logs/eb2_ctcf_clf_%j.err
#SBATCH --mail-type=END,FAIL
# =============================================================================
# CTCF Stage-2 CLASSIFICATION (contrast head).  s = ||P(h1) - P(h2)||, p = sigmoid(a*s + b),
# precision-weighted BCE on the binary AS label. REUSES the already-trained ENC-002 CTCF Stage-1
# binding trunk (init_backbone_from loads backbone.* ONLY, so the trunk's old regression head is
# irrelevant) -- only Stage-2 is done here.
#
# This is the APPLES-TO-APPLES comparison to the regression two-stage on CTCF: same ENC-002 CTCF
# trunk, same TF. Regression-twin leak-free AUROC was 0.6652 (2a) / 0.6748 (2b); this run asks
# whether the symmetric contrast-classification head beats that on identical inputs.
#
#   STEP 4c  build Stage-2 CLASSIFICATION dataset (hap_pair, as_class label, held-out chroms -> TEST)
#   STEP 5c  Stage-2a: FROZEN trunk, contrast head (proj_dim d), balanced sampler; sweep s; pick best by dev AUROC
#   STEP 6c  Stage-2b: gentle unfreeze from best-2a (low LR)
#   STEP 7c  score ADASTRA + hetSNV, leak-free, for best-2a and (if it beats 2a on dev) 2b
#
# PREREQS (fail-fast in preflight):
#   * trained Stage-1 trunk  $STAGE1_CKPT
#   * beta-binomial counts   $COUNTS         (rebuilt from hetSNVs if absent)
#   * ADASTRA eval set + CTCF reference AUROC
#   * PATCHED repo: model.py (contrast head), model_io.py (task-aware inference),
#     run_experiment.py (as_class label), finetune_entexbert2.py (classification + balanced_sampler),
#     score_asb.py (task-aware ranking).  Preflight greps for the as_class builder to catch a stale repo.
#
# DIAG=1 (default): capped steps + 2-value s-sweep for a fast end-to-end check.
# DIAG=0          : full run (raise --time to ~12h).
# =============================================================================
set -euo pipefail
mkdir -p logs

module purge
module load StdEnv
module load miniconda
conda activate eb2

# ---- paths (must match the regression run) ----------------------------------
REPO=$HOME/entexBERT-2
export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
MODEL=$REPO/DNABERT-2-117M-attention
REF=$HOME/reference_genome/hg38.fa
export HETSNV=$HOME/palmer_scratch/entex_data/hetSNVs.tsv
ASSAY=CTCF
export ADASTRA=$HOME/palmer_scratch/entexBERT-2_experiments/ADASTRA/ctcf_adastra_evalset.csv
export REFCSV=$HOME/palmer_scratch/entexBERT-2_experiments/ADASTRA/ctcf_benchmark_reference_auroc.csv
# fresh Stage-2 workdir for the classification head (keeps the regression stage2_ctcf_asb/ intact)
export WORK=$HOME/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/stage2_ctcf_clf_full
mkdir -p "$WORK"
# REUSE the trained ENC-002 CTCF Stage-1 trunk (EDIT if your path differs; preflight will halt if missing).
# init_backbone_from loads backbone.* only, so the trunk's regression head is discarded -- fine.
export STAGE1_CKPT=$HOME/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/binding_reg_ENC-002_CTCF_ctcf_binding_regression_multitissue_enc002/runs/reg/pytorch_model.bin
export COUNTS=$WORK/ctcf_betabinom_counts.csv   # rebuilt from hetSNVs (assay=CTCF) if absent
export BUILD=$WORK/inputs_clf                    # Stage-2 classification dataset
cd "$REPO"

# ---- diagnostic vs full -----------------------------------------------------
DIAG="${DIAG:-0}"   # FULL run by default; set DIAG=1 for the fast capped check
if [[ "$DIAG" == "1" ]]; then
  echo "*** DIAGNOSTIC MODE: capped steps + 2-value s-sweep. NOT a final result. ***"
  S2A_MAX_STEPS=400; S2B_MAX_STEPS=300; EVAL_EVERY=100; DIAG_SGRID="s_hat 20"
else
  echo "*** FULL RUN: 3-epoch 2a over s-sweep {s_hat,20,50,100} + 2-epoch 2b. This is the real number. ***"
  S2A_MAX_STEPS=0;   S2B_MAX_STEPS=0;   EVAL_EVERY=250; DIAG_SGRID=""   # epochs (no step cap); eval every 250 to trim wall time
fi
# emit "--max_steps N" only when N>0 (HF default -1 => use epochs)
maxsteps() { [[ "${1:-0}" -gt 0 ]] && echo "--max_steps $1" || true; }

# ---- preflight --------------------------------------------------------------
echo "=== preflight ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available())"
# repo must have the classification patches
python -c "import entexbert2.run_experiment as R; assert 'as_class' in R.LABEL_BUILDERS, 'as_class label missing -- push the classification patches'; print('  repo OK: as_class label present')"
grep -q "task == \"classification\"" "$REPO/src/entexbert2/model_io.py" || { echo "FATAL: model_io.py lacks the classification inference branch -- push the patch"; exit 1; }
for f in "$MODEL" "$REF" "$HETSNV" "$ADASTRA" "$REFCSV" "$STAGE1_CKPT"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }; echo "  found: $f"
done

# ---- STEP 3c: counts (reuse) + held-out chroms + s_hat ----------------------
echo ""; echo "=== STEP 3c: counts + held-out chroms + s_hat ==="
if [[ ! -f "$COUNTS" ]]; then
  echo "[STEP 3c] counts absent -> building from hetSNVs"
  python build_betabinom_counts.py --hetsnv_tsv "$HETSNV" --assay "$ASSAY" --min_total_reads 20 --out "$COUNTS"
else
  echo "[STEP 3c] reusing existing counts: $COUNTS"
fi
SEL=$WORK/ctcf_heldout_clf
python select_heldout_chroms.py --eval_csv "$COUNTS" --label_col imbalance_significance \
    --target 0.10 --top_train_spare 2 --max_set 2 --ci_floor 100 \
    --special chrX chrY chrM chr1 --out_prefix "$SEL" | tee "$WORK/heldout_select_clf.log"
HELDOUT=$(grep -m1 "HELD-OUT SET:" "$WORK/heldout_select_clf.log" | sed -E 's/.*HELD-OUT SET: *([^ ]+).*/\1/' | tr '+' ',')
echo "[STEP 3c] held-out chroms = $HELDOUT"
[[ -n "$HELDOUT" ]] || { echo "FATAL: no held-out chroms parsed"; exit 1; }
FOLD=$(python - <<PY
chroms="$HELDOUT".split(",")
print("{" + ", ".join(f"{c}: 0" for c in chroms if c) + "}")
PY
)
SHAT=$(python - <<'PY'
import os, pandas as pd, numpy as np, sys
df=pd.read_csv(os.environ["COUNTS"])
k=df["k"].to_numpy(float); n=df["n"].to_numpy(float)
pbar=k.sum()/n.sum()
num=np.sum((k-n*pbar)**2/(n*pbar*(1-pbar)))
N=len(n); nbar=n.mean()
rho=(num-(N-1))/(np.sum(n)-nbar); rho=min(max(rho,1e-3),0.99)
s=(1-rho)/rho
print(f"{s:.1f}")
print(f"[s_hat] rho={rho:.4f} -> s_hat={s:.1f} (n loci={N}, mean n={nbar:.1f})", file=sys.stderr)
PY
)
if [[ -n "$DIAG_SGRID" ]]; then SGRID=$(echo "$DIAG_SGRID" | sed "s/s_hat/$SHAT/"); else SGRID="$SHAT 20 50 100"; fi
echo "[STEP 3c] s_hat=$SHAT | s-sweep = $SGRID"

# ---- STEP 4c: build Stage-2 CLASSIFICATION dataset --------------------------
echo ""; echo "=== STEP 4c: build Stage-2 classification dataset ==="
cat > "$WORK/stage2_clf.yaml" <<YAML
experiment: ctcf_stage2_asb_clf_enc002
ref_fasta: $REF
output_dir: $BUILD
row_source: {type: betabinom_counts, path: $COUNTS, assay: $ASSAY, min_total_reads: 20}
primary_label: {type: as_class, column: imbalance_significance, name: as_class}
sequence: {input_mode: hap_pair}
window: {left_bp: 128, right_bp: 128, snv_offset_mode: fixed, jitter_max_bp: 0}
head: {task: classification, head_num_layers: 2, head_hidden_size: 128, proj_dim: 128}
depth_col: n
split: {mode: train_dev_test, ratio: [0.8, 0.1, 0.1], seed: 42, group: locus}
partition:
  enabled: true
  bin_size: 100000
  fold_id: 0
  fold_assignment: $FOLD
  exclude_boundary: true
  boundary_bp: 128
balance: {strategy: none}
YAML
python -m entexbert2.run_experiment "$WORK/stage2_clf.yaml" --ref_fasta "$REF" --output_dir "$BUILD"
python - <<PY
import pandas as pd
d=pd.read_csv("$BUILD/train.csv")
vc=d["label"].value_counts().sort_index().to_dict()
frac=float((d["label"]>0.5).mean())
print(f"[STEP 4c] train.csv cols: {list(d.columns)} | rows: {len(d)}")
print(f"[STEP 4c] label counts: {vc} | positive fraction: {frac:.4f}")
assert set(d["label"].unique()) <= {0,1,0.0,1.0}, f"as_class label is not binary: {d['label'].unique()[:8]}"
assert 0 < frac < 1, "degenerate label (all one class)"
PY
echo "[STEP 4c] OK"

# ---- STEP 5c: Stage-2a frozen-trunk contrast head (s sweep, pick best dev AUROC) ----
echo ""; echo "=== STEP 5c: Stage-2a frozen contrast head (s sweep) ==="
BEST_2A=""; BEST_S=""; BEST_AU="-1"
for S in $SGRID; do
  RUN=$WORK/runs/clf_2a_s${S}
  echo "--- Stage-2a s=$S -> $RUN ---"
  python -m entexbert2.finetune_entexbert2 \
      --model_name_or_path "$MODEL" \
      --data_path "$BUILD" \
      --task classification --input_mode hap_pair \
      --pooling_mode center_mean --center_pool_width 5 \
      --head_num_layers 2 --head_hidden_size 128 --proj_dim 128 \
      --balanced_sampler True --neff_s "$S" \
      --init_backbone_from "$STAGE1_CKPT" --freeze_backbone True \
      --model_max_length 512 \
      --per_device_train_batch_size 32 --per_device_eval_batch_size 64 \
      --num_train_epochs 3 $(maxsteps "$S2A_MAX_STEPS") \
      --learning_rate 1e-3 --warmup_steps 50 --weight_decay 0.01 \
      --logging_steps 20 --eval_steps "$EVAL_EVERY" --save_steps "$EVAL_EVERY" --save_total_limit 2 \
      --evaluation_strategy steps --load_best_model_at_end True \
      --metric_for_best_model auroc --greater_is_better True \
      --bf16 --save_model True --eval_and_save_results True \
      --output_dir "$RUN" --run_name "clf_2a_s${S}"
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
  echo "    dev AUROC = $AU"
  awk "BEGIN{exit !($AU > $BEST_AU)}" && { BEST_AU="$AU"; BEST_2A="$RUN"; BEST_S="$S"; }
done
echo "[STEP 5c] BEST 2a: s=$BEST_S dev_AUROC=$BEST_AU ckpt=$BEST_2A"
[[ -n "$BEST_2A" ]] || { echo "FATAL: no 2a run succeeded"; exit 1; }
echo "[STEP 5c] OK"

# ---- STEP 6c: Stage-2b gentle unfreeze from best-2a -------------------------
echo ""; echo "=== STEP 6c: Stage-2b gentle unfreeze (s=$BEST_S) ==="
RUN2B=$WORK/runs/clf_2b_s${BEST_S}
python -m entexbert2.finetune_entexbert2 \
    --model_name_or_path "$MODEL" \
    --data_path "$BUILD" \
    --task classification --input_mode hap_pair \
    --pooling_mode center_mean --center_pool_width 5 \
    --head_num_layers 2 --head_hidden_size 128 --proj_dim 128 \
    --balanced_sampler True --neff_s "$BEST_S" \
    --init_backbone_from "$BEST_2A/pytorch_model.bin" \
    --model_max_length 512 \
    --per_device_train_batch_size 16 --per_device_eval_batch_size 64 \
    --num_train_epochs 2 $(maxsteps "$S2B_MAX_STEPS") \
    --learning_rate 3e-6 --warmup_steps 50 --weight_decay 0.01 \
    --logging_steps 20 --eval_steps "$EVAL_EVERY" --save_steps "$EVAL_EVERY" --save_total_limit 2 \
    --evaluation_strategy steps --load_best_model_at_end True \
    --metric_for_best_model auroc --greater_is_better True \
    --bf16 --save_model True --eval_and_save_results True \
    --output_dir "$RUN2B" --run_name "clf_2b_s${BEST_S}"
# promote 2b only if it beats 2a on dev AUROC (else keep 2a) -- 2b hurt in the DIAG regression run.
AU2B=$(python - <<PY
import json,glob,os
st=os.path.join("$RUN2B","trainer_state.json")
val=json.load(open(st)).get("best_metric") if os.path.exists(st) else None
if val is None:
    er=glob.glob("$RUN2B/results/*/eval_results.json")
    if er: val=json.load(open(er[0])).get("eval_auroc")
print(val if val is not None else -1)
PY
)
echo "[STEP 6c] 2b dev AUROC=$AU2B  vs  best-2a dev AUROC=$BEST_AU"
if awk "BEGIN{exit !($AU2B > $BEST_AU)}"; then
  echo "[STEP 6c] 2b improves on 2a -> will score 2b too"; SCORE_2B=1
else
  echo "[STEP 6c] 2b did NOT beat 2a on dev -> scoring 2a only (2b kept on disk)"; SCORE_2B=0
fi
echo "[STEP 6c] OK"

# ---- STEP 7c: score (leak-free) ---------------------------------------------
echo ""; echo "=== STEP 7c: score (leak-free) ==="
TAGS=("2a:$BEST_2A")
[[ "$SCORE_2B" == "1" ]] && TAGS+=("2b:$RUN2B")
for TAG in "${TAGS[@]}"; do
  NAME=${TAG%%:*}; CKPT=${TAG#*:}
  echo "--- scoring $NAME ($CKPT) ---"
  python -m entexbert2.score_asb \
      --eval adastra --checkpoint_dir "$CKPT" --ref_fasta "$REF" \
      --eval_csv "$ADASTRA" --assay "$ASSAY" ${REFCSV:+--reference_csv "$REFCSV"} \
      --train_coords "$BUILD/train.meta.csv" "$BUILD/dev.meta.csv" --drop_leaky \
      --left_bp 128 --right_bp 128 --batch_size 64 --device cuda \
      --out "$WORK/score_clf_${NAME}_adastra"
  python -m entexbert2.score_asb \
      --eval hetsnv --checkpoint_dir "$CKPT" --ref_fasta "$REF" \
      --hetsnv_tsv "$HETSNV" --assay "$ASSAY" \
      --train_coords "$BUILD/train.meta.csv" "$BUILD/dev.meta.csv" --drop_leaky \
      --left_bp 128 --right_bp 128 --batch_size 64 --device cuda \
      --out "$WORK/score_clf_${NAME}_hetsnv"
done
echo "[STEP 7c] OK"

echo ""
echo "============================================================"
echo " CTCF STAGE-2 CLASSIFICATION COMPLETE.  (DIAG=$DIAG)"
echo "   best 2a: s=$BEST_S (dev AUROC $BEST_AU) -> $BEST_2A"
echo "   2b:      dev AUROC $AU2B  (scored: $SCORE_2B)"
echo "   outputs: $WORK/score_clf_2a_*  (and score_clf_2b_* if 2b won)"
echo " COMPARISON: regression-twin on the SAME CTCF trunk was leak-free 0.6652 (2a) / 0.6748 (2b)."
echo "   Beating that => the contrast-classification head extracts more ASB signal than the signed twin."
echo "   Benchmark context: best sequence-only DNNs on CTCF ~0.73-0.77 (Sei/DeepSEA/Enformer)."
[[ "$DIAG" == "1" ]] && echo " NOTE: DIAG mode -> AUROC is UNDER-TRAINED, not final. Rerun with DIAG=0 for the real number."
echo "============================================================"
