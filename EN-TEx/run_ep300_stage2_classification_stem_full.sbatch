#!/bin/bash
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=rtx3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --job-name=eb2_ep300_clf_stem_full
#SBATCH --output=logs/eb2_ep300_clf_stem_full_%j.log
#SBATCH --error=logs/eb2_ep300_clf_stem_full_%j.err
#SBATCH --mail-type=END,FAIL
# =============================================================================
# EP300 Stage-2 CLASSIFICATION (contrast head).  s = ||P(h1) - P(h2)||, p = sigmoid(a*s + b),
# precision-weighted BCE on the binary AS label. REUSES the already-trained Stage-1 binding
# trunk and the beta-binomial counts from the regression run -- only Stage-2 is re-done.
#
#   STEP 4c  build Stage-2 CLASSIFICATION dataset (hap_pair, as_class label, held-out chroms -> TEST)
#   STEP 5c  Stage-2a: FROZEN trunk, contrast head (proj_dim d), balanced sampler; sweep s; pick best by dev AUROC
#   STEP 6c  Stage-2b: gentle unfreeze from best-2a (low LR)
#   STEP 7c  score ADASTRA + hetSNV, leak-free, for best-2a and (if it beats 2a on dev) 2b
#
# PREREQS (fail-fast in preflight):
#   * trained Stage-1 trunk  $STAGE1_CKPT
#   * beta-binomial counts   $COUNTS         (rebuilt from hetSNVs if absent)
#   * ADASTRA eval set + EP300 reference AUROC
#   * PATCHED repo: model.py (contrast head), model_io.py (task-aware inference),
#     run_experiment.py (as_class label), finetune_entexbert2.py (classification + balanced_sampler),
#     score_asb.py (task-aware ranking).  Preflight greps for the as_class builder to catch a stale repo.
#
# *** CNN-STEM EP300 FULL RUN (DIAG hardcoded 0 below -- the real number) ***
# 3-epoch 2a over full s-sweep {s_hat,20,50,100}, no step cap, --time 20h. Reuses the EP300 trunk +
# counts + build (clean ablation); writes runs/scores under runs_stem_full/ + score_stem_full/ so it
# collides with NEITHER the no-stem EP300 diagnostic (runs/ + score_clf_*) NOR the stem diagnostic
# (runs_stem/ + score_stem/).
# Submit plainly:  sbatch run_ep300_stage2_classification_stem_full.sbatch
# WIN CONDITION: leak-free classification AUROC > ~0.596 (Enformer, best DNN on EP300).
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
ASSAY=EP300
export ADASTRA=$HOME/palmer_scratch/entexBERT-2_experiments/ADASTRA/ep300_adastra_evalset.csv.gz
export REFCSV=$HOME/palmer_scratch/entexBERT-2_experiments/ADASTRA/ep300_benchmark_reference_auroc.csv
export WORK=$HOME/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/ep300_asb
export STAGE1_CKPT=$WORK/stage1/runs/reg/pytorch_model.bin
export COUNTS=$WORK/ep300_betabinom_counts.csv
export BUILD=$WORK/inputs_clf              # SEPARATE from the regression inputs (different label)
# STEM ABLATION: reuse the EP300 trunk + counts + build (clean ablation, identical data), but write
# runs and scores under a stem-tagged root so the no-stem EP300 diagnostic baseline is untouched.
export RUNROOT=$WORK/runs_stem_full
export SCOREROOT=$WORK/score_stem_full
mkdir -p "$RUNROOT" "$SCOREROOT"
cd "$REPO"

# ---- diagnostic vs full -----------------------------------------------------
DIAG=0   # HARDCODED FULL run: 3-epoch 2a over full s-sweep {s_hat,20,50,100}, no step cap. The real number.
if [[ "$DIAG" == "1" ]]; then
  echo "*** DIAGNOSTIC MODE: capped steps + 2-value s-sweep. NOT a final result. ***"
  S2A_MAX_STEPS=400; S2B_MAX_STEPS=300; EVAL_EVERY=100; DIAG_SGRID="s_hat 20"
else
  S2A_MAX_STEPS=0;   S2B_MAX_STEPS=0;   EVAL_EVERY=100; DIAG_SGRID=""
fi
# emit "--max_steps N" only when N>0 (HF default -1 => use epochs)
maxsteps() { [[ "${1:-0}" -gt 0 ]] && echo "--max_steps $1" || true; }

# ---- preflight --------------------------------------------------------------
echo "=== preflight ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available())"
# repo must have the classification patches
python -c "import entexbert2.run_experiment as R; assert 'as_class' in R.LABEL_BUILDERS, 'as_class label missing -- push the classification patches'; print('  repo OK: as_class label present')"
python -c "from entexbert2.model import OneHotCNNStem, one_hot_encode; print('  repo OK: CNN stem present in model.py')" \
  || { echo "FATAL: model.py lacks OneHotCNNStem/one_hot_encode -- push the stem patch (model.py v5)"; exit 1; }
grep -q "use_cnn_stem" "$REPO/src/entexbert2/finetune_entexbert2.py" \
  || { echo "FATAL: finetune_entexbert2.py lacks --use_cnn_stem -- apply finetune_cnn_stem_diff.md"; exit 1; }
grep -q "def _batch_onehot" "$REPO/src/entexbert2/model_io.py" \
  || { echo "FATAL: model_io.py lacks the stem inference plumbing -- push model_io.py v5"; exit 1; }
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
SEL=$WORK/ep300_heldout_clf
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
experiment: ep300_stage2_asb_clf_enc001
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
if [[ -f "$BUILD/train.csv" && -f "$BUILD/test.csv" ]]; then
  echo "[STEP 4c] reusing existing baseline split at $BUILD (clean ablation -> identical data as no-stem EP300)"
else
  echo "[STEP 4c] building dataset at $BUILD"
  python -m entexbert2.run_experiment "$WORK/stage2_clf.yaml" --ref_fasta "$REF" --output_dir "$BUILD"
fi
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
  RUN=$RUNROOT/clf_2a_s${S}
  echo "--- Stage-2a s=$S -> $RUN ---"
  python -m entexbert2.finetune_entexbert2 \
      --model_name_or_path "$MODEL" \
      --data_path "$BUILD" \
      --task classification --input_mode hap_pair \
      --pooling_mode center_mean --center_pool_width 5 \
      --head_num_layers 2 --head_hidden_size 128 --proj_dim 128 \
      --use_cnn_stem True --cnn_channels 64 --cnn_out_dim 64 \
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
# SKIPPED BY DEFAULT (RUN_2B=0): the no-stem CTCF full run showed 2b (unfreeze) HURTS this
# distance head (leak-free 0.7053 < 2a 0.7208). Set RUN_2B=1 to re-enable if you want to check it.
RUN_2B="${RUN_2B:-0}"
RUN2B=$RUNROOT/clf_2b_s${BEST_S}
if [[ "$RUN_2B" != "1" ]]; then
  echo ""; echo "=== STEP 6c: SKIPPED (RUN_2B=0; 2b degraded the distance head). Scoring 2a only. ==="
  AU2B="-1"; SCORE_2B=0
else
echo ""; echo "=== STEP 6c: Stage-2b gentle unfreeze (s=$BEST_S) ==="
python -m entexbert2.finetune_entexbert2 \
    --model_name_or_path "$MODEL" \
    --data_path "$BUILD" \
    --task classification --input_mode hap_pair \
    --pooling_mode center_mean --center_pool_width 5 \
    --head_num_layers 2 --head_hidden_size 128 --proj_dim 128 \
    --use_cnn_stem True --cnn_channels 64 --cnn_out_dim 64 \
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
      --out "$SCOREROOT/score_clf_${NAME}_adastra"
  python -m entexbert2.score_asb \
      --eval hetsnv --checkpoint_dir "$CKPT" --ref_fasta "$REF" \
      --hetsnv_tsv "$HETSNV" --assay "$ASSAY" \
      --train_coords "$BUILD/train.meta.csv" "$BUILD/dev.meta.csv" --drop_leaky \
      --left_bp 128 --right_bp 128 --batch_size 64 --device cuda \
      --out "$SCOREROOT/score_clf_${NAME}_hetsnv"
done
echo "[STEP 7c] OK"

echo ""
echo "============================================================"
echo " EP300 STAGE-2 CLASSIFICATION + CNN STEM COMPLETE.  (DIAG=$DIAG)"
echo "   best 2a: s=$BEST_S (dev AUROC $BEST_AU) -> $BEST_2A"
echo "   2b:      dev AUROC $AU2B  (scored: $SCORE_2B)"
echo "   outputs: $SCOREROOT/score_clf_2a_*  (and score_clf_2b_* if RUN_2B=1 and it won)"
echo " WIN CONDITION: leak-free classification AUROC > ~0.596 (Enformer, best DNN on EP300)."
echo " STEM RATIONALE: EP300 is motif-less -> BPE has no stable local grammar, so base-resolution"
echo "   input should help MORE here than on CTCF (where the stem added only +0.006, n.s.)."
echo "   Compare this stem number to the no-stem EP300 diagnostic (runs/ + score_clf_*)."
[[ "$DIAG" == "1" ]] && echo " NOTE: DIAG mode -> AUROC is UNDER-TRAINED, not final. Rerun full (DIAG=0) for the real number."
echo "============================================================"
