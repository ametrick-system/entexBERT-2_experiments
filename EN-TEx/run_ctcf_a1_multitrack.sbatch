#!/bin/bash
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=rtx3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=eb2_ctcf_a1_multitrack
#SBATCH --output=logs/eb2_ctcf_a1_multitrack_%j.log
#SBATCH --error=logs/eb2_ctcf_a1_multitrack_%j.err
#SBATCH --mail-type=END,FAIL
# =============================================================================
# LEVER A1 — MULTI-TRACK Stage-1 supervision (CTCF, ENC-002).  Predict each of the 18 EN-TEx
# tissues' fold-change as its OWN output track (masked multi-output regression), instead of the
# mean fold-change the single-track binding trunk uses. Then run the SAME Stage-2 ASB contrast
# head on top and score leak-free ADASTRA. Controlled A/B:
#
#   single-track trunk -> Stage-2 ASB : 0.7208  (verified baseline)
#   A1 multi-track trunk -> Stage-2 ASB : the A1 number  (this run, arm "a1")
#   NO-DAPT (pretrained trunk, no Stage-1) -> Stage-2 ASB : the control (this run, arm "nodapt")
#
# The nodapt arm is the user's requested control: does Stage-1 DAPT earn its keep for the contrast
# head at all? It is the SAME Stage-2 recipe with --init_backbone_from OMITTED (so the trunk stays
# at pretrained DNABERT-2 weights, trunk unfrozen). Everything else — data build, split, head,
# eval — is held identical across arms so the only moving parts are (i) multi-track vs mean Stage-1
# and (ii) DAPT vs no-DAPT.
#
#   STEP 1   build Stage-1 MULTI-TRACK binding dataset (ref_single, 18 y_track/m_track cols)
#   STEP 2   train Stage-1 multi-track trunk (recipe IDENTICAL to the single-track binding trunk,
#            + --num_labels 18); metric_for_best_model = eval_spearman (mean per-track)
#   STEP 3   build Stage-2 ASB classification dataset (hap_pair, as_class) — reuse baseline split
#   STEP 4   Stage-2a frozen contrast head, s-sweep, pick best dev AUROC — for BOTH arms
#   STEP 5   score leak-free ADASTRA + hetSNV for both arms
#
# DIAG=0 (default): full run. DIAG=1: capped steps + 2-value s-sweep for a fast end-to-end check.
#   Submit the diagnostic first:  sbatch --export=ALL,DIAG=1 run_ctcf_a1_multitrack.sbatch
#   (env-var-via-shell-prefix does NOT propagate into SLURM; use --export=ALL,DIAG=1.)
#
# PREREQS (fail-fast in preflight): PATCHED repo with the multitrack support:
#   build_inputs.py (y_track/m_track emission), run_experiment.py (multitrack label + head width),
#   model.py (num_labels head + masked loss), finetune_entexbert2.py (multitrack dataset/metrics).
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

# A1 experiment root (separate from the single-track baseline so nothing is clobbered)
export A1ROOT=$HOME/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/ctcf_a1_multitrack
mkdir -p "$A1ROOT"
# Stage-1 multi-track config (mirror of the single-track binding config; only label+head differ).
# EDIT this path if you keep configs elsewhere; it must be the ctcf_binding_regression_multitrack_enc002.yaml.
export MT_CONFIG=/home/asm242/palmer_scratch/entexBERT-2_experiments/EN-TEx/configs/ctcf_binding_regression_multitrack_enc002.yaml
export MT_BUILD=$A1ROOT/stage1_inputs           # Stage-1 multi-track train/dev/test
export MT_TRUNK_DIR=$A1ROOT/stage1_runs/reg     # Stage-1 multi-track trunk output
export MT_TRUNK=$MT_TRUNK_DIR/pytorch_model.bin
mkdir -p "$MT_BUILD" "$MT_TRUNK_DIR"            # FIX: tee opens train.log BEFORE the trainer makes
                                                # --output_dir; create it up front or the pipe fails
                                                # under `set -o pipefail` and aborts the whole script.

# Stage-2 ASB build: REUSE the exact baseline split so the A/B differs only in the trunk.
export BASE_WORK=$HOME/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/stage2_ctcf_clf
export COUNTS=$BASE_WORK/ctcf_betabinom_counts.csv
export BUILD=$BASE_WORK/inputs_clf
cd "$REPO"

# ---- diagnostic vs full -----------------------------------------------------
DIAG="${DIAG:-0}"
if [[ "$DIAG" == "1" ]]; then
  echo "*** DIAGNOSTIC MODE: capped steps + 2-value s-sweep. NOT a final result. ***"
  S1_MAX_STEPS=400; S2A_MAX_STEPS=400; EVAL_EVERY=100; DIAG_SGRID="s_hat 20"
else
  echo "*** FULL RUN: Stage-1 3-epoch multi-track + Stage-2a 3-epoch s-sweep {s_hat,20,50,100}. ***"
  S1_MAX_STEPS=0;   S2A_MAX_STEPS=0;   EVAL_EVERY=250; DIAG_SGRID=""
fi
maxsteps() { [[ "${1:-0}" -gt 0 ]] && echo "--max_steps $1" || true; }

# ---- preflight --------------------------------------------------------------
echo "=== preflight ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available())"
# repo must carry the multitrack patches (fail fast on a stale checkout)
python -c "import entexbert2.run_experiment as R; assert 'multitrack' in R.LABEL_BUILDERS, 'multitrack label missing -- push the A1 patches'; print('  repo OK: multitrack label present')"
python -c "import inspect, entexbert2.model as M; src=inspect.getsource(M.entexBERT2ForSequencePrediction.__init__); assert 'num_labels' in src, 'model.py lacks num_labels head -- push the A1 model patch'; print('  repo OK: num_labels head present')"
grep -q "y_track_" "$REPO/src/entexbert2/build_inputs.py" || { echo "FATAL: build_inputs.py lacks y_track_ emission -- apply the A1 build diff"; exit 1; }
grep -q "self.multitrack" "$REPO/src/entexbert2/finetune_entexbert2.py" || { echo "FATAL: finetune lacks multitrack dataset -- apply the A1 finetune diff"; exit 1; }
grep -q "as_class" <(python -c "import entexbert2.run_experiment as R; print(' '.join(R.LABEL_BUILDERS))") || { echo "FATAL: as_class label missing (Stage-2 needs it)"; exit 1; }
[ -f "$MT_CONFIG" ] || { echo "FATAL: missing Stage-1 multitrack config $MT_CONFIG"; exit 1; }
for f in "$MODEL" "$REF" "$HETSNV" "$ADASTRA" "$REFCSV"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }; echo "  found: $f"
done

# =============================================================================
# STEP 1 — build Stage-1 MULTI-TRACK binding dataset
# =============================================================================
echo ""; echo "=== STEP 1: build Stage-1 multi-track binding dataset ==="
if [[ -f "$MT_BUILD/train.csv" && -f "$MT_BUILD/tracks.json" ]]; then
  echo "[STEP 1] reusing existing multi-track build at $MT_BUILD"
else
  python -m entexbert2.run_experiment "$MT_CONFIG" --ref_fasta "$REF" --output_dir "$MT_BUILD"
fi
# sanity: T=18 y_track/m_track columns present and tracks.json names 18 tissues
python - <<PY
import pandas as pd, json, os
d = pd.read_csv(os.path.join("$MT_BUILD","train.csv"), nrows=50)
y = sorted([c for c in d.columns if c.startswith("y_track_")], key=lambda c:int(c.split("_")[-1]))
m = sorted([c for c in d.columns if c.startswith("m_track_")], key=lambda c:int(c.split("_")[-1]))
print(f"[STEP 1] y_track cols: {len(y)} | m_track cols: {len(m)} | has 'sequence': {'sequence' in d.columns}")
assert len(y)==18 and len(m)==18, f"expected 18 tracks, got y={len(y)} m={len(m)}"
tj = json.load(open(os.path.join("$MT_BUILD","tracks.json")))
print(f"[STEP 1] tracks.json num_tracks: {tj['num_tracks']} | first tissue: {tj['tracks'][0]['tissue']}")
assert tj["num_tracks"]==18
PY
echo "[STEP 1] OK"

# =============================================================================
# STEP 2 — train Stage-1 MULTI-TRACK trunk (recipe identical to single-track binding trunk)
# =============================================================================
echo ""; echo "=== STEP 2: train Stage-1 multi-track trunk (num_labels=18) ==="
if [[ -f "$MT_TRUNK" ]]; then
  echo "[STEP 2] reusing existing multi-track trunk: $MT_TRUNK"
else
  python -m entexbert2.finetune_entexbert2 \
      --model_name_or_path "$MODEL" \
      --data_path "$MT_BUILD" \
      --input_mode ref_single \
      --task regression \
      --num_labels 18 \
      --pooling_mode center_mean \
      --center_pool_width 5 \
      --head_num_layers 1 \
      --model_max_length 512 \
      --per_device_train_batch_size 16 \
      --gradient_accumulation_steps 8 \
      --per_device_eval_batch_size 32 \
      --num_train_epochs 3 $(maxsteps "$S1_MAX_STEPS") \
      --learning_rate 3e-5 \
      --weight_decay 0.01 \
      --warmup_steps 50 \
      --logging_steps 50 --eval_steps "$EVAL_EVERY" --save_steps "$EVAL_EVERY" \
      --evaluation_strategy steps --save_total_limit 2 \
      --load_best_model_at_end True \
      --metric_for_best_model eval_spearman --greater_is_better True \
      --bf16 True --save_model True --eval_and_save_results True \
      --run_name "ctcf_a1_multitrack_stage1" \
      --output_dir "$MT_TRUNK_DIR" 2>&1 | tee "$MT_TRUNK_DIR/train.log"
fi
[ -f "$MT_TRUNK" ] || { echo "FATAL: Stage-1 multi-track trunk not produced at $MT_TRUNK"; exit 1; }
echo "[STEP 2] OK -> $MT_TRUNK"

# =============================================================================
# STEP 3 — Stage-2 ASB dataset (reuse baseline split) + held-out fold + s_hat
# =============================================================================
echo ""; echo "=== STEP 3: Stage-2 ASB dataset (reuse baseline) + s_hat ==="
[[ -f "$COUNTS" ]] || { echo "[STEP 3] counts absent -> building"; python build_betabinom_counts.py --hetsnv_tsv "$HETSNV" --assay "$ASSAY" --min_total_reads 20 --out "$COUNTS"; }
if [[ -f "$BUILD/train.csv" && -f "$BUILD/test.csv" ]]; then
  echo "[STEP 3] reusing baseline Stage-2 split at $BUILD (identical data for the A/B)"
else
  echo "FATAL: baseline Stage-2 build $BUILD/train.csv not found -- run the single-track pipeline first"; exit 1
fi
SHAT=$(python - <<'PY'
import os, pandas as pd, numpy as np, sys
df=pd.read_csv(os.environ["COUNTS"]); k=df["k"].to_numpy(float); n=df["n"].to_numpy(float)
pbar=k.sum()/n.sum(); num=np.sum((k-n*pbar)**2/(n*pbar*(1-pbar)))
N=len(n); nbar=n.mean(); rho=(num-(N-1))/(np.sum(n)-nbar); rho=min(max(rho,1e-3),0.99); s=(1-rho)/rho
print(f"{s:.1f}")
PY
)
if [[ -n "$DIAG_SGRID" ]]; then SGRID=$(echo "$DIAG_SGRID" | sed "s/s_hat/$SHAT/"); else SGRID="$SHAT 20 50 100"; fi
echo "[STEP 3] s_hat=$SHAT | s-sweep = $SGRID"

# =============================================================================
# STEP 4 — Stage-2a frozen contrast head, for BOTH arms (a1 trunk, nodapt pretrained)
# =============================================================================
# arm "a1"     : --init_backbone_from $MT_TRUNK  (loads the multi-track trunk backbone), frozen
# arm "nodapt" : NO --init_backbone_from         (pretrained DNABERT-2 backbone), UNFROZEN so the
#                trunk can at least adapt (a frozen pretrained trunk was the ~0.5 frozen-probe rung).
run_arm () {
  local ARM="$1"; local INIT_FLAGS="$2"; local FREEZE="$3"
  echo ""; echo "=== STEP 4[$ARM]: Stage-2a s-sweep ($ARM) ==="
  local BEST=""; local BEST_S=""; local BEST_AU="-1"
  for S in $SGRID; do
    local RUN=$A1ROOT/stage2_$ARM/runs/clf_2a_s${S}
    echo "--- [$ARM] Stage-2a s=$S -> $RUN ---"
    python -m entexbert2.finetune_entexbert2 \
        --model_name_or_path "$MODEL" \
        --data_path "$BUILD" \
        --task classification --input_mode hap_pair \
        --pooling_mode center_mean --center_pool_width 5 \
        --head_num_layers 2 --head_hidden_size 128 --proj_dim 128 \
        --balanced_sampler True --neff_s "$S" \
        $INIT_FLAGS $FREEZE \
        --model_max_length 512 \
        --per_device_train_batch_size 32 --per_device_eval_batch_size 64 \
        --num_train_epochs 3 $(maxsteps "$S2A_MAX_STEPS") \
        --learning_rate 1e-3 --warmup_steps 50 --weight_decay 0.01 \
        --logging_steps 20 --eval_steps "$EVAL_EVERY" --save_steps "$EVAL_EVERY" --save_total_limit 2 \
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
    echo "    [$ARM] dev AUROC = $AU"
    awk "BEGIN{exit !($AU > $BEST_AU)}" && { BEST_AU="$AU"; BEST="$RUN"; BEST_S="$S"; }
  done
  echo "[STEP 4[$ARM]] BEST 2a: s=$BEST_S dev_AUROC=$BEST_AU ckpt=$BEST"
  [[ -n "$BEST" ]] || { echo "FATAL: no 2a run succeeded for arm $ARM"; exit 1; }
  # ---- score this arm leak-free ----
  echo "=== STEP 5[$ARM]: score leak-free ($ARM, best 2a) ==="
  python -m entexbert2.score_asb \
      --eval adastra --checkpoint_dir "$BEST" --ref_fasta "$REF" \
      --eval_csv "$ADASTRA" --assay "$ASSAY" ${REFCSV:+--reference_csv "$REFCSV"} \
      --train_coords "$BUILD/train.meta.csv" "$BUILD/dev.meta.csv" --drop_leaky \
      --left_bp 128 --right_bp 128 --batch_size 64 --device cuda \
      --out "$A1ROOT/score_${ARM}_adastra"
  python -m entexbert2.score_asb \
      --eval hetsnv --checkpoint_dir "$BEST" --ref_fasta "$REF" \
      --hetsnv_tsv "$HETSNV" --assay "$ASSAY" \
      --train_coords "$BUILD/train.meta.csv" "$BUILD/dev.meta.csv" --drop_leaky \
      --left_bp 128 --right_bp 128 --batch_size 64 --device cuda \
      --out "$A1ROOT/score_${ARM}_hetsnv"
  echo "[STEP 5[$ARM]] scored -> $A1ROOT/score_${ARM}_adastra*"
}

# arm 1: A1 multi-track trunk (frozen, exactly like the 0.7208 baseline Stage-2a)
run_arm "a1"     "--init_backbone_from $MT_TRUNK" "--freeze_backbone True"
# arm 2: no-DAPT control (pretrained backbone, unfrozen so it can adapt during Stage-2)
run_arm "nodapt" "" ""

echo ""
echo "============================================================"
echo " CTCF A1 MULTI-TRACK COMPLETE.  (DIAG=$DIAG)"
echo "   Stage-1 multi-track trunk: $MT_TRUNK"
echo "   arm a1     -> $A1ROOT/score_a1_adastra*"
echo "   arm nodapt -> $A1ROOT/score_nodapt_adastra*"
echo " A/B TARGET: single-track trunk -> Stage-2 ASB was leak-free AUROC 0.7208 [0.7031,0.7380]."
echo "   a1 > 0.7208 (non-overlapping CI) => per-track supervision helps."
echo "   a1 ~ nodapt                      => DAPT is not earning its keep for the contrast head."
[[ "$DIAG" == "1" ]] && echo " NOTE: DIAG mode -> UNDER-TRAINED, not final. Rerun with DIAG=0 for the real numbers."
echo "============================================================"
