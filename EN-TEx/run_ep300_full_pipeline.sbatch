#!/bin/bash
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=rtx3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=eb2_ep300_diag
#SBATCH --output=logs/eb2_ep300_diag_%j.log
#SBATCH --error=logs/eb2_ep300_diag_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# entexBERT-2 FULL EP300 PIPELINE (individual 1, donor ENCDO845WKR)
# -----------------------------------------------------------------------------
# End-to-end: input generation -> Stage-1 binding trunk -> Stage-2 ASB (2a+2b) -> scoring.
# EP300 = motif-less coactivator: on the Han benchmark the PWM/motif methods CANNOT score it,
# and the best DNN (Enformer) is only 0.596 -- so beating ~0.60 leak-free is the win condition.
#
#   STEP 1  build Stage-1 multi-tissue binding-regression inputs (5 EP300 tissues, consensus merge)
#   STEP 2  Stage-1: train the binding trunk (regression on mean fold-change) -> the ~0.68-analogue trunk
#   STEP 3  build EP300 beta-binomial counts + SELECT held-out chroms + estimate s-hat
#   STEP 4  build Stage-2 ASB dataset (hap_pair, logit-ratio label, held-out chroms -> TEST)
#   STEP 5  Stage-2a: FROZEN trunk, linear head, sweep s; pick best by dev spearman
#   STEP 6  Stage-2b: gentle unfreeze from best 2a (low LR)
#   STEP 7  score ADASTRA + hetSNV, leak-free, for best-2a and 2b -> benchmark placement
#
# TWO PREREQUISITES ARE CHECKED IN PREFLIGHT AND HALT THE JOB WITH A CLEAR MESSAGE:
#   (A) the 5 EP300 tissue files exist under $ENTEX_EP300  (from download_entex_chipseq.sh)
#   (B) hetSNVs.tsv actually CONTAINS EP300 rows          (Stage-2 has no data otherwise)
# =============================================================================

set -euo pipefail
mkdir -p logs

module purge
module load StdEnv
module load miniconda
conda activate eb2

# ---- paths (EDIT if yours differ) -------------------------------------------
REPO=$HOME/entexBERT-2
export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
MODEL=$REPO/DNABERT-2-117M-attention
REF=$HOME/reference_genome/hg38.fa
CHROMSIZES=$HOME/reference_genome/hg38.chrom.sizes
export HETSNV=$HOME/palmer_scratch/entex_data/hetSNVs.tsv
ASSAY=EP300                       # hetSNVs.tsv assay label (STEP-B check verifies this exists)
DONOR=ENC-001

# EP300 ChIP-seq tissue files (downloaded via download_entex_chipseq.sh <manifest> ep300_1 $ENTEX_EP300)
ENTEX_EP300=$HOME/palmer_scratch/entex_data/EP300/ENC-001

# Stage-1 config (multi-tissue, 5 EP300 tissues). Place under $REPO/configs/.
STAGE1_CFG=$HOME/palmer_scratch/entexBERT-2_experiments/EN-TEx/configs/ep300_binding_regression_multitissue_enc001.yaml

# ADASTRA benchmark eval set + reference AUROC (built locally, upload alongside CTCF's)
export ADASTRA=$HOME/palmer_scratch/entexBERT-2_experiments/ADASTRA/ep300_adastra_evalset.csv.gz
export REFCSV=$HOME/palmer_scratch/entexBERT-2_experiments/ADASTRA/ep300_benchmark_reference_auroc.csv

# experiment tree
export WORK=$HOME/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/ep300_asb
export STAGE1_DIR=$WORK/stage1
export STAGE1_CKPT=$STAGE1_DIR/runs/reg/pytorch_model.bin
export BUILD=$WORK/inputs
mkdir -p "$WORK" "$STAGE1_DIR"
cd "$REPO"

# -----------------------------------------------------------------------------
# DIAGNOSTIC MODE (default ON here; export DIAG=0 for the full run).
# Purpose: exercise ALL 7 steps end-to-end in <=4h to validate the pipeline + get a
# first (under-trained) EP300 number, NOT a final result. It bounds every training step
# with a hard --max_steps (wall time is then independent of dataset size) and shrinks the
# Stage-2a s-sweep to 2 values. Everything else (builds, held-out selection, scoring) runs
# in full -- those are fast and are exactly what we want to smoke-test.
# For the real run: export DIAG=0 and (recommended) raise #SBATCH --time back to 20:00:00.
# -----------------------------------------------------------------------------
DIAG="${DIAG:-1}"
if [[ "$DIAG" == "1" ]]; then
  S1_MAX_STEPS=400          # Stage-1 trunk: ~cap; CTCF showed dev pearson plateaus by ~step 400
  S2A_MAX_STEPS=300         # Stage-2a per-s frozen-head fit
  S2B_MAX_STEPS=300         # Stage-2b unfreeze
  DIAG_SGRID="s_hat 20"     # 2-value sweep instead of 4 (resolved after s_hat in STEP 3)
  EVAL_EVERY=100            # bounded evals (few passes, not hundreds)
  echo "*** DIAGNOSTIC MODE: capped training (S1<=$S1_MAX_STEPS, 2a<=$S2A_MAX_STEPS, 2b<=$S2B_MAX_STEPS steps), 2-value s-sweep. NOT a final result. ***"
else
  S1_MAX_STEPS=-1; S2A_MAX_STEPS=-1; S2B_MAX_STEPS=-1; DIAG_SGRID=""; EVAL_EVERY=500
  echo "*** FULL MODE ***"
fi
# helper: emit '--max_steps N' only when N>0 (HF: max_steps=-1 means 'use num_train_epochs')
maxsteps() { [[ "$1" -gt 0 ]] && echo "--max_steps $1"; }

# =============================================================================
# PREFLIGHT
# =============================================================================
echo "=== preflight ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch,transformers; print('torch',torch.__version__,'| cuda',torch.cuda.is_available())"

for f in "$MODEL" "$REF" "$CHROMSIZES" "$HETSNV" "$ADASTRA" "$REFCSV" "$STAGE1_CFG"; do
  [ -e "$f" ] && echo "  found: $f" || { echo "  MISSING: $f"; exit 1; }
done

# (A) all 5 EP300 tissue files present?
echo "--- (A) EP300 tissue files under $ENTEX_EP300 ---"
missing=0
for t in transverse_colon upper_lobe_of_left_lung esophagus_muscularis_mucosa tibial_nerve gastroesophageal_sphincter; do
  for suf in narrowPeak fc.bigWig; do
    f="$ENTEX_EP300/ep300_1_${t}.${suf}"
    if [[ -s "$f" ]]; then echo "  found: $(basename "$f")"; else echo "  MISSING: $f"; missing=1; fi
  done
done
[[ "$missing" == 0 ]] || { echo "FATAL (A): EP300 tissue files incomplete -- run download_entex_chipseq.sh first."; exit 1; }

# (B) does hetSNVs.tsv actually contain EP300 rows? (Stage-2 blocker)
echo "--- (B) EP300 presence in hetSNVs.tsv ---"
python - <<PY
import os, sys, pandas as pd
tsv=os.environ["HETSNV"]; assay="$ASSAY"
# read only the assay column in chunks to keep memory bounded
hdr=pd.read_csv(tsv, sep="\t", nrows=0).columns.tolist()
acol=next((c for c in ("assay","Assay","assay_name","tf") if c in hdr), None)
if acol is None:
    sys.exit(f"FATAL (B): no assay-like column in hetSNVs header: {hdr[:12]}")
# SUBSTRING match (case-insensitive) -- the assay column holds labels like 'TF-ChIP-seq_EP300',
# not a bare 'EP300', and this is EXACTLY what build_betabinom_counts.py --assay does (str.contains).
n=0
for ch in pd.read_csv(tsv, sep="\t", usecols=[acol], chunksize=2_000_000):
    n += int(ch[acol].astype(str).str.contains(assay, case=False, na=False).sum())
print(f"  hetSNVs rows matching assay~'{assay}' (col '{acol}'): {n}")
if n < 1000:
    sys.exit(f"FATAL (B): only {n} {assay} rows in hetSNVs.tsv -- Stage-2 ASB has insufficient data. STOP.")
print(f"  (B) OK: {assay} present in hetSNVs.tsv")
PY
echo "[preflight] OK"

# =============================================================================
# STEP 1 -- build Stage-1 multi-tissue binding-regression inputs.
# =============================================================================
echo ""; echo "=== STEP 1: build Stage-1 (multi-tissue) inputs ==="
# Post-refactor: input generation is run_experiment.run_from_config -> build_dataset
# (the old generate_all_inputs.py was removed in the streamline). It writes train/dev/test.csv
# + *.meta.csv FLAT into --output_dir (no dataset/fold0 nesting), reading row_source.tissue_tracks.
DATASET="${DONOR}__TF-ChIP-seq_EP300"
S1_DATA="$STAGE1_DIR/inputs"
if [[ -f "$S1_DATA/train.csv" ]]; then
  echo "[STEP 1] $S1_DATA/train.csv exists -- skipping build"
else
  python -m entexbert2.run_experiment "$STAGE1_CFG" --ref_fasta "$REF" --output_dir "$S1_DATA"
fi
# regression sanity: peak windows must carry higher signal than background
python - "$S1_DATA/train.csv" "$S1_DATA/train.meta.csv" <<'PY'
import sys, pandas as pd, numpy as np
d=pd.read_csv(sys.argv[1]); lab=d["label"].astype(float)
# NaN-label count: consensus loci where every tissue was empty give nanmean-of-empty -> NaN label.
# (These are the 'Mean of empty slice' RuntimeWarnings at build time.) A large fraction => the
# 5-tissue merge is leaving gaps; a handful is benign. We report it and HARD-FAIL only if severe.
n_nan=int(lab.isna().sum()); frac=n_nan/max(len(d),1)
print(f"[STEP 1] train n={len(d)} label[mean={np.nanmean(lab):.3f} sd={np.nanstd(lab):.3f}] "
      f"NaN_labels={n_nan} ({frac:.1%})")
if n_nan:
    print(f"[STEP 1] WARNING: {n_nan} NaN labels -- these rows train to NaN loss. "
          f"Inspect the 5-tissue merge if frac is high.")
assert frac < 0.20, f"[STEP 1] FATAL: {frac:.1%} of labels are NaN (>20%) -- merge is leaving too many empty loci. STOP."
try:
    m=pd.read_csv(sys.argv[2])
    if "feature_type" in m.columns:
        g=m.assign(label=lab).groupby("feature_type")["label"].mean()
        pk,bg=g.get("peak",float("nan")),g.get("background",float("nan"))
        print(f"[STEP 1] peak_mean={pk:.3f} bg_mean={bg:.3f} sep={'OK' if pk>bg else 'BROKEN(peak<=bg) -- STOP'}")
        assert pk>bg, "peak signal <= background: bigWig path/coords wrong"
except FileNotFoundError: pass
PY
echo "[STEP 1] OK"

# =============================================================================
# STEP 2 -- Stage-1: train the EP300 binding trunk (regression on mean fold-change).
# =============================================================================
echo ""; echo "=== STEP 2: Stage-1 binding-trunk training ==="
if [[ -f "$STAGE1_CKPT" ]]; then
  echo "[STEP 2] trunk exists ($STAGE1_CKPT) -- skipping (delete to retrain)"
else
  # single-window binding regression. NOTE current (refactored) finetune contract:
  #   input_mode is 'single' (not 'ref_single'); no --main_num_labels (num_labels hardcoded 1);
  #   metric key is HF-prefixed eval_spearman.
  python -m entexbert2.finetune_entexbert2 \
      --model_name_or_path "$MODEL" \
      --data_path "$S1_DATA" \
      --input_mode single --task regression \
      --pooling_mode center_mean --center_pool_width 5 --head_num_layers 1 \
      --model_max_length 512 \
      --per_device_train_batch_size 16 --gradient_accumulation_steps 8 \
      --per_device_eval_batch_size 32 \
      --num_train_epochs 3 $(maxsteps "$S1_MAX_STEPS") \
      --learning_rate 3e-5 --weight_decay 0.01 --warmup_steps 50 \
      --logging_steps 50 --eval_steps "$EVAL_EVERY" --save_steps "$EVAL_EVERY" --save_total_limit 2 \
      --evaluation_strategy steps --load_best_model_at_end True \
      --metric_for_best_model spearman --greater_is_better True \
      --bf16 True --save_model True --eval_and_save_results True \
      --run_name "${DATASET}__stage1" --output_dir "$STAGE1_DIR/runs/reg"
fi
[ -f "$STAGE1_CKPT" ] || { echo "FATAL: Stage-1 trunk not written at $STAGE1_CKPT"; exit 1; }
echo "[STEP 2] OK -- trunk = $STAGE1_CKPT"

# =============================================================================
# STEP 3 -- EP300 beta-binomial counts + held-out chrom selection + s-hat.
# =============================================================================
echo ""; echo "=== STEP 3: counts + held-out chroms + s_hat ==="
COUNTS=$WORK/ep300_betabinom_counts.csv
if [[ ! -f "$COUNTS" ]]; then
  python build_betabinom_counts.py --hetsnv_tsv "$HETSNV" --assay "$ASSAY" --min_total_reads 20 --out "$COUNTS"
fi
# select held-out chroms on the REAL EP300 count distribution (chr1 in --special => spared, ~10% target).
# select_heldout_chroms.py writes only "<prefix>_per_chrom_counts.csv" and PRINTS the chosen set as
# "HELD-OUT SET: chrA+chrB  pos=..." to stdout -- capture that line (tee so it still lands in the log).
SEL=$WORK/ep300_heldout
python select_heldout_chroms.py --eval_csv "$COUNTS" --label_col imbalance_significance \
    --target 0.10 --top_train_spare 2 --max_set 2 --ci_floor 100 \
    --special chrX chrY chrM chr1 --out_prefix "$SEL" | tee "$WORK/heldout_select.log"
# parse "chrA+chrB" from the HELD-OUT SET line; '+'-separated -> comma-separated
HELDOUT=$(grep -m1 "HELD-OUT SET:" "$WORK/heldout_select.log" | sed -E 's/.*HELD-OUT SET: *([^ ]+).*/\1/' | tr '+' ',')
echo "[STEP 3] held-out chroms = $HELDOUT"
[[ -n "$HELDOUT" ]] || { echo "FATAL: no held-out chroms parsed from selector output"; exit 1; }
# build the fold_assignment YAML fragment: {chrA: 0, chrB: 0}
FOLD=$(python - <<PY
chroms="$HELDOUT".split(",")
print("{" + ", ".join(f"{c}: 0" for c in chroms if c) + "}")
PY
)

# s_hat via method-of-moments (Kleinman rho), same estimator as the CTCF run
SHAT=$(python - <<'PY'
import os, pandas as pd, numpy as np, sys
df=pd.read_csv(os.environ["WORK"]+"/ep300_betabinom_counts.csv")
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
# DIAG: 2-value sweep {s_hat, 20}; FULL: 4-value {s_hat, 20, 50, 100}. 's_hat' token -> real $SHAT.
if [[ -n "$DIAG_SGRID" ]]; then
  SGRID=$(echo "$DIAG_SGRID" | sed "s/s_hat/$SHAT/")
else
  SGRID="$SHAT 20 50 100"
fi
echo "[STEP 3] s_hat=$SHAT  sweep=$SGRID"
echo "[STEP 3] OK"

# =============================================================================
# STEP 4 -- build the Stage-2 ASB dataset (hap_pair, logit-ratio, held-out chroms -> TEST).
# =============================================================================
echo ""; echo "=== STEP 4: build Stage-2 ASB dataset ==="
cat > "$WORK/stage2.yaml" <<YAML
experiment: ep300_stage2_asb_enc001
ref_fasta: $REF
output_dir: $BUILD
row_source: {type: betabinom_counts, path: $COUNTS, assay: $ASSAY, min_total_reads: 20}
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
  fold_assignment: $FOLD
  exclude_boundary: true
  boundary_bp: 128
balance: {strategy: none}
YAML
python -m entexbert2.run_experiment "$WORK/stage2.yaml" --ref_fasta "$REF" --output_dir "$BUILD"
python -c "import pandas as pd; d=pd.read_csv('$BUILD/train.csv'); print('[STEP 4] train.csv cols:', list(d.columns), '| rows:', len(d))"
echo "[STEP 4] OK"

# =============================================================================
# STEP 5 -- Stage-2a: FROZEN trunk, LINEAR head, sweep s. Pick best by dev spearman.
# =============================================================================
echo ""; echo "=== STEP 5: Stage-2a frozen-trunk head training (s sweep) ==="
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
      --init_backbone_from "$STAGE1_CKPT" --freeze_backbone True \
      --model_max_length 512 \
      --per_device_train_batch_size 32 --per_device_eval_batch_size 64 \
      --num_train_epochs 1 $(maxsteps "$S2A_MAX_STEPS") \
      --learning_rate 1e-3 --warmup_steps 50 --weight_decay 0.01 \
      --logging_steps 20 --eval_steps "$EVAL_EVERY" --save_steps "$EVAL_EVERY" --save_total_limit 2 \
      --evaluation_strategy steps --load_best_model_at_end True \
      --metric_for_best_model spearman --greater_is_better True \
      --bf16 --save_model True --eval_and_save_results True \
      --output_dir "$RUN" --run_name "asb_2a_s${S}"
  SP=$(python - <<PY
import json,glob,os
st=os.path.join("$RUN","trainer_state.json")
val=json.load(open(st)).get("best_metric") if os.path.exists(st) else None
if val is None:
    er=glob.glob("$RUN/results/*/eval_results.json")
    if er: val=json.load(open(er[0])).get("eval_spearman")
print(val if val is not None else -1)
PY
)
  echo "    dev spearman = $SP"
  awk "BEGIN{exit !($SP > $BEST_SP)}" && { BEST_SP="$SP"; BEST_2A="$RUN"; BEST_S="$S"; }
done
echo "[STEP 5] BEST: s=$BEST_S dev_spearman=$BEST_SP ckpt=$BEST_2A"
echo "[STEP 5] OK"

# =============================================================================
# STEP 6 -- Stage-2b: gentle unfreeze from the best 2a checkpoint (low LR, no freeze).
# save_total_limit=2 (NOT 1): load_best_model_at_end needs best+latest, else _rotate_checkpoints crashes.
# =============================================================================
echo ""; echo "=== STEP 6: Stage-2b gentle unfreeze (s=$BEST_S) ==="
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
    --num_train_epochs 2 $(maxsteps "$S2B_MAX_STEPS") \
    --learning_rate 3e-6 --warmup_steps 50 --weight_decay 0.01 \
    --logging_steps 20 --eval_steps "$EVAL_EVERY" --save_steps "$EVAL_EVERY" --save_total_limit 2 \
    --evaluation_strategy steps --load_best_model_at_end True \
    --metric_for_best_model spearman --greater_is_better True \
    --bf16 --save_model True --eval_and_save_results True \
    --output_dir "$RUN2B" --run_name "asb_2b_s${BEST_S}"
echo "[STEP 6] OK"

# =============================================================================
# STEP 7 -- score ADASTRA + hetSNV (leak-free) for best-2a and 2b.
# =============================================================================
echo ""; echo "=== STEP 7: score (leak-free) ==="
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
      --hetsnv_tsv "$HETSNV" --assay "$ASSAY" \
      --train_coords "$BUILD/train.meta.csv" "$BUILD/dev.meta.csv" --drop_leaky \
      --left_bp 128 --right_bp 128 --batch_size 64 --device cuda \
      --out "$WORK/score_${NAME}_hetsnv"
done
echo "[STEP 7] OK"
echo ""
echo "============================================================"
echo " EP300 PIPELINE COMPLETE.  (mode: DIAG=$DIAG)"
if [[ "$DIAG" == "1" ]]; then
  echo "   *** DIAGNOSTIC run: training was step-capped and the s-sweep was 2-value."
  echo "       The AUROC below is UNDER-TRAINED -- it validates the pipeline and gives a"
  echo "       first signal, NOT a final number. For the real result: export DIAG=0 and"
  echo "       raise #SBATCH --time to ~20:00:00."
fi
echo "   Stage-1 trunk: $STAGE1_CKPT"
echo "   held-out:      $HELDOUT"
echo "   best s:        $BEST_S  (dev spearman $BEST_SP)"
echo "   2a / 2b:       $BEST_2A  |  $RUN2B"
echo "   scores:        $WORK/score_2a_*  and  $WORK/score_2b_*"
echo " WIN CONDITION: leak-free |Delta| AUROC > ~0.596 (Enformer, best DNN on EP300)."
echo " Motif/PWM methods do NOT score EP300 -- the comparison is DNN-only."
echo "============================================================"
