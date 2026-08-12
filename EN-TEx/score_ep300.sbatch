#!/bin/bash
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=rtx3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=eb2_ep300_score
#SBATCH --output=logs/eb2_ep300_score_%j.log
#SBATCH --error=logs/eb2_ep300_score_%j.err
#SBATCH --mail-type=END,FAIL
# =============================================================================
# STANDALONE EP300 SCORING (STEP 7 only).
# Re-runs leak-free ADASTRA + hetSNV scoring for the best Stage-2a checkpoint and
# the Stage-2b checkpoint from an ALREADY-COMPLETED training run. Use after the
# training steps finished but STEP 7 crashed (e.g. the CTCF_AUROC column bug).
# Requires the PATCHED score_asb.py (generic *_AUROC column) on the cluster:
#   /home/asm242/entexBERT-2/src/entexbert2/score_asb.py
# =============================================================================
set -euo pipefail
mkdir -p logs

module purge
module load StdEnv
module load miniconda
conda activate eb2

# ---- paths (must match the training run) ------------------------------------
REPO=$HOME/entexBERT-2
export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
REF=$HOME/reference_genome/hg38.fa
export HETSNV=$HOME/palmer_scratch/entex_data/hetSNVs.tsv
ASSAY=EP300
export ADASTRA=$HOME/palmer_scratch/entexBERT-2_experiments/ADASTRA/ep300_adastra_evalset.csv.gz
export REFCSV=$HOME/palmer_scratch/entexBERT-2_experiments/ADASTRA/ep300_benchmark_reference_auroc.csv
export WORK=$HOME/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/ep300_asb
export BUILD=$WORK/inputs
cd "$REPO"

# ---- preflight: patched scorer + required inputs present ---------------------
echo "=== preflight ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
SCORER=$REPO/src/entexbert2/score_asb.py   # the copy python -m entexbert2.score_asb runs
[ -f "$SCORER" ] || { echo "FATAL: $SCORER not found"; exit 1; }
if grep -q 'sort_values("CTCF_AUROC"' "$SCORER"; then
  echo "FATAL: $SCORER still hardcodes CTCF_AUROC -- apply the generic-*_AUROC patch first. STOP."; exit 1
fi
python - <<PY
import pandas as pd, os, sys
ref=os.environ["REFCSV"]
if not os.path.exists(ref): sys.exit(f"FATAL: REFCSV not found: {ref}")
cols=pd.read_csv(ref, nrows=0).columns.tolist()
want="${ASSAY}_AUROC"
if want not in cols and len([c for c in cols if c.endswith("_AUROC")])!=1:
    sys.exit(f"FATAL: reference has no '{want}' and not exactly one *_AUROC col: {cols}")
print(f"  reference OK ({want} resolvable)")
PY
for f in "$REF" "$ADASTRA" "$HETSNV" "$BUILD/train.meta.csv" "$BUILD/dev.meta.csv"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }; echo "  found: $f"
done

# ---- auto-discover checkpoints from the training tree ------------------------
# best Stage-2a = the asb_2a_s* run with the highest dev spearman (same rule as STEP 5:
# trainer_state.json best_metric, fallback results/*/eval_results.json eval_spearman).
echo ""; echo "=== locating checkpoints under $WORK/runs ==="
SEL=$(mktemp)
python - > "$SEL" <<'PY'
import json, glob, os
work=os.environ["WORK"]; best=(-1.0, "", "")
for run in sorted(glob.glob(os.path.join(work,"runs","asb_2a_s*"))):
    s=os.path.basename(run).replace("asb_2a_s","")
    val=None
    st=os.path.join(run,"trainer_state.json")
    if os.path.exists(st):
        val=json.load(open(st)).get("best_metric")
    if val is None:
        er=glob.glob(os.path.join(run,"results","*","eval_results.json"))
        if er: val=json.load(open(er[0])).get("eval_spearman")
    if val is not None and float(val) > best[0]:
        best=(float(val), run, s)
print(f"{best[1]}\t{best[2]}\t{best[0]:.5f}" if best[1] else "\t\t-1")
PY
BEST_2A=$(cut -f1 "$SEL"); BEST_S=$(cut -f2 "$SEL"); BEST_SP=$(cut -f3 "$SEL"); rm -f "$SEL"
[ -n "$BEST_2A" ] || { echo "FATAL: no asb_2a_s* run found under $WORK/runs"; exit 1; }
echo "  best 2a: s=$BEST_S dev_spearman=$BEST_SP -> $BEST_2A"

# Stage-2b run: prefer the s-matched name, else the sole asb_2b_s* dir
RUN2B=$WORK/runs/asb_2b_s${BEST_S}
if [ ! -d "$RUN2B" ]; then
  RUN2B=$(ls -d "$WORK"/runs/asb_2b_s* 2>/dev/null | head -1 || true)
fi
if [ -n "${RUN2B:-}" ] && [ -d "$RUN2B" ]; then
  echo "  2b run: $RUN2B"
else
  echo "  WARNING: no asb_2b_s* run found -- scoring 2a only"; RUN2B=""
fi

# ---- score (leak-free): best-2a and (if present) 2b -------------------------
echo ""; echo "=== scoring (leak-free) ==="
TAGS=("2a:$BEST_2A")
[ -n "$RUN2B" ] && TAGS+=("2b:$RUN2B")
for TAG in "${TAGS[@]}"; do
  NAME=${TAG%%:*}; CKPT=${TAG#*:}
  echo "--- scoring $NAME ($CKPT) ---"
  python -m entexbert2.score_asb \
      --eval adastra --checkpoint_dir "$CKPT" --ref_fasta "$REF" \
      --eval_csv "$ADASTRA" --assay "$ASSAY" ${REFCSV:+--reference_csv "$REFCSV"} \
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

echo ""
echo "============================================================"
echo " EP300 SCORING COMPLETE."
echo "   best 2a: s=$BEST_S (dev spearman $BEST_SP) -> $BEST_2A"
echo "   2b:      ${RUN2B:-<none>}"
echo "   outputs: $WORK/score_2a_*  and  $WORK/score_2b_*"
echo " WIN CONDITION: leak-free |Delta| AUROC > ~0.596 (Enformer, best DNN on EP300)."
echo "============================================================"
