# =============================================================================
# entexBERT-2 Stage-2 SMOKE TEST (Tier-0 end-to-end, real DNABERT-2 backbone)
# -----------------------------------------------------------------------------
# Purpose: confirm the refactored repo runs a REAL forward pass on the cluster
# (all prior tests used a mock backbone locally). This is NOT a real experiment
# -- tiny build, 10 steps, no metric interpretation. It exercises the full path:
#     build_inputs -> finetune (twin + precision weight) -> score_asb (dump pools)
#     -> probe_frozen_trunk (Tier-0 verdict)
# If every STEP prints "OK", the pipeline is wired correctly end-to-end.
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

WORK=$HOME/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/smoke
BUILD=$WORK/inputs
RUN=$WORK/runs/asb
mkdir -p "$WORK"
cd "$REPO"

echo "=== preflight ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch,transformers; print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'| transformers',transformers.__version__)"
for f in "$MODEL" "$REF" "$HETSNV" "$ADASTRA"; do
  [ -e "$f" ] && echo "  found: $f" || { echo "  MISSING: $f"; exit 1; }
done

# =============================================================================
# STEP 1 -- build a TINY Stage-2 dataset (betabinom counts -> logit-ratio target)
# We restrict to CTCF and a small chromosome subset so the build is seconds, not
# minutes. run_experiment reads the yaml, writes {train,dev,test}.csv to --output_dir.
# =============================================================================

# 1a. aggregate hetSNV counts -> per-locus (k,n). Keep it tiny: CTCF only.
#     (build_betabinom_counts writes chr,ref_start,...,k,n,imbalance_significance,assay,n_tissues)
python build_betabinom_counts.py \
    --hetsnv_tsv "$HETSNV" \
    --assay CTCF \
    --min_total_reads 20 \
    --out "$WORK/smoke_counts_full.csv"

# 1b. subsample to ~2000 loci (all we need to smoke-test; keeps the build fast).
python - <<PY
import pandas as pd
df = pd.read_csv("$WORK/smoke_counts_full.csv")
# keep a couple of held-out chroms + a few train chroms so the split is non-empty on both sides
keep = df[df["chr"].isin(["chr1","chr2","chr5","chr12"])]
keep = keep.sample(n=min(2000, len(keep)), random_state=1)
keep.to_csv("$WORK/smoke_counts.csv", index=False)
print("[1b] wrote", len(keep), "loci ->", "$WORK/smoke_counts.csv",
      "| chroms:", sorted(keep['chr'].unique()))
PY

python -m entexbert2.run_experiment "$WORK/smoke_stage2.yaml" \
    --ref_fasta "$REF" --output_dir "$BUILD"

echo "[STEP 1] build outputs:"; ls -la "$BUILD"
head -2 "$BUILD/train.csv"
python -c "import pandas as pd; d=pd.read_csv('$BUILD/train.csv'); print('[STEP 1] train.csv cols:', list(d.columns), '| rows:', len(d))"
echo "[STEP 1] OK"

# =============================================================================
# STEP 2 -- finetune 10 steps (real backbone forward + precision-weighted loss)
# Tiny: 1 epoch cap via --max_steps 10, small batch. We only care that it RUNS and
# writes a checkpoint + run_config.json, not that the metrics are good.
# =============================================================================
echo ""; echo "=== STEP 2: finetune (10 steps, real backbone) ==="
python -m entexbert2.finetune_entexbert2 \
    --model_name_or_path "$MODEL" \
    --data_path "$BUILD" \
    --task regression --input_mode hap_pair \
    --pooling_mode center_mean --center_pool_width 5 \
    --head_num_layers 1 --neff_s 50 \
    --model_max_length 512 \
    --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
    --max_steps 10 --num_train_epochs 1 \
    --learning_rate 3e-5 --warmup_steps 2 \
    --logging_steps 1 --eval_steps 10 --save_steps 10 --save_total_limit 1 \
    --evaluation_strategy steps --load_best_model_at_end False \
    --bf16 --save_model True \
    --output_dir "$RUN"

echo "[STEP 2] run dir:"; ls -la "$RUN"
test -f "$RUN/run_config.json" && echo "[STEP 2] run_config.json written" || { echo "[STEP 2] FAIL: no run_config.json"; exit 1; }
echo "[STEP 2] OK"

# =============================================================================
# STEP 3 -- score ADASTRA + DUMP the frozen-trunk pools (the Tier-0 input)
# --dump_embeddings writes {out}_pools.npz (id, pool_ref, pool_alt, label, leaky).
# leak filter: pass the built train/dev meta so trained bins are flagged.
# =============================================================================
echo ""; echo "=== STEP 3: score_asb --eval adastra --dump_embeddings ==="
python -m entexbert2.score_asb \
    --eval adastra \
    --checkpoint_dir "$RUN" \
    --ref_fasta "$REF" \
    --eval_csv "$ADASTRA" \
    ${REFCSV:+--reference_csv "$REFCSV"} \
    --train_coords "$BUILD/train.meta.csv" "$BUILD/dev.meta.csv" \
    --drop_leaky \
    --left_bp 128 --right_bp 128 \
    --batch_size 64 --device cuda \
    --dump_embeddings \
    --out "$WORK/smoke_adastra"

test -f "$WORK/smoke_adastra_pools.npz" && echo "[STEP 3] pools npz written" || { echo "[STEP 3] FAIL: no _pools.npz"; exit 1; }
python -c "import numpy as np; z=np.load('$WORK/smoke_adastra_pools.npz', allow_pickle=True); print('[STEP 3] npz keys:', list(z.keys()), '| n:', len(z[list(z.keys())[0]]))"
echo "[STEP 3] OK"

# =============================================================================
# STEP 4 -- Tier-0 probe on the dumped pools (CPU-fine, but we're already on GPU node)
# Fits precision-weighted linear+mlp heads on the frozen trunk, sweeps s, prints
# the GO/MARGINAL/STOP verdict vs the ~0.68 |Delta| probe baseline.
# =============================================================================
echo ""; echo "=== STEP 4: probe_frozen_trunk (Tier-0 verdict) ==="
python -m entexbert2.probe_frozen_trunk \
    --pools "$WORK/smoke_adastra_pools.npz" \
    --evalset "$ADASTRA" \
    --heldout_chroms chr5 chr12 \
    --neff_s 0 20 50 100 \
    --heads linear mlp \
    --probe_baseline 0.68 \
    --out "$WORK/smoke_probe"

echo "[STEP 4] OK"
echo ""
echo "============================================================"
echo " SMOKE TEST COMPLETE -- if all 4 STEPs printed OK, the"
echo " refactored pipeline runs end-to-end on the real backbone."
echo " (Verdict/AUROC numbers here are NOT meaningful: 10-step"
echo "  train on ~2k loci. This only proves the wiring.)"
echo "============================================================"
