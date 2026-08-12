#!/usr/bin/env bash
#SBATCH --job-name=eb2_ctcf_betabinom_asb
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

# ============================================================
# CTCF ALLELE-SPECIFIC-BINDING regression under a BETA-BINOMIAL read-count likelihood.
# entexBERT-2 predicts logit p = P(read from hap1) via the hap1-vs-hap2 twin; the loss is the
# beta-binomial NLL over (k=hap1 reads, n=total reads). This is the SUPERVISED Track-2 model:
# unlike the binding-regression Track-1, it trains ON allelic counts, so leakage is central and
# the held-out chromosomes (select_heldout_chroms.py) are load-bearing.
#
# CODE PRECONDITIONS (all in the committed finetune_entexbert2.py betabinomial task):
#   * --task betabinomial  reads a [k, n] label pair from train.csv (H1).
#   * --input_mode hap_pair is REQUIRED (H4c): the twin must be (hap1, hap2) so mu=-twin=logit
#     P(hap1) matches k=hap1_count. Any other mode raises (would invert calibration).
#   * best checkpoint on eval_spearman = Spearman(pred logit p, observed logit k/n) = the
#     calibration-gap metric; eval also reports pearson + balanced ASB-auroc.
#
# >>> UPSTREAM BUILD DEPENDENCY (NOT yet wired — see the STEP-1 guard below):
#     train.csv MUST carry BOTH a 'k' and an 'n' column. The current build
#     (generate_all_inputs -> build_dataset) does not emit them for any label type. The STEP-1
#     guard below fails fast until the build carries k,n. Resolution fork is in chat.
# ============================================================

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/ctcf_betabinom_asb.yaml}"
TAG="$(basename "$CONFIG" .yaml)"
DATASET="${DATASET:-ALLDONOR__TF-ChIP-seq_CTCF}"   # betabinom aggregates across donors+tissues
EXP_DIR="${EXP_DIR:-experiments/betabinom_CTCF_${TAG}}"
DATA_DIR="$EXP_DIR/inputs/$DATASET/fold0"
RUNS_DIR="$EXP_DIR/runs"
OUT="$RUNS_DIR/bb"
MODEL="${MODEL:-$HOME/entexBERT-2/DNABERT-2-117M-attention}"
CONDA_ENV="${CONDA_ENV:-eb2}"
mkdir -p "$EXP_DIR" logs "$OUT"

module load miniconda 2>/dev/null || true
# shellcheck disable=SC1091
source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

# ============================================================
# STEP 0 — (run ONCE before submitting, documented here for reproducibility; not auto-run)
#   1) aggregate per-(donor,locus) counts:
#      python build_betabinom_counts.py --hetsnv_tsv $HOME/entex_data/hetSNVs.tsv \
#          --assay CTCF --out ctcf_betabinom_counts.csv
#   2) pick held-out chromosomes on THIS table (uses imbalance_significance as the positive col):
#      python select_heldout_chroms.py --eval_csv ctcf_betabinom_counts.csv \
#          --label_col imbalance_significance --out_prefix ctcf_bb
#      -> put the printed HELD-OUT SET into the config's held-out chromosome list.
# ============================================================

# ============================================================
# STEP 1 — build betabinomial inputs (hap_pair windows + 100kb bin-split leakage dedup,
#          held-out chromosomes excluded from train, k/n carried into train.csv).
# ============================================================
if [[ -f "$DATA_DIR/train.csv" ]]; then
    echo "[build] $DATA_DIR/train.csv exists — skipping build (delete to force rebuild)"
else
    echo "[build] generating CTCF betabinomial inputs ($TAG) for $DATASET"
    python -m entexbert2.scripts.generate_all_inputs \
        "$CONFIG" \
        --output_dir "$EXP_DIR/inputs"
fi

# --- STEP 1a: hap_pair build MUST produce sequence1/sequence2 (the twin) ---
HEADER="$(head -1 "$DATA_DIR/train.csv")"
grep -q "sequence1" <<< "$HEADER" || { echo "FATAL: 'sequence1' missing — hap_pair twin build expected"; exit 1; }
grep -q "sequence2" <<< "$HEADER" || { echo "FATAL: 'sequence2' missing — hap_pair twin build expected"; exit 1; }

# --- STEP 1b: FAIL FAST on the known build gap — betabinomial needs k AND n columns ---
grep -q "\bk\b" <<< "$(echo "$HEADER" | tr ',' '\n')" || { echo "FATAL: 'k' column missing from train.csv — build not wired to carry hap1 counts (see UPSTREAM BUILD DEPENDENCY)"; exit 1; }
grep -q "\bn\b" <<< "$(echo "$HEADER" | tr ',' '\n')" || { echo "FATAL: 'n' column missing from train.csv — build not wired to carry total counts"; exit 1; }
echo "[build] twin columns (sequence1, sequence2) + count columns (k, n) present"

# --- STEP 1c: count-target sanity — n should be >= k >= 0, and some loci imbalanced ---
echo "=== [build] betabinomial target summary (k/n distribution + depth) ==="
for split in train dev test; do
    f="$DATA_DIR/${split}.csv"; [[ -f "$f" ]] || continue
    python - "$f" "$split" <<'PY'
import sys, pandas as pd, numpy as np
f, split = sys.argv[1], sys.argv[2]
d = pd.read_csv(f)
k = d["k"].astype(float); n = d["n"].astype(float)
bad = int(((k < 0) | (n < k) | (n <= 0)).sum())
phat = (k / n).replace([np.inf, -np.inf], np.nan).dropna()
print(f"  {split:5s} n_rows={len(d):7d}  n[median={n.median():.0f} max={n.max():.0f}]  "
      f"k/n[median={phat.median():.3f}]  frac|k/n-0.5|>0.2={ (abs(phat-0.5)>0.2).mean():.3f}  "
      f"malformed_rows={bad}{'  <-- STOP' if bad else ''}")
PY
done
echo "    ^ malformed_rows MUST be 0 (need n>=k>=0, n>0); k/n median near 0.5 with a spread is expected."

# ============================================================
# STEP 2 — finetune: BETA-BINOMIAL task, hap_pair twin, single scalar mu head.
#   NOTE vs regression sbatch: --task betabinomial, --input_mode hap_pair (REQUIRED),
#   no --predict_sigma / --heteroscedastic (H4c forbids combining them).
# ============================================================
echo "=== [betabinom $TAG] $DATASET | hetSNV k/n | hap_pair twin | beta-binomial NLL ==="
python -m entexbert2.finetune_entexbert2 \
    --model_name_or_path "$MODEL" \
    --data_path "$DATA_DIR" \
    --input_mode hap_pair \
    --task betabinomial \
    --main_num_labels 1 \
    --pooling_mode center_mean \
    --center_pool_width 5 \
    --head_num_layers 1 \
    --model_max_length 512 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 3 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --warmup_steps 50 \
    --logging_steps 50 \
    --eval_steps 100 \
    --save_steps 100 \
    --evaluation_strategy steps \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_spearman \
    --greater_is_better True \
    --bf16 True \
    --run_name "${DATASET}__${TAG}" \
    --output_dir "$OUT" \
    --save_model True \
    --eval_and_save_results True \
    2>&1 | tee "$OUT/train.log"

echo "=== training DONE. Beta-binomial metrics (eval_spearman = calibration-gap number,"
echo "    eval_pearson, eval_auroc = balanced ASB-AUROC) in $OUT/train.log ==="
echo "    NEXT: score held-out chromosomes + the ADASTRA CTCF eval set with the checkpoint in $OUT."
