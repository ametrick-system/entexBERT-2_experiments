#!/bin/bash
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=rtx3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=eb2_stage2b_score
#SBATCH --output=logs/eb2_stage2b_score_%j.log
#SBATCH --error=logs/eb2_stage2b_score_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# entexBERT-2 Stage-2b + scoring  (STANDALONE — resumes from a finished STEP 2a)
# -----------------------------------------------------------------------------
# Run this AFTER the STEP-2a sweep in run_stage2_ctcf_asb.sbatch has finished
# (you can kill that job the moment the last "--- Stage-2a s=..." run completes;
#  every 2a sweep checkpoint is already saved under $WORK/runs/asb_2a_s*).
#   STEP A  rediscover the completed 2a runs on disk, pick best by dev spearman
#   STEP 2b gentle unfreeze from the best 2a checkpoint (low LR, no freeze)
#   STEP 3  score ADASTRA + hetSNV, leak-free, for BOTH the best 2a and the 2b
# Head: linear. Loss: precision-weighted MSE. Matches run_stage2_ctcf_asb.sbatch.
# =============================================================================

set -euo pipefail
mkdir -p logs

module purge
module load StdEnv
module load miniconda
conda activate eb2

# ---- paths (IDENTICAL to run_stage2_ctcf_asb.sbatch — edit only if yours moved) ----
REPO=$HOME/entexBERT-2
export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
MODEL=$REPO/DNABERT-2-117M-attention
REF=$HOME/reference_genome/hg38.fa
HETSNV=$HOME/palmer_scratch/entex_data/hetSNVs.tsv
ADASTRA=$HOME/palmer_scratch/entexBERT-2_experiments/ADASTRA/ctcf_adastra_evalset.csv
REFCSV=$HOME/palmer_scratch/entexBERT-2_experiments/ADASTRA/ctcf_benchmark_reference_auroc.csv

export WORK=$HOME/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/stage2_ctcf_asb   # exported: STEP A's python reads os.environ["WORK"]
BUILD=$WORK/inputs
cd "$REPO"

echo "=== preflight ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch,transformers; print('torch',torch.__version__,'| cuda',torch.cuda.is_available())"
for f in "$MODEL" "$REF" "$HETSNV" "$ADASTRA" "$BUILD/train.meta.csv" "$BUILD/dev.meta.csv"; do
  [ -e "$f" ] && echo "  found: $f" || { echo "  MISSING: $f"; exit 1; }
done

# =============================================================================
# STEP A -- rediscover completed 2a sweep runs and pick the best by dev spearman.
# Mirrors the in-job selection: best_metric from trainer_state.json, else
# eval_results.json eval_spearman. A run counts only if pytorch_model.bin exists.
# =============================================================================
echo ""; echo "=== STEP A: select best 2a checkpoint ==="
read BEST_2A BEST_S BEST_SP < <(python - <<'PY'
import json, glob, os, sys
# ALL diagnostics -> stderr; ONLY the final "path s spearman" answer -> stdout,
# so the shell `read` captures exactly the answer line (not a [cand]/[skip] line).
work = os.environ["WORK"]
best = (None, None, -1e9)
runs = sorted(glob.glob(os.path.join(work, "runs", "asb_2a_s*")))
if not runs:
    sys.exit("NO 2a runs found under %s/runs/asb_2a_s* -- has STEP 2a finished?" % work)
for run in runs:
    if not os.path.exists(os.path.join(run, "pytorch_model.bin")):
        print(f"[skip] {os.path.basename(run)}: no pytorch_model.bin (incomplete)", file=sys.stderr)
        continue
    s = os.path.basename(run).split("asb_2a_s", 1)[1]
    val = None
    st = os.path.join(run, "trainer_state.json")
    if os.path.exists(st):
        val = json.load(open(st)).get("best_metric")
    if val is None:
        er = glob.glob(os.path.join(run, "results", "*", "eval_results.json"))
        if er:
            val = json.load(open(er[0])).get("eval_spearman")
    val = -1e9 if val is None else float(val)
    print(f"[cand] s={s:<6} dev_spearman={val:.5f}  {run}", file=sys.stderr)
    if val > best[2]:
        best = (run, s, val)
if best[0] is None:
    sys.exit("No COMPLETE 2a run (with pytorch_model.bin) found.")
print(f"{best[0]} {best[1]} {best[2]:.5f}")   # the ONLY stdout line
PY
)
echo "[STEP A] BEST 2a: s=$BEST_S  dev_spearman=$BEST_SP"
echo "[STEP A] ckpt = $BEST_2A"
[ -f "$BEST_2A/pytorch_model.bin" ] || { echo "best 2a ckpt missing pytorch_model.bin"; exit 1; }
echo "[STEP A] OK"

# =============================================================================
# STEP 2b -- gentle unfreeze from the best 2a checkpoint (low LR, no freeze).
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
    --logging_steps 20 --eval_steps 500 --save_steps 500 --save_total_limit 2 \
    --evaluation_strategy steps --load_best_model_at_end True \
    `# save_total_limit=2 (NOT 1): load_best_model_at_end needs best+latest both kept;` \
    `# save_total_limit=1 crashes in _rotate_checkpoints once the best is an earlier ckpt.` \
    --metric_for_best_model spearman --greater_is_better True \
    --bf16 --save_model True --eval_and_save_results True \
    --output_dir "$RUN2B" --run_name "asb_2b_s${BEST_S}"
echo "[STEP 2b] OK"

# =============================================================================
# STEP 3 -- score ADASTRA + hetSNV (leak-free) for BOTH the best 2a and the 2b.
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
echo " STAGE-2b + SCORING COMPLETE."
echo "   best 2a: s=$BEST_S  ($BEST_2A)"
echo "   2b:      $RUN2B"
echo "   scores:  $WORK/score_2a_*  and  $WORK/score_2b_*"
echo " Compare leak-free |Delta| AUROC vs the ~0.68 probe baseline"
echo " and the benchmark table (adastra scorer prints the ranking)."
echo "============================================================"
