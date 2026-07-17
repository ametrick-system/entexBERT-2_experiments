#!/bin/bash
#SBATCH --output logs/%A_%a.out
#SBATCH --array 0-104
#SBATCH --job-name dsq-rerun_hp
#SBATCH --partition pi_gerstein_gpu --gpus 1 --cpus-per-task 4 --mem 48G --time 06:00:00

# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py --job-file /vast/palmer/scratch/gerstein/asm242/entexBERT-2_experiments/EN-TEx/hp_tune/rerun_hp.txt --status-dir /vast/palmer/scratch/gerstein/asm242/entexBERT-2_experiments/EN-TEx/hp_tune

