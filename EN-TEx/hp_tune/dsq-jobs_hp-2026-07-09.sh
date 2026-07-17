#!/bin/bash
#SBATCH --output logs/%A_%a.out
#SBATCH --array 0-119
#SBATCH --job-name dsq-jobs_hp
#SBATCH --partition pi_gerstein_gpu --gpus 1 --cpus-per-task 4 --mem 48G --time 06:00:00

# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py --job-file /vast/palmer/home.mccleary/asm242/entexBERT-2_experiments/EN-TEx/hp_tune/jobs_hp.txt --status-dir /vast/palmer/home.mccleary/asm242/entexBERT-2_experiments/EN-TEx/hp_tune

