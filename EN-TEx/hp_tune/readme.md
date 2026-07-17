```bash
cd ~/entexBERT-2_experiments/EN-TEx/setup
python plan_inputs.py --tsv /home/asm242/entex_data/hetSNVs.tsv \
    --tf CTCF --min_total_reads 10 --suggest_test_frac 0.10 \
    --out_prefix input_plan/ctcf_ref_single/ctcf_ref_single

./build_inputs.sh ../configs/ctcf_ref_single.yaml ctcf_ref_single

cd ../hp_tune
python make_hp_configs.py --n 10 --sampler_seed 0 --out hp_configs.tsv

python make_hp_task_manifest.py \
    --hp_configs hp_configs.tsv \
    --inputs_manifest ../setup/inputs/ctcf_ref_single/generate_all_inputs_manifest.json \
    --seeds 42 13 1 \
    --format dsq --runner run_hp_job.sh --out jobs_hp.txt

module load dSQ
dsq --job-file jobs_hp.txt \
    --partition pi_gerstein_gpu \
    --gpus 1 --cpus-per-task 4 --mem 48G --time 06:00:00 \
    --output logs/%A_%a.out
sbatch dsq-jobs_hp-2026-07-09.sh
```