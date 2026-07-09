```bash
cd ~/entexBERT-2_experiments/EN-TEx/setup
python plan_inputs.py --tsv /home/asm242/entex_data/hetSNVs.tsv \
    --tf CTCF --min_total_reads 10 --suggest_test_frac 0.10 \
    --out_prefix input_plan/ctcf_ref_single/ctcf_ref_single
```