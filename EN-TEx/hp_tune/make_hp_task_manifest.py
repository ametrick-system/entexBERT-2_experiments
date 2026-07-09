#!/usr/bin/env python3
"""
make_hp_task_manifest.py — cartesian product of {hp configs} x {datasets} x {seeds} -> the SLURM task
manifest run_hp_search.sbatch consumes. One line = one baseline fine-tune (a single config on a
single dataset with a single training seed). Selection is on the DEV split; test is never touched.

Datasets come from a generate_all_inputs batch manifest (so data_dir points at each dataset's fold dir).

Output TSV columns (tab-separated):
  run_tag  config_id  data_dir  seed  learning_rate  batch_size  epochs  weight_decay  warmup_steps  head_num_layers
  run_tag = <config_id>__<dataset_id>__seed<seed>   (unique per task)

Usage:
  python make_hp_task_manifest.py \
      --hp_configs hp/hp_configs.tsv \
      --inputs_manifest inputs/<run>/generate_all_inputs_manifest.json \
      --seeds 42 1 2 \
      --out hp/tasks_hp.tsv
"""
import argparse, csv, json, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hp_configs", required=True, help="hp_configs.tsv from make_hp_configs.py")
    ap.add_argument("--inputs_manifest", required=True, help="generate_all_inputs_manifest.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2], help="training seeds")
    ap.add_argument("--only_ok", action="store_true", help="only datasets with generate status ok/skipped")
    ap.add_argument("--out", default="tasks_hp.tsv")
    args = ap.parse_args()

    with open(args.hp_configs) as f:
        configs = list(csv.DictReader(f, delimiter="\t"))
    datasets = json.load(open(args.inputs_manifest))["datasets"]
    if args.only_ok:
        datasets = [c for c in datasets if c["status"] in ("ok", "skipped")]

    hp_cols = ["learning_rate", "batch_size", "epochs", "weight_decay", "warmup_steps", "head_num_layers"]
    out_cols = ["run_tag", "config_id", "data_dir", "seed"] + hp_cols

    n = 0
    with open(args.out, "w") as f:
        f.write("\t".join(out_cols) + "\n")
        for cfg in configs:
            for dataset in datasets:
                for seed in args.seeds:
                    run_tag = f"{cfg['config_id']}__{dataset['dataset_id']}__seed{seed}"
                    row = [run_tag, cfg["config_id"], dataset["output_dir"], str(seed)] + [cfg[c] for c in hp_cols]
                    f.write("\t".join(row) + "\n")
                    n += 1
    print(f"{n} task line(s) -> {args.out}  "
          f"({len(configs)} configs x {len(datasets)} datasets x {len(args.seeds)} seeds). "
          f"Set --array=0-{n-1}.", file=sys.stderr)
    print(n)


if __name__ == "__main__":
    main()
