#!/usr/bin/env python3
"""
make_hp_task_manifest.py — expand {hp configs} x {datasets} x {seeds} into a task list for the
ref_single BASELINE hyperparameter search. Two output formats:

  --format tsv  (default): columns run_hp_search.sbatch reads (the classic SLURM-array path).
      run_tag  config_id  data_dir  seed  learning_rate  batch_size  epochs  weight_decay  warmup_steps  head_num_layers
  --format dsq: a dead-Simple-Queue job file, ONE self-contained command per line:
      bash run_hp_job.sh <run_tag> <data_dir> <seed> <lr> <batch> <epochs> <wd> <warmup> <head_layers>
      Submit with:  module load dSQ && dsq --job-file <out> --partition gpu --account <acct> \
                        --gpus 1 --cpus-per-task 4 --mem 48G --time 06:00:00 --output logs/%A_%a.out
      Re-run only failures later:  dsqa -j <arrayjobid> > rerun_hp.txt   (dSQAutopsy)

One task = one baseline fine-tune (single config, single dataset, single seed). Selection is on the
DEV split; test is never touched. run_tag = <config_id>__<dataset_id>__seed<seed> (unique).
Datasets come from a generate_all_inputs batch manifest (data_dir = each dataset's fold dir).

Usage:
  python make_hp_task_manifest.py \
      --hp_configs hp_tune/hp_configs.tsv \
      --inputs_manifest inputs/<run>/generate_all_inputs_manifest.json \
      --seeds 42 1 2 \
      --format dsq --out hp_tune/jobs_hp.txt
"""
import argparse, csv, json, sys

HP_COLS = ["learning_rate", "batch_size", "epochs", "weight_decay", "warmup_steps", "head_num_layers"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hp_configs", required=True, help="hp_configs.tsv from make_hp_configs.py")
    ap.add_argument("--inputs_manifest", required=True, help="generate_all_inputs_manifest.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2], help="training seeds")
    ap.add_argument("--only_ok", action="store_true", help="only datasets with generate status ok/skipped")
    ap.add_argument("--format", choices=["tsv", "dsq"], default="tsv",
                    help="tsv = column manifest for run_hp_search.sbatch; dsq = job file (one command per line)")
    ap.add_argument("--runner", default="run_hp_job.sh",
                    help="dsq only: per-job runner script referenced in each command line")
    ap.add_argument("--out", default="tasks_hp.tsv")
    args = ap.parse_args()

    with open(args.hp_configs) as f:
        configs = list(csv.DictReader(f, delimiter="\t"))
    datasets = json.load(open(args.inputs_manifest))["datasets"]
    if args.only_ok:
        datasets = [c for c in datasets if c["status"] in ("ok", "skipped")]

    n = 0
    with open(args.out, "w") as f:
        if args.format == "tsv":
            f.write("\t".join(["run_tag", "config_id", "data_dir", "seed"] + HP_COLS) + "\n")
        for cfg in configs:
            for dataset in datasets:
                for seed in args.seeds:
                    run_tag = f"{cfg['config_id']}__{dataset['dataset_id']}__seed{seed}"
                    if args.format == "tsv":
                        row = [run_tag, cfg["config_id"], dataset["output_dir"], str(seed)] + [cfg[c] for c in HP_COLS]
                        f.write("\t".join(row) + "\n")
                    else:  # dsq: one self-contained command per line
                        vals = [run_tag, dataset["output_dir"], str(seed)] + [cfg[c] for c in HP_COLS]
                        f.write("bash " + args.runner + " " + " ".join(vals) + "\n")
                    n += 1
    kind = "task line(s)" if args.format == "tsv" else "dsq job(s)"
    print(f"{n} {kind} -> {args.out}  "
          f"({len(configs)} configs x {len(datasets)} datasets x {len(args.seeds)} seeds).", file=sys.stderr)
    if args.format == "tsv":
        print(f"Set --array=0-{n-1} in run_hp_search.sbatch.", file=sys.stderr)
    else:
        print(f"Submit: module load dSQ && dsq --job-file {args.out} --partition gpu --account <acct> "
              f"--gpus 1 --cpus-per-task 4 --mem 48G --time 06:00:00 --output logs/%A_%a.out", file=sys.stderr)
    print(n)


if __name__ == "__main__":
    main()
