#!/usr/bin/env python3
"""
aggregate_hp.py — rank hyperparameter configs by DEV AUPRC, aggregated across cells x seeds.

Reads, per task output dir, the best dev auprc the Trainer achieved. Preference order:
  1. trainer_state.json `best_metric` (set when metric_for_best_model=auprc), searching
     <out_dir> and <out_dir>/checkpoint-* (the HF Trainer writes trainer_state.json into each
     checkpoint during training).
  2. fallback: max `eval_auprc` over log_history entries.
Groups by config_id, reports mean/std/min dev auprc across the cells x seeds for that config,
ranks descending, writes hp_results.csv, and prints the winning config + ready-to-use CLI flags.

Usage:
  python aggregate_hp.py --tasks hp/tasks_hp.tsv --runs_base hp/runs --out hp/hp_results.csv
"""
import argparse, csv, glob, json, os, statistics, sys


def best_dev_auprc(out_dir):
    candidates = []
    paths = [os.path.join(out_dir, "trainer_state.json")]
    paths += sorted(glob.glob(os.path.join(out_dir, "checkpoint-*", "trainer_state.json")))
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            st = json.load(open(p))
        except Exception:
            continue
        bm = st.get("best_metric")
        if isinstance(bm, (int, float)):
            candidates.append(float(bm))
        # fallback: scan log_history for eval_auprc
        for e in st.get("log_history", []):
            if "eval_auprc" in e and isinstance(e["eval_auprc"], (int, float)):
                candidates.append(float(e["eval_auprc"]))
    return max(candidates) if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="tasks_hp.tsv from make_hp_task_manifest.py")
    ap.add_argument("--runs_base", required=True, help="base dir holding <run_tag>/ output dirs")
    ap.add_argument("--out", default="hp_results.csv")
    args = ap.parse_args()

    with open(args.tasks) as f:
        tasks = list(csv.DictReader(f, delimiter="\t"))

    hp_cols = ["learning_rate", "batch_size", "epochs", "weight_decay", "warmup_steps", "head_num_layers"]
    per_config = {}   # config_id -> {"hp": {...}, "scores": [...], "missing": int}
    for t in tasks:
        cid = t["config_id"]
        rec = per_config.setdefault(cid, {"hp": {c: t[c] for c in hp_cols}, "scores": [], "missing": 0})
        score = best_dev_auprc(os.path.join(args.runs_base, t["run_tag"]))
        if score is None:
            rec["missing"] += 1
        else:
            rec["scores"].append(score)

    rows = []
    for cid, rec in per_config.items():
        s = rec["scores"]
        rows.append({
            "config_id": cid,
            **rec["hp"],
            "n_runs": len(s),
            "n_missing": rec["missing"],
            "dev_auprc_mean": round(statistics.mean(s), 5) if s else float("nan"),
            "dev_auprc_std": round(statistics.pstdev(s), 5) if len(s) > 1 else 0.0,
            "dev_auprc_min": round(min(s), 5) if s else float("nan"),
        })
    # rank by mean dev auprc (configs with no runs sink to the bottom)
    rows.sort(key=lambda r: (r["dev_auprc_mean"] if r["dev_auprc_mean"] == r["dev_auprc_mean"] else -1),
              reverse=True)

    out_cols = (["config_id"] + hp_cols +
                ["n_runs", "n_missing", "dev_auprc_mean", "dev_auprc_std", "dev_auprc_min"])
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {args.out} ({len(rows)} configs)")
    if rows and rows[0]["dev_auprc_mean"] == rows[0]["dev_auprc_mean"]:
        best = rows[0]
        print(f"\nWINNER: {best['config_id']}  dev_auprc={best['dev_auprc_mean']} "
              f"+/- {best['dev_auprc_std']} (min {best['dev_auprc_min']}, n={best['n_runs']})")
        print("Apply to the full ref_single run (both arms):")
        print(f"  --learning_rate {best['learning_rate']} "
              f"--per_device_train_batch_size {best['batch_size']} "
              f"--num_train_epochs {best['epochs']} --weight_decay {best['weight_decay']} "
              f"--warmup_steps {best['warmup_steps']} --head_num_layers {best['head_num_layers']}")
    else:
        print("No completed runs found — check runs_base and that trainer_state.json exists.", file=sys.stderr)


if __name__ == "__main__":
    main()
