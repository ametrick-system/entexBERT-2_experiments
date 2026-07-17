#!/usr/bin/env python3
"""
show_hp_results.py — read HP-sweep dev-AUPRC results directly from the runs/ tree, WITHOUT needing
the tasks_hp.tsv manifest. Globs <runs_dir>/<run_tag>/[checkpoint-*/]trainer_state.json, extracts the
best dev AUPRC per run, and prints (a) every completed run best-first and (b) a per-config summary
(mean/std/n over the datasets x seeds that completed). run_tag is decoded as
<config_id>__<dataset_id>__seed<seed> (dataset_id may itself contain '__').

Usage:
  python show_hp_results.py --runs_dir runs
  python show_hp_results.py --runs_dir runs --out_csv hp_results.csv
"""
import argparse, csv, glob, json, os, statistics, sys


def decode_run_tag(run_tag):
    parts = run_tag.split("__")
    config_id = parts[0]
    if parts[-1].startswith("seed"):
        seed = parts[-1][4:]
        dataset_id = "__".join(parts[1:-1])
    else:
        seed = ""
        dataset_id = "__".join(parts[1:])
    return config_id, dataset_id, seed


def best_dev_auprc(run_dir):
    """best_metric (preferred) or max eval_auprc in log_history, over top-level + deepest checkpoint."""
    states = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json")),
                    key=lambda p: int(p.split("checkpoint-")[1].split(os.sep)[0]))
    top = os.path.join(run_dir, "trainer_state.json")
    if os.path.exists(top):
        states.append(top)
    best = None
    for p in states:
        try:
            if os.path.getsize(p) == 0:
                continue
            st = json.load(open(p))
        except Exception:
            continue
        bm = st.get("best_metric")
        if isinstance(bm, (int, float)):
            best = bm if best is None else max(best, bm)
        for e in st.get("log_history", []):
            v = e.get("eval_auprc")
            if isinstance(v, (int, float)):
                best = v if best is None else max(best, v)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out_csv", default=None, help="optional: also write a per-run CSV")
    args = ap.parse_args()

    run_dirs = sorted(d for d in glob.glob(os.path.join(args.runs_dir, "*"))
                      if os.path.isdir(d) and not os.path.basename(d).startswith("checkpoint-"))
    if not run_dirs:
        print(f"No run dirs under {args.runs_dir}/", file=sys.stderr); sys.exit(1)

    per_run = []          # (auprc, run_tag, config_id, dataset_id, seed)
    per_config = {}       # config_id -> [auprc,...]
    n_empty = 0
    for d in run_dirs:
        rt = os.path.basename(d)
        cid, ds, seed = decode_run_tag(rt)
        a = best_dev_auprc(d)
        if a is None:
            n_empty += 1
            continue
        per_run.append((a, rt, cid, ds, seed))
        per_config.setdefault(cid, []).append(a)

    per_run.sort(reverse=True)
    print(f"# {len(per_run)} completed runs ({n_empty} dirs had no readable metric) under {args.runs_dir}/\n")
    print("=== per-run (best dev AUPRC, high to low) ===")
    print(f"{'dev_auprc':>9}  {'config':<8} {'dataset':<28} {'seed':>4}")
    for a, rt, cid, ds, seed in per_run:
        print(f"{a:>9.4f}  {cid:<8} {ds:<28} {seed:>4}")

    print("\n=== per-config summary (over completed dataset x seed runs) ===")
    print(f"{'config':<8} {'n':>3} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    rows = []
    for cid, s in per_config.items():
        rows.append((statistics.mean(s), cid, len(s), statistics.pstdev(s) if len(s) > 1 else 0.0,
                     min(s), max(s)))
    for mean, cid, n, std, mn, mx in sorted(rows, reverse=True):
        print(f"{cid:<8} {n:>3} {mean:>8.4f} {std:>8.4f} {mn:>8.4f} {mx:>8.4f}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run_tag", "config_id", "dataset_id", "seed", "dev_auprc"])
            for a, rt, cid, ds, seed in per_run:
                w.writerow([rt, cid, ds, seed, round(a, 5)])
        print(f"\nwrote per-run CSV -> {args.out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
