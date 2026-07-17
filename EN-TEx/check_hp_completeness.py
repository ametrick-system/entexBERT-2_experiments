#!/usr/bin/env python3
"""
check_hp_completeness.py — after a partial/crashed HP dSQ run (e.g. disk filled mid-sweep), figure out
which tasks actually produced a VALID result and re-emit the exact original command lines for the rest,
so you can resubmit just the incomplete ones with dsq.

Why not just `dsqa`? dsqa re-runs tasks SLURM saw fail (nonzero exit / cancelled). But a disk-full crash
can leave a task that exited 0 with a TRUNCATED trainer_state.json — dsqa treats it as done. This tool
validates by parsing each run's trainer_state.json (the same signal aggregate_hp.py uses), so it catches
truncated/zero-byte/missing states too.

A run is COMPLETE iff runs_dir/<run_tag>/ (or a checkpoint-*/ under it) has a trainer_state.json that
json-parses AND carries a numeric best_metric or at least one numeric eval_auprc in log_history.

Usage:
  python check_hp_completeness.py --job_file jobs_hp.txt --runs_dir runs --out_rerun rerun_hp.txt
  # then, if rerun_hp.txt is non-empty:
  #   module load dSQ && dsq --job-file rerun_hp.txt --partition pi_gerstein_gpu --gpus 1 \
  #        --cpus-per-task 4 --mem 48G --time 06:00:00 --output logs/%A_%a.out
"""
import argparse, glob, json, os, shlex, sys


def run_tag_from_line(line):
    """The runner is invoked as `bash run_*_job.sh <run_tag> <data_dir> ...`; run_tag is the arg
    right after the *.sh script token. Falls back to None if no .sh token is found."""
    toks = shlex.split(line)
    for i, t in enumerate(toks):
        if t.endswith(".sh") and i + 1 < len(toks):
            return toks[i + 1]
    return None


def is_complete(run_dir):
    if not os.path.isdir(run_dir):
        return False, "missing_dir"
    states = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json")),
                    key=lambda p: int(p.split("checkpoint-")[1].split(os.sep)[0]))
    top = os.path.join(run_dir, "trainer_state.json")
    if os.path.exists(top):
        states.append(top)
    if not states:
        return False, "no_trainer_state"
    # validate the deepest/newest state
    p = states[-1]
    try:
        if os.path.getsize(p) == 0:
            return False, "zero_byte_state"
        st = json.load(open(p))
    except Exception as e:
        return False, f"corrupt_json({type(e).__name__})"
    bm = st.get("best_metric")
    if isinstance(bm, (int, float)):
        return True, "ok_best_metric"
    for e in st.get("log_history", []):
        if isinstance(e.get("eval_auprc"), (int, float)):
            return True, "ok_eval_auprc"
    return False, "no_metric_in_state"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_file", required=True, help="the dSQ job file that was submitted (one task/line)")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out_rerun", default="rerun_hp.txt")
    args = ap.parse_args()

    lines = [ln.rstrip("\n") for ln in open(args.job_file) if ln.strip() and not ln.lstrip().startswith("#")]
    complete, incomplete, unparsed = [], [], []
    reasons = {}
    for ln in lines:
        rt = run_tag_from_line(ln)
        if rt is None:
            unparsed.append(ln)
            continue
        ok, why = is_complete(os.path.join(args.runs_dir, rt))
        reasons[why] = reasons.get(why, 0) + 1
        (complete if ok else incomplete).append((rt, ln, why))

    with open(args.out_rerun, "w") as f:
        for rt, ln, why in incomplete:
            f.write(ln + "\n")

    print(f"total tasks in job file : {len(lines)}")
    print(f"  complete              : {len(complete)}")
    print(f"  incomplete/redo       : {len(incomplete)}  -> {args.out_rerun}")
    if unparsed:
        print(f"  UNPARSED lines        : {len(unparsed)} (no .sh token found — check job file format)")
    print("  breakdown by reason   :")
    for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"      {why:24s} {n}")
    if incomplete:
        print("  sample of runs to redo:")
        for rt, ln, why in incomplete[:8]:
            print(f"      {why:24s} {rt}")


if __name__ == "__main__":
    main()
