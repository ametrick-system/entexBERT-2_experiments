#!/usr/bin/env python3
"""
build_eb2_crossdonor_pool.py -- pool the entexBERT-2 cross-donor test set for the
leak-free head-to-head, then evaluate it against the ENC-001 checkpoint via analyze.py.

DESIGN (mirrors the DNABERT-1 cross-donor logic, using entexBERT-2's own pipeline):
generate_all_inputs builds per-donor fold0/{train,dev,test}.csv with the SAME
partition salt, so the global 100kb bin->split map is IDENTICAL across donors:
a bin that is 'test' for ENC-002/3/4 is also 'test' (never 'train') for ENC-001.
Thus each other-donor fold0/test.csv is already in globally-held-out bins,
disjoint from ENC-001's train by construction. We pool them and, belt-and-suspenders,
DROP any row whose locus_id appears in ENC-001's train.meta.csv.

Run generate_all_inputs FIRST with a datasets_csv listing ENC-002/3/4 CTCF and the
SAME base config used for ENC-001 (same salt). Then this script pools + cross-checks.
Evaluate with analyze.py --data_csv <pooled test.csv> --checkpoint_dir <ENC-001 runs/clf>.
"""
import argparse, os, glob
import numpy as np, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs_root", required=True,
                    help="base output_dir passed to generate_all_inputs for the cross-donor build")
    ap.add_argument("--donors", default="ENC-002,ENC-003,ENC-004")
    ap.add_argument("--assay", default="TF-ChIP-seq_CTCF")
    ap.add_argument("--enc001_train_meta", required=True,
                    help="ENC-001 fold0/train.meta.csv -> locus_ids to exclude (belt-and-suspenders)")
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--balance", action="store_true", default=True,
                    help="downsample negs to pos in the pooled set (match a balanced-eval comparison)")
    a = ap.parse_args()

    train_loci = set(pd.read_csv(a.enc001_train_meta, usecols=["locus_id"])["locus_id"])
    print(f"[exclude] {len(train_loci)} ENC-001 train loci")

    frames = []
    for donor in [d.strip() for d in a.donors.split(",") if d.strip()]:
        dsdir = os.path.join(a.inputs_root, f"{donor}__{a.assay}".replace(" ", "_"), a.fold)
        meta = os.path.join(dsdir, "test.meta.csv")
        if not os.path.exists(meta):
            raise SystemExit(f"MISSING {meta} -- run generate_all_inputs for {donor} first")
        m = pd.read_csv(meta)
        m["donor"] = donor
        frames.append(m)
        print(f"[load] {donor}: {len(m)} test rows  pos={int((m['label']==1).sum())}")
    pool = pd.concat(frames, ignore_index=True)

    # belt-and-suspenders: drop any locus that is in ENC-001 train (should already be ~0)
    before = len(pool)
    n_in_train = int(pool["locus_id"].isin(train_loci).sum())
    pool = pool[~pool["locus_id"].isin(train_loci)].copy()
    print(f"[leak-check] dropped {n_in_train}/{before} rows whose locus is in ENC-001 train "
          f"(expect ~0: same-salt bins are disjoint by construction)")

    if a.balance:
        pos = pool[pool.label == 1]; neg = pool[pool.label == 0]; n = min(len(pos), len(neg))
        pool = pd.concat([pos.sample(n=n, random_state=23), neg.sample(n=n, random_state=23)])
        print(f"[balance] pooled cross-donor test n={len(pool)} pos_frac={pool.label.mean():.4f}")

    os.makedirs(a.out_dir, exist_ok=True)
    pool = pool.sample(frac=1, random_state=23)
    # analyze.py reads a 'sequence' + 'label' csv (single-window); write test.csv + full meta
    seq_col = "sequence" if "sequence" in pool.columns else "sequence1"
    pool[[seq_col, "label"]].rename(columns={seq_col: "sequence"}).to_csv(
        os.path.join(a.out_dir, "test.csv"), index=False)
    pool.to_csv(os.path.join(a.out_dir, "test.meta.csv"), index=False)
    print(f"[done] wrote test.csv (n={len(pool)}) + test.meta.csv to {a.out_dir}")
    print(f"       eval: python -m entexbert2.analyze --checkpoint_dir <ENC-001 runs/clf> "
          f"--data_csv {a.out_dir}/test.csv --output_dir {a.out_dir}/eval")

if __name__ == "__main__":
    main()
