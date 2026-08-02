#!/usr/bin/env python3
"""
score_ctcf_adastra.py — score a trained entexBERT-2 CTCF binding regressor on the
ADASTRA ASB benchmark (Han et al. 2024, Fig 3A) by ref-vs-alt Delta, and report
AUROC alongside the 14 published models.

The model was trained to predict CTCF BINDING signal (BigWig fold-change), never
ASB labels. Here we score each ASB variant by the SIGNED contrast
    Delta = head(alt window) - head(ref window)
(the pipeline's own twin score-and-subtract, matching how the model was trained),
then compute AUROC on |Delta| vs the binary ASB label. The scoring convention
this script implements — score = abs(predicted ref/alt difference), negatives
subsampled to the positive count, fixed seed — is meant to line up with how the
14 models are evaluated in the benchmark, so the numbers are comparable. Before
quoting the result as strictly identical to the published numbers, confirm the
exact evaluation details (negative-subsampling scheme and seed) against the
benchmark's own released code/Methods.

WHAT YOU NEED ON THE CLUSTER (all small — no 246 MB ADASTRA download):
  --checkpoint_dir   the trained regressor: .../runs/reg  (has run_config.json)
  --eval_csv         ctcf_adastra_evalset.csv.gz   (shipped: 9071 pos + 242720 neg)
  --ref_fasta        /home/asm242/reference_genome/hg38.fa
  --reference_csv    ctcf_benchmark_reference_auroc.csv   (shipped: 14-model table)
  [--train_coords]   OPTIONAL the build's meta sidecars — fold0/train.meta.csv AND
                     fold0/dev.meta.csv — to drop eval variants whose 100kb bin was
                     seen in training/validation (leak-free number). NOT test.meta.csv.

Run (on a GPU node, in the eb2 env, from the repo root so `entexbert2` imports):
  python score_ctcf_adastra.py \
      --checkpoint_dir experiments/binding_reg_ENC-001_CTCF_.../runs/reg \
      --eval_csv ctcf_adastra_evalset.csv.gz \
      --ref_fasta /home/asm242/reference_genome/hg38.fa \
      --reference_csv ctcf_benchmark_reference_auroc.csv \
      --out ctcf_adastra_entexbert2_result
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from sklearn.metrics import roc_auc_score, average_precision_score

# The pipeline's own inference primitive (loads run_config.json, tokenizes,
# batches, twin score-and-subtract for pair inputs). Import from the installed repo.
from entexbert2.analyze import run_inference


# ----------------------------------------------------------------------
# Window construction: extract the ref window and the alt window (alt base
# substituted at the SNV center) from hg38, matching the training window.
# ----------------------------------------------------------------------
def build_windows(eval_df, ref_fasta, left_bp, right_bp):
    """Return (ref_seqs, alt_seqs, keep_mask). pos is 1-based (ADASTRA/VCF)."""
    fa = Fasta(ref_fasta, sequence_always_upper=True)
    win = left_bp + 1 + right_bp
    ref_seqs, alt_seqs, keep = [], [], []
    n_oob = n_refmismatch = n_badchrom = 0
    for chrom, pos1, ref_a, alt_a in zip(
        eval_df["chr"], eval_df["pos"], eval_df["ref"], eval_df["alt"]
    ):
        if chrom not in fa:
            keep.append(False); ref_seqs.append(""); alt_seqs.append("")
            n_badchrom += 1; continue
        p0 = int(pos1) - 1                 # 0-based SNV position
        start = p0 - left_bp
        end = p0 + right_bp + 1            # half-open; length = win
        clen = len(fa[chrom])
        if start < 0 or end > clen:
            keep.append(False); ref_seqs.append(""); alt_seqs.append("")
            n_oob += 1; continue
        seq = str(fa[chrom][start:end])
        if len(seq) != win:
            keep.append(False); ref_seqs.append(""); alt_seqs.append("")
            n_oob += 1; continue
        center = left_bp                    # SNV offset within the window
        # sanity: hg38 base at center should equal the ADASTRA ref allele
        if seq[center] != str(ref_a).upper():
            n_refmismatch += 1              # keep but count; ADASTRA ref is hg38 by construction
        alt_seq = seq[:center] + str(alt_a).upper() + seq[center + 1:]
        ref_seqs.append(seq); alt_seqs.append(alt_seq); keep.append(True)
    print(f"[windows] built {sum(keep)}/{len(eval_df)}  "
          f"(dropped: {n_oob} out-of-bounds, {n_badchrom} bad-chrom; "
          f"ref-base!=hg38 on {n_refmismatch} kept rows)")
    return ref_seqs, alt_seqs, np.array(keep, dtype=bool)


# ----------------------------------------------------------------------
# Balanced AUROC exactly as the paper: subsample negatives to #positives,
# seed=1, AUROC on the score. Report a bootstrap CI too (the paper reports a
# single point; the CI tells you whether the difference from a model is real).
# ----------------------------------------------------------------------
def balanced_auroc(score, label, seed=1, n_boot=1000):
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=int)
    pos_idx = np.where(label == 1)[0]
    neg_idx = np.where(label == 0)[0]
    m = len(pos_idx)
    rng = np.random.default_rng(seed)
    sel_neg = rng.choice(neg_idx, size=m, replace=False)
    idx = np.concatenate([pos_idx, sel_neg])
    y = label[idx]; s = score[idx]
    point = roc_auc_score(y, s)
    aupr = average_precision_score(y, s)
    # bootstrap over the balanced set
    boots = []
    for _ in range(n_boot):
        b = rng.integers(0, len(idx), len(idx))
        if len(np.unique(y[b])) < 2:
            continue
        boots.append(roc_auc_score(y[b], s[b]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return point, aupr, (lo, hi), m


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint_dir", required=True,
                    help=".../runs/reg (the trained regressor; has run_config.json)")
    ap.add_argument("--eval_csv", required=True, help="ctcf_adastra_evalset.csv.gz")
    ap.add_argument("--ref_fasta", required=True, help="hg38.fa")
    ap.add_argument("--reference_csv", default=None,
                    help="ctcf_benchmark_reference_auroc.csv (14-model table)")
    ap.add_argument("--train_coords", default=None, nargs="+",
                    help="OPTIONAL one or more meta files from the build "
                         "(fold0/train.meta.csv fold0/dev.meta.csv). Their (chr, bin) "
                         "pairs form the SEEN set; ADASTRA variants in any seen bin are "
                         "flagged as leaky. Pass BOTH train and dev for a truly leak-free "
                         "number (the model selected its checkpoint on dev). Do NOT pass "
                         "test.meta.csv — those bins are held out and fine to score.")
    ap.add_argument("--bin_size", type=int, default=100000,
                    help="must match the training config partition bin_size")
    ap.add_argument("--drop_leaky", action="store_true",
                    help="drop eval variants whose bin is in the training set "
                         "(default: keep + report both numbers)")
    ap.add_argument("--left_bp", type=int, default=128)
    ap.add_argument("--right_bp", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overrides", default=None,
                    help="JSON dict of run_config overrides (rarely needed)")
    ap.add_argument("--out", default="ctcf_adastra_entexbert2_result")
    args = ap.parse_args()

    overrides = json.loads(args.overrides) if args.overrides else {}

    ev = pd.read_csv(args.eval_csv)
    print(f"[load] eval set: {len(ev)} rows  "
          f"(pos={int((ev.label==1).sum())}, neg={int((ev.label==0).sum())})")

    # 1) build ref/alt windows
    ref_seqs, alt_seqs, keep = build_windows(ev, args.ref_fasta, args.left_bp, args.right_bp)
    ev = ev.loc[keep].reset_index(drop=True)
    pairs = [[r, a] for r, a in zip(np.asarray(ref_seqs)[keep], np.asarray(alt_seqs)[keep])]

    # 2) score by twin Delta = head(alt) - head(ref)  (pair mode)
    print(f"[score] running inference on {len(pairs)} variants (pair mode, twin Delta)...")
    logits, _emb, run_config = run_inference(
        args.checkpoint_dir, pairs, args.batch_size, args.device, overrides)
    delta = np.asarray(logits, dtype=float).reshape(len(pairs), -1)[:, 0]
    ev["delta"] = delta
    ev["abs_delta"] = np.abs(delta)

    # 3) optional leakage audit: collect the SEEN (chr, bin) set from every meta
    #    file passed (train + dev), then flag ADASTRA variants in any seen bin.
    leaky_mask = np.zeros(len(ev), dtype=bool)
    coord_files = args.train_coords or []
    seen_bins = set()
    for path in coord_files:
        if not os.path.exists(path):
            print(f"[leakage] WARNING: {path} not found; skipping.")
            continue
        tc = pd.read_csv(path)
        chrom_col = "chr" if "chr" in tc.columns else tc.columns[0]
        pos_col = ("SNV" if "SNV" in tc.columns
                   else "pos" if "pos" in tc.columns
                   else "anchor" if "anchor" in tc.columns else None)
        if pos_col is None:
            print(f"[leakage] {path}: no SNV/pos/anchor column ({list(tc.columns)[:6]}...); "
                  f"skipping this file.")
            continue
        before = len(seen_bins)
        seen_bins |= set(zip(tc[chrom_col].astype(str),
                             (tc[pos_col].astype(int) // args.bin_size)))
        print(f"[leakage] {os.path.basename(path)}: +{len(seen_bins)-before} bins "
              f"(via '{pos_col}'), {len(seen_bins)} seen total")
    if seen_bins:
        # ev['pos'] is 1-based -> match the build's SNV (0-based anchor) binning
        ev_bins = list(zip(ev["chr"].astype(str),
                           ((ev["pos"].astype(int) - 1) // args.bin_size)))
        leaky_mask = np.array([b in seen_bins for b in ev_bins])
        n_pos_leak = int(leaky_mask[ev["label"].to_numpy() == 1].sum())
        print(f"[leakage] {leaky_mask.sum()}/{len(ev)} eval variants "
              f"({100*leaky_mask.mean():.2f}%) fall in a seen (train/dev) bin; "
              f"{n_pos_leak} of them are ASB-positive. These are DROPPED for leak_free.")
    else:
        print("[leakage] no usable --train_coords given; reporting the full-set number only. "
              "For a leak-free number, pass fold0/train.meta.csv fold0/dev.meta.csv "
              "(NOT test.meta.csv).")

    # 4) AUROC — full set, and leak-free subset if we have the audit
    def report(tag, sub):
        pt, aupr, (lo, hi), m = balanced_auroc(sub["abs_delta"].to_numpy(),
                                               sub["label"].to_numpy())
        print(f"[AUROC:{tag}] balanced on {m} pos + {m} neg  "
              f"AUROC={pt:.4f}  95%CI[{lo:.4f},{hi:.4f}]  AUPRC={aupr:.4f}")
        return {"regime": tag, "auroc": pt, "auroc_lo": lo, "auroc_hi": hi,
                "auprc": aupr, "n_pos": int(m)}

    results = [report("full", ev)]
    if leaky_mask.any():
        results.append(report("leak_free", ev.loc[~leaky_mask]))
        if args.drop_leaky:
            print("[note] --drop_leaky set: the leak_free number is the headline.")

    # 5) place against the 14 models
    if args.reference_csv and os.path.exists(args.reference_csv):
        ref = pd.read_csv(args.reference_csv).sort_values("CTCF_AUROC", ascending=False)
        eb2 = results[-1]["auroc"]  # leak_free if present else full
        print("\n=== entexBERT-2 vs the 14 models (CTCF AUROC) ===")
        placed = False
        for _, r in ref.iterrows():
            if not placed and eb2 >= r["CTCF_AUROC"]:
                print(f"  >>> entexBERT-2 (this run)   {eb2:.4f}  <<<")
                placed = True
            print(f"      {r['model']:20s} {r['family']:9s} {r['CTCF_AUROC']:.4f}")
        if not placed:
            print(f"  >>> entexBERT-2 (this run)   {eb2:.4f}  (below all listed) <<<")

    # 6) save
    ev[["chr", "pos", "ref", "alt", "snp", "label", "delta", "abs_delta"]].to_csv(
        f"{args.out}_perVariant.csv.gz", index=False, compression="gzip")
    with open(f"{args.out}_metrics.json", "w") as f:
        json.dump({"results": results,
                   "checkpoint_dir": args.checkpoint_dir,
                   "run_config_task": run_config.get("task"),
                   "n_scored": int(len(ev)),
                   "left_bp": args.left_bp, "right_bp": args.right_bp,
                   "leaky_excluded": bool(leaky_mask.any())}, f, indent=2)
    print(f"\n[done] wrote {args.out}_metrics.json + {args.out}_perVariant.csv.gz")


if __name__ == "__main__":
    main()
