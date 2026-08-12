#!/usr/bin/env python3
"""
score_hetsnv.py — score a trained entexBERT-2 CTCF binding regressor on the EN-TEx
het-SNV ASB set by ref/alt twin Delta, reported PER DONOR and PER TISSUE.

Why this eval (vs ADASTRA): the EN-TEx hetSNV table keeps the TISSUE dimension.
`imbalance_significance` is a per-(SNV, tissue) ASB call, and the table carries a
continuous allelic effect size (signed_log_count_ratio) — so unlike ADASTRA we can
(a) break AUROC down by tissue and (b) measure real effect-size calibration
Spearman(|Delta|, |signed_log_count_ratio|), not just a binary-label AUROC.

The model predicts CTCF BINDING signal (BigWig fold-change), never ASB. We score
each variant by the SIGNED twin contrast
    Delta = head(alt window) - head(ref window)
(the same score-and-subtract the model was trained under), then AUROC on |Delta| vs
the binary imbalance_significance label, balanced (negatives subsampled to #positives,
fixed seed) exactly as score_ctcf_adastra.py does, for cross-comparability.

hetSNV table columns used (see load_as_table in utils.py):
  chr, ref_start(0-based SNV pos), ref_end, ref_allele, hap1_allele, hap2_allele,
  donor, tissue, assay, cA,cC,cG,cT, ref_allele_ratio, p_betabinom, imbalance_significance
Derived here: hap1_count/hap2_count (from cA..cT via the hap alleles), total_reads,
signed_log_count_ratio = log2((hap1_count+0.5)/(hap2_count+0.5)).

LEAKAGE: the binding model trained on ENC-002 peaks. Pass --train_coords
fold0/train.meta.csv fold0/dev.meta.csv to flag/drop hetSNV variants whose 100kb bin
was seen in training (only affects the matched donor, ENC-002).

Cluster inputs (all small):
  --checkpoint_dir   trained regressor .../runs/reg  (has run_config.json)
  --hetsnv_tsv       /home/asm242/entex_data/hetSNVs.tsv
  --ref_fasta        /home/asm242/reference_genome/hg38.fa
Run from repo root in the eb2 env (needs entexbert2 + pyfaidx + sklearn + scipy).
"""
import argparse, os, json
import numpy as np, pandas as pd
from pyfaidx import Fasta
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
from entexbert2.analyze import run_inference

_BASECOL = {"A": "cA", "C": "cC", "G": "cG", "T": "cT"}


def build_windows(eval_df, ref_fasta, left_bp, right_bp):
    """Return (ref_seqs, alt_seqs, keep_mask). SNV pos is 0-based (ref_start).
    For a hetSNV the two 'alleles' are hap1 vs hap2; we place hap1 in the ref
    window and hap2 in the alt window so Delta = head(hap2) - head(hap1)."""
    fa = Fasta(ref_fasta, sequence_always_upper=True)
    win = left_bp + 1 + right_bp
    ref_seqs, alt_seqs, keep = [], [], []
    n_oob = n_badchrom = n_refmismatch = 0
    for chrom, p0, ref_a, h1, h2 in zip(
        eval_df["chr"], eval_df["ref_start"], eval_df["ref_allele"],
        eval_df["hap1_allele"], eval_df["hap2_allele"]
    ):
        if chrom not in fa:
            keep.append(False); ref_seqs.append(""); alt_seqs.append("")
            n_badchrom += 1; continue
        p0 = int(p0)
        start = p0 - left_bp
        end = p0 + right_bp + 1
        clen = len(fa[chrom])
        if start < 0 or end > clen:
            keep.append(False); ref_seqs.append(""); alt_seqs.append("")
            n_oob += 1; continue
        seq = str(fa[chrom][start:end])
        if len(seq) != win:
            keep.append(False); ref_seqs.append(""); alt_seqs.append("")
            n_oob += 1; continue
        center = left_bp
        if seq[center] != str(ref_a).upper():
            n_refmismatch += 1
        # hap1 window = reference base -> hap1 allele; hap2 window = hap2 allele
        h1_seq = seq[:center] + str(h1).upper() + seq[center + 1:]
        h2_seq = seq[:center] + str(h2).upper() + seq[center + 1:]
        ref_seqs.append(h1_seq); alt_seqs.append(h2_seq); keep.append(True)
    print(f"[windows] built {sum(keep)}/{len(eval_df)}  "
          f"(dropped: {n_oob} oob, {n_badchrom} bad-chrom; "
          f"hg38-base!=ref_allele on {n_refmismatch} kept rows)")
    return ref_seqs, alt_seqs, np.array(keep, dtype=bool)


def balanced_auroc(score, label, seed=1, n_boot=1000):
    score = np.asarray(score, dtype=float); label = np.asarray(label, dtype=int)
    pos_idx = np.where(label == 1)[0]; neg_idx = np.where(label == 0)[0]
    m = min(len(pos_idx), len(neg_idx))
    if m < 10 or len(pos_idx) == 0 or len(neg_idx) == 0:
        return np.nan, np.nan, (np.nan, np.nan), m
    rng = np.random.default_rng(seed)
    # subsample the LARGER class down to the smaller so the set is balanced
    if len(neg_idx) >= len(pos_idx):
        sel_neg = rng.choice(neg_idx, size=m, replace=False); sel_pos = pos_idx
    else:
        sel_pos = rng.choice(pos_idx, size=m, replace=False); sel_neg = neg_idx
    idx = np.concatenate([sel_pos, sel_neg])
    y = label[idx]; sc = score[idx]
    point = roc_auc_score(y, sc); aupr = average_precision_score(y, sc)
    boots = []
    for _ in range(n_boot):
        b = rng.integers(0, len(idx), len(idx))
        if len(np.unique(y[b])) < 2:
            continue
        boots.append(roc_auc_score(y[b], sc[b]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return point, aupr, (lo, hi), m


def load_hetsnv(path, assay, min_total_reads):
    """Load the EN-TEx hetSNV TSV, derive counts + effect size, return a tidy frame.
    Keeps ALL donors/tissues (filtering by donor/tissue happens in the caller)."""
    usecols = ["chr", "ref_start", "ref_end", "ref_allele", "hap1_allele",
               "hap2_allele", "donor", "tissue", "assay",
               "cA", "cC", "cG", "cT",
               "ref_allele_ratio", "p_betabinom", "imbalance_significance"]
    df = pd.read_csv(path, sep="\t", usecols=lambda c: c in usecols)
    # assay filter (CTCF); assay strings in the table look like 'TF-ChIP-seq_CTCF'
    if assay and assay.upper() != "ALL":
        df = df[df["assay"].astype(str).str.contains(assay, case=False, na=False)]
    # derive hap counts from the per-base columns
    def base_count(row, allele_col):
        col = _BASECOL.get(str(row[allele_col]).upper())
        return float(row[col]) if col in row and pd.notna(row[col]) else 0.0
    df = df.reset_index(drop=True)
    df["hap1_count"] = df.apply(lambda r: base_count(r, "hap1_allele"), axis=1)
    df["hap2_count"] = df.apply(lambda r: base_count(r, "hap2_allele"), axis=1)
    df["total_reads"] = df["hap1_count"] + df["hap2_count"]
    df["signed_log_count_ratio"] = np.log2(
        (df["hap1_count"] + 0.5) / (df["hap2_count"] + 0.5))
    if min_total_reads:
        n0 = len(df)
        df = df[df["total_reads"] >= min_total_reads].reset_index(drop=True)
        print(f"[filter] total_reads>={min_total_reads}: {len(df)}/{n0} rows kept")
    df["label"] = df["imbalance_significance"].astype(int)
    return df


def seen_bins_from_meta(coord_files, bin_size):
    seen = set()
    for path in coord_files or []:
        if not os.path.exists(path):
            print(f"[leakage] WARNING: {path} not found; skipping."); continue
        tc = pd.read_csv(path)
        chrom_col = "chr" if "chr" in tc.columns else tc.columns[0]
        pos_col = ("SNV" if "SNV" in tc.columns else "pos" if "pos" in tc.columns
                   else "anchor" if "anchor" in tc.columns else None)
        if pos_col is None:
            print(f"[leakage] {path}: no SNV/pos/anchor col; skipping."); continue
        before = len(seen)
        seen |= set(zip(tc[chrom_col].astype(str),
                        (tc[pos_col].astype(int) // bin_size)))
        print(f"[leakage] {os.path.basename(path)}: +{len(seen)-before} bins, "
              f"{len(seen)} total")
    return seen


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--hetsnv_tsv", required=True)
    ap.add_argument("--ref_fasta", required=True)
    ap.add_argument("--assay", default="CTCF")
    ap.add_argument("--donors", nargs="+",
                    default=["ENC-001", "ENC-002", "ENC-003", "ENC-004"],
                    help="donors to score, reported separately")
    ap.add_argument("--matched_donor", default="ENC-002",
                    help="the donor the binding model trained on; LABELS rows matched vs "
                         "cross-donor. The seen-bin leak filter applies to ALL donors "
                         "(sequences are reference-based, so a trained 100kb bin is leaked "
                         "at that locus regardless of which donor is scored there).")
    ap.add_argument("--min_total_reads", type=int, default=20,
                    help="read-count floor; try 30 for a stricter labelset")
    ap.add_argument("--min_tissue_pos", type=int, default=20,
                    help="skip a per-tissue AUROC if it has fewer than this many positives")
    ap.add_argument("--train_coords", nargs="+", default=None,
                    help="fold0/train.meta.csv fold0/dev.meta.csv (matched-donor leak filter)")
    ap.add_argument("--bin_size", type=int, default=100000)
    ap.add_argument("--drop_leaky", action="store_true")
    ap.add_argument("--left_bp", type=int, default=128)
    ap.add_argument("--right_bp", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--out", default="hetsnv_entexbert2_result")
    args = ap.parse_args()
    overrides = json.loads(args.overrides) if args.overrides else {}

    full = load_hetsnv(args.hetsnv_tsv, args.assay, args.min_total_reads)
    print(f"[load] {len(full)} hetSNV rows over donors={sorted(full.donor.unique())}")

    seen = seen_bins_from_meta(args.train_coords, args.bin_size)
    all_rows = []
    summary = []
    for donor in args.donors:
        d = full[full["donor"] == donor].reset_index(drop=True)
        if len(d) == 0:
            print(f"\n[{donor}] no rows; skipping."); continue
        print(f"\n===== donor {donor}: {len(d)} rows "
              f"(pos={int((d.label==1).sum())}, neg={int((d.label==0).sum())}) =====")
        ref_seqs, alt_seqs, keep = build_windows(d, args.ref_fasta,
                                                 args.left_bp, args.right_bp)
        d = d.loc[keep].reset_index(drop=True)
        pairs = [[r, a] for r, a in zip(np.asarray(ref_seqs)[keep],
                                        np.asarray(alt_seqs)[keep])]
        logits, _emb, _cfg = run_inference(
            args.checkpoint_dir, pairs, args.batch_size, args.device, overrides)
        arr = np.asarray(logits, dtype=float).reshape(len(pairs), -1)
        delta = arr[:, 0]
        d["delta"] = delta; d["abs_delta"] = np.abs(delta)
        if arr.shape[1] >= 2:                       # test-time sigma head present
            sigma = np.exp(0.5 * arr[:, 1])
            d["sigma"] = sigma
            d["zscore"] = np.abs(delta) / (sigma + 1e-6)   # calibrated statistic |Delta|/sigma
        d["donor"] = donor

        # Leak filter applies to ALL donors. Sequences are reference-based (hg38):
        # a hetSNV window is identical across donors at a given locus, so any variant
        # whose 100kb bin was in training is leaked regardless of donor. The trained
        # regions are GENOMIC (ENC-002 peaks/background at fixed bins); scoring a
        # cross-donor variant there is still scoring a memorized region.
        # matched_donor is now only a LABEL (matched vs cross-donor), not a filter gate.
        leaky = np.zeros(len(d), dtype=bool)
        if seen:
            dbins = list(zip(d["chr"].astype(str),
                             (d["ref_start"].astype(int) // args.bin_size)))
            leaky = np.array([b in seen for b in dbins])
            kind = "matched" if donor == args.matched_donor else "cross-donor"
            print(f"[leakage] {leaky.sum()}/{len(d)} {donor} ({kind}) variants in a "
                  f"seen bin ({int(leaky[d.label.to_numpy()==1].sum())} positive).")
        d["leaky"] = leaky
        d["donor_kind"] = "matched" if donor == args.matched_donor else "cross-donor"

        def rep(tag, sub):
            pt, aupr, (lo, hi), m = balanced_auroc(sub["abs_delta"].to_numpy(),
                                                   sub["label"].to_numpy())
            mag = spearmanr(sub["abs_delta"], sub["signed_log_count_ratio"].abs()
                            ).correlation if len(sub) > 10 else np.nan
            sgn = spearmanr(sub["delta"], sub["signed_log_count_ratio"]
                            ).correlation if len(sub) > 10 else np.nan
            print(f"  [{donor}:{tag}] AUROC={pt:.4f} CI[{lo:.4f},{hi:.4f}] "
                  f"AUPRC={aupr:.4f} n_pos={m} | mag_Spearman={mag:.4f} "
                  f"signed_Spearman={sgn:.4f}")
            z_au = z_mag = np.nan
            if "zscore" in sub.columns:
                z_au = balanced_auroc(sub["zscore"].to_numpy(),
                                      sub["label"].to_numpy())[0]
                z_mag = (spearmanr(sub["zscore"],
                                   sub["signed_log_count_ratio"].abs()).correlation
                         if len(sub) > 10 else np.nan)
                print(f"  [{donor}:{tag}] SIGMA  AUROC(|d|/sigma)={z_au:.4f}  "
                      f"mag_Spearman(z)={z_mag:.4f}  (vs |Delta| above)")
            summary.append(dict(donor=donor,
                                donor_kind=("matched" if donor == args.matched_donor
                                            else "cross-donor"),
                                regime=tag, tissue="ALL",
                                auroc=pt, auroc_lo=lo, auroc_hi=hi, auprc=aupr,
                                n_pos=m, mag_spearman=mag, signed_spearman=sgn,
                                auroc_z=z_au, mag_spearman_z=z_mag))
        rep("full", d)
        if leaky.any():
            rep("leak_free", d.loc[~leaky])

        # per-tissue breakdown (tissue is what ADASTRA could not do). Runs on the
        # LEAK-FREE subset when a seen-bin set was given, so per-tissue numbers are
        # not inflated by memorized bins; full set otherwise.
        d_tis = d.loc[~leaky] if leaky.any() else d
        tis_regime = "leak_free" if leaky.any() else "full"
        print(f"  --- per-tissue [{tis_regime}] (>= {args.min_tissue_pos} pos & neg) ---")
        for tis, sub in d_tis.groupby("tissue"):
            npos = int((sub.label == 1).sum()); nneg = int((sub.label == 0).sum())
            if npos < args.min_tissue_pos or nneg < args.min_tissue_pos:
                continue
            pt, aupr, (lo, hi), m = balanced_auroc(sub["abs_delta"].to_numpy(),
                                                   sub["label"].to_numpy())
            mag = spearmanr(sub["abs_delta"], sub["signed_log_count_ratio"].abs()
                            ).correlation
            print(f"    {tis:32s} AUROC={pt:.4f} n_pos={m} mag_Sp={mag:.4f}")
            summary.append(dict(donor=donor,
                                donor_kind=("matched" if donor == args.matched_donor
                                            else "cross-donor"),
                                regime=tis_regime, tissue=tis,
                                auroc=pt, auroc_lo=lo, auroc_hi=hi, auprc=aupr,
                                n_pos=m, mag_spearman=mag, signed_spearman=np.nan))
        all_rows.append(d)

    if all_rows:
        alld = pd.concat(all_rows, ignore_index=True)
        pv = args.out + "_perVariant.csv.gz"
        pv_cols = ["chr", "ref_start", "ref_allele", "hap1_allele", "hap2_allele",
                   "donor", "donor_kind", "tissue", "label", "total_reads",
                   "signed_log_count_ratio", "delta", "abs_delta", "leaky"]
        if "sigma" in alld.columns:
            pv_cols += ["sigma", "zscore"]
        alld[pv_cols].to_csv(pv, index=False)
        print(f"\n[write] {pv}  ({len(alld)} scored variants)")
    sm = pd.DataFrame(summary)
    smf = args.out + "_summary.csv"
    sm.to_csv(smf, index=False)
    print(f"[write] {smf}")
    print("\n=== donor-level (ALL-tissue) summary ===")
    show = sm[sm.tissue == "ALL"]
    if len(show):
        print(show[["donor", "donor_kind", "regime", "auroc", "n_pos",
                    "mag_spearman", "signed_spearman"]].to_string(index=False))


if __name__ == "__main__":
    main()
