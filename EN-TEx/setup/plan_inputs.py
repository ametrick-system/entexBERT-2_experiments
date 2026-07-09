#!/usr/bin/env python3
"""
plan_inputs.py — input-planning pass over the raw EN-TEx hetSNVs TSV, run BEFORE building inputs.

Experiment unit (matches the training pipeline): ONE MODEL = one donor x one assay, with ALL
TISSUES POOLED. That is exactly what load_as_table(assay, donor, tissue=None) returns, and it is
the default "dataset" here: (donor, assay), tissues pooled. (--group_tissue is the opt-in to split by
tissue instead; leave it off for the all-tissues-pooled experiments.)

Typical use: scope to ONE transcription factor / assay and get one dataset PER DONOR ---

  python plan_inputs.py --tsv hetSNVs.tsv --tf CTCF --min_total_reads 10 \
      --suggest_test_frac 0.10 --out_prefix input_plan/ctcf/ctcf

  --tf CTCF resolves (case-insensitive substring) to the EN-TEx assay 'TF-ChIP-seq_CTCF' and scopes
  the ENTIRE pass to it, so: (a) the per-dataset table lists one row per donor for that TF, (b) the
  <prefix>_datasets.csv drops straight into build_inputs.sh (one model per donor, tissues pooled), and
  (c) the held-out-chromosome suggestion is computed on THAT TF's positive distribution, not pooled
  across all assays. Use --assay for the exact assay string, or neither to survey every assay.

It answers the two questions the hybrid genomic-bin partition needs up front:
  1. NaN p_betabinom fraction per dataset (rows the exclude policy DROPS).
  2. Per-chromosome POSITIVE counts -> a held-out TEST chromosome set (--suggest_test_frac).

Definitions (match the pipeline exactly):
  * positive  = imbalance_significance == 1
  * tested    = p_betabinom is NOT NaN   (drop_aux_nan excludes untested rows)
  * depth     = total_reads (c<hap1_allele> + c<hap2_allele>) >= --min_total_reads
  * chrom key = 'chr'; position anchor = 'ref_start' (== assign_split_column's SNV anchor)
Reads the (large) TSV in chunks; no genome file or model needed.
"""

import argparse
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

BASES = ["A", "C", "G", "T"]

USECOLS = ["chr", "ref_start", "donor", "assay", "tissue",
           "hap1_allele", "hap2_allele", "cA", "cC", "cG", "cT",
           "p_betabinom", "imbalance_significance"]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tsv", required=True, help="Path to the raw hetSNVs TSV.")
    p.add_argument("--out_prefix", default=None,
                   help="If set, write <prefix>_datasets.csv and <prefix>_per_chrom.csv.")
    p.add_argument("--label_col", default="imbalance_significance")
    p.add_argument("--p_col", default="p_betabinom")
    p.add_argument("--min_total_reads", type=int, default=None,
                   help="Read-depth filter (matches the pipeline). Counts reported AFTER it.")
    p.add_argument("--group_tissue", action="store_true",
                   help="Split datasets by tissue too (default: pool tissues within donor+assay).")
    # --- scoping: --tf is the friendly per-TF interface; --assay for an exact name; --donors to subset ---
    p.add_argument("--tf", default=None,
                   help="Scope to one TF/assay by case-insensitive substring match against the "
                        "assay names present in the TSV (e.g. 'CTCF' -> 'TF-ChIP-seq_CTCF'). "
                        "Errors and lists matches if the token is ambiguous.")
    p.add_argument("--assay", default=None,
                   help="Scope to one EXACT assay string (alternative to --tf).")
    p.add_argument("--donors", default=None, help="Comma-separated donors to keep (default: all).")
    # --- held-out test-set suggestion ---
    p.add_argument("--suggest_test_frac", type=float, default=None,
                   help="Suggest a held-out TEST chromosome set carrying ~this fraction of "
                        "positives (e.g. 0.10) and print a paste-ready fold_assignment dict.")
    p.add_argument("--exclude_from_test", default="chr1,chr2,chrX,chrY,chrM,chrMT",
                   help="Chromosomes never suggested for the held-out test set (too big / special).")
    p.add_argument("--test_n_chroms", default="2,3",
                   help="Comma-separated set sizes to consider for the held-out set (default 2,3): "
                        "spreading the test set over a few mid-size chromosomes avoids any single "
                        "chromosome's GC/gene-density dominating the estimate.")
    p.add_argument("--max_single_chrom_frac", type=float, default=0.07,
                   help="Reject any held-out set containing a chromosome that alone carries more "
                        "than this fraction of positives (keeps individual holdouts from "
                        "dominating; default 0.07).")
    p.add_argument("--chunksize", type=int, default=200000)
    return p.parse_args()


def resolve_assay(tsv, tf_token, chunksize):
    """Resolve a --tf substring token to a single assay name present in the TSV."""
    seen = set()
    for chunk in pd.read_csv(tsv, sep="\t", usecols=["assay"], chunksize=chunksize):
        seen.update(chunk["assay"].dropna().astype(str).unique().tolist())
    tok = tf_token.strip().lower()
    matches = sorted(a for a in seen if tok in a.lower())
    if not matches:
        raise SystemExit(f"--tf {tf_token!r}: no assay in the TSV matches. Present assays:\n  "
                         + "\n  ".join(sorted(seen)))
    if len(matches) > 1:
        raise SystemExit(f"--tf {tf_token!r} is ambiguous, matches {len(matches)}:\n  "
                         + "\n  ".join(matches) + "\nUse --assay for the exact name.")
    return matches[0]


def hap_read_counts(df, allele_col):
    """Vectorized: for each row, the cA/cC/cG/cT count matching the allele in allele_col."""
    out = np.zeros(len(df), dtype=float)
    alleles = df[allele_col].astype(str).str.upper().to_numpy()
    for b in BASES:
        col = f"c{b}"
        if col not in df.columns:
            raise ValueError(f"Expected column {col!r} in TSV for read-depth filtering.")
        m = alleles == b
        if m.any():
            out[m] = pd.to_numeric(df.loc[m, col], errors="coerce").to_numpy()
    return out


def suggest_heldout(pooled_pos, exclude, target_frac, set_sizes, max_single_frac):
    """
    Pick a held-out TEST chromosome set whose pooled positive fraction is closest to target_frac.
    Searches combinations of the given set sizes over non-excluded chromosomes, rejecting any set
    that contains a chromosome carrying more than max_single_frac of positives (so no single
    chromosome dominates the test estimate). Returns (chosen_list, held_frac) or (None, None).
    """
    total = sum(pooled_pos.values())
    if not total:
        return None, None
    frac = {c: pooled_pos[c] / total for c in pooled_pos}
    cands = [c for c in pooled_pos if c not in exclude and frac[c] <= max_single_frac]
    best = None  # (abs_dist, -nchroms, spread, combo)
    for k in set_sizes:
        for combo in itertools.combinations(cands, k):
            s = sum(frac[c] for c in combo)
            spread = max(frac[c] for c in combo) - min(frac[c] for c in combo)
            key = (abs(s - target_frac), -len(combo), spread)
            if best is None or key < best[0]:
                best = (key, s, combo)
    if best is None:
        return None, None
    _, held_frac, combo = best
    return list(combo), held_frac


def chrom_sort_key(c):
    s = c[3:] if c.lower().startswith("chr") else c
    return (0, int(s)) if s.isdigit() else (1, s)


def main():
    args = parse_args()

    # resolve scoping: --tf (substring) or --assay (exact); mutually exclusive-ish
    if args.tf and args.assay:
        raise SystemExit("Pass either --tf or --assay, not both.")
    scoped_assay = None
    if args.tf:
        scoped_assay = resolve_assay(args.tsv, args.tf, args.chunksize)
        print(f"--tf {args.tf!r} -> assay {scoped_assay!r}")
    elif args.assay:
        scoped_assay = args.assay

    keep_donors = set(s.strip() for s in args.donors.split(",")) if args.donors else None
    set_sizes = [int(s) for s in args.test_n_chroms.split(",") if s.strip()]
    dataset_keys = ["donor", "assay"] + (["tissue"] if args.group_tissue else [])

    n_total = defaultdict(int)
    n_pos = defaultdict(int)
    n_nan_p = defaultdict(int)
    n_pos_tested = defaultdict(int)
    chrom_pos = defaultdict(lambda: defaultdict(int))
    chrom_total = defaultdict(lambda: defaultdict(int))

    label_col, p_col = args.label_col, args.p_col
    n_chunks = 0
    for chunk in pd.read_csv(args.tsv, sep="\t", usecols=USECOLS, chunksize=args.chunksize):
        n_chunks += 1
        if scoped_assay is not None:
            chunk = chunk[chunk["assay"] == scoped_assay]
        if keep_donors is not None:
            chunk = chunk[chunk["donor"].isin(keep_donors)]
        if chunk.empty:
            continue

        if args.min_total_reads is not None:
            tot = hap_read_counts(chunk, "hap1_allele") + hap_read_counts(chunk, "hap2_allele")
            chunk = chunk[tot >= args.min_total_reads]
            if chunk.empty:
                continue

        lab = pd.to_numeric(chunk[label_col], errors="coerce")
        is_pos = (lab == 1).to_numpy()
        is_nan_p = chunk[p_col].isna().to_numpy()
        is_tested = ~is_nan_p
        chroms = chunk["chr"].astype(str).to_numpy()

        gcols = chunk[dataset_keys].astype(str)
        dataset_series = gcols.agg("|".join, axis=1).to_numpy()

        for dataset in np.unique(dataset_series):
            m = dataset_series == dataset
            key = tuple(dataset.split("|"))
            n_total[key] += int(m.sum())
            n_pos[key] += int(is_pos[m].sum())
            n_nan_p[key] += int(is_nan_p[m].sum())
            n_pos_tested[key] += int((m & is_pos & is_tested).sum())
            mt = m & is_tested
            cp, ct = chrom_pos[key], chrom_total[key]
            uch, inv = np.unique(chroms[mt], return_inverse=True)
            tot_by = np.bincount(inv, minlength=len(uch))
            pos_by = np.bincount(inv, weights=is_pos[mt].astype(float), minlength=len(uch))
            for c, tt, pp in zip(uch.tolist(), tot_by.tolist(), pos_by.tolist()):
                ct[c] += int(tt)
                cp[c] += int(pp)

    print(f"Read {n_chunks} chunk(s).")
    if not n_total:
        print("No rows matched the scoping filters.")
        return

    # ---- per-dataset table (one row per donor when scoped to a single assay) ----
    rows = []
    for key in sorted(n_total):
        tot = n_total[key]
        rec = dict(zip(dataset_keys, key))
        rec.update({
            "n_total": tot,
            "n_positive": n_pos[key],
            "frac_positive": (n_pos[key] / tot) if tot else np.nan,
            "n_nan_p": n_nan_p[key],
            "frac_nan_p": (n_nan_p[key] / tot) if tot else np.nan,
            "n_positive_tested": n_pos_tested[key],
        })
        rows.append(rec)
    datasets = pd.DataFrame(rows)
    scope_note = f"assay={scoped_assay}" if scoped_assay else "all assays"
    print(f"\n=== per-dataset summary ({scope_note}; "
          f"{'post min_total_reads>=%d' % args.min_total_reads if args.min_total_reads is not None else 'no read-depth filter'}) ===")
    print(datasets.to_string(index=False))
    denom = datasets["n_total"].sum()
    print(f"\npooled positive fraction: {datasets['n_positive'].sum()/denom:.4f}")
    print(f"pooled NaN-p fraction (rows the exclude policy DROPS): {datasets['n_nan_p'].sum()/denom:.4f}")
    print(f"frac_nan_p range across datasets: {datasets['frac_nan_p'].min():.4f} - {datasets['frac_nan_p'].max():.4f}")

    # ---- per-chromosome table (pooled over datasets, TESTED set) ----
    pooled_pos, pooled_tot = defaultdict(int), defaultdict(int)
    for key in chrom_pos:
        for c, v in chrom_pos[key].items():
            pooled_pos[c] += v
        for c, v in chrom_total[key].items():
            pooled_tot[c] += v

    ptot_pos = sum(pooled_pos.values())
    prows = [{"chr": c, "n_total_tested": pooled_tot[c], "n_positive_tested": pooled_pos[c],
              "frac_of_all_positives": pooled_pos[c] / ptot_pos if ptot_pos else np.nan}
             for c in sorted(pooled_pos, key=chrom_sort_key)]
    per_chrom = pd.DataFrame(prows)
    print(f"\n=== per-chromosome positive counts ({scope_note}, TESTED rows only) ===")
    print(per_chrom.to_string(index=False))
    print(f"\ntotal tested positives: {ptot_pos}")

    # ---- held-out test-chromosome suggestion (closest-to-target, no single chrom dominates) ----
    if args.suggest_test_frac is not None and ptot_pos:
        exclude = set(s.strip() for s in args.exclude_from_test.split(",") if s.strip())
        chosen, held = suggest_heldout(pooled_pos, exclude, args.suggest_test_frac,
                                       set_sizes, args.max_single_chrom_frac)
        print(f"\n=== suggested held-out TEST set (~{args.suggest_test_frac:.0%} of positives; "
              f"sizes {set_sizes}, no single chrom > {args.max_single_chrom_frac:.0%}) ===")
        if chosen is None:
            print("No valid set found — relax --max_single_chrom_frac or --test_n_chroms.")
        else:
            chosen = sorted(chosen, key=chrom_sort_key)
            fold_assignment = {c: 0 for c in chosen}
            print(f"chromosomes: {chosen}")
            print(f"positives held out: {held*ptot_pos:.0f} / {ptot_pos} = {held:.3f}")
            print("fold_assignment (paste into the config's partition: block):")
            print(f"  {fold_assignment}")
            print("NOTE: fold_assignment must be IDENTICAL across all datasets or cross-individual "
                  "leakage returns.")

    if args.out_prefix:
        datasets.to_csv(f"{args.out_prefix}_datasets.csv", index=False)
        per_chrom.to_csv(f"{args.out_prefix}_per_chrom.csv", index=False)
        print(f"\nWrote {args.out_prefix}_datasets.csv and {args.out_prefix}_per_chrom.csv")


if __name__ == "__main__":
    main()
