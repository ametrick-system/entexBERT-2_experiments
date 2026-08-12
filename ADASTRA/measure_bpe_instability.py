#!/usr/bin/env python3
"""
measure_bpe_instability.py — quantify how a single SNV disrupts DNABERT-2 BPE tokenization.

For each variant, tokenize the ref and alt 257bp windows with offset mapping, then record:
  1. tok_differs      : do the token-ID sequences differ at all?
  2. dlen             : len(tokens_alt) - len(tokens_ref)  (nonzero => misaligned twin)
  3. disruption_span  : bp from the SNV to the FARTHEST changed token boundary
  4. center_tok_len   : length (bp) of the ref token containing the SNV
The twin Delta = f(alt)-f(ref) is noisy exactly when these are large.

Run in eb2 (DNABERT-2 tokenizer present) from repo root:
python measure_bpe_instability.py \
    --eval_csv ctcf_adastra_evalset.csv --ref_fasta /home/asm242/reference_genome/hg38.fa \
    --model_dir /home/asm242/entexBERT-2/DNABERT-2-117M-attention \
    --n 25000 --left_bp 128 --right_bp 128 --out bpe_instability
"""
import argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def changed_span_bp(off_ref, off_alt, center):
    """Farthest distance (bp) from `center` to any token-boundary position that differs
    between ref and alt. Boundaries = the set of all start/end cut points."""
    b_ref = set()
    for s, e in off_ref: b_ref.add(s); b_ref.add(e)
    b_alt = set()
    for s, e in off_alt: b_alt.add(s); b_alt.add(e)
    diff = b_ref.symmetric_difference(b_alt)
    if not diff:
        return 0.0
    return float(max(abs(p - center) for p in diff))

def center_token_len(off_ref, center):
    for s, e in off_ref:
        if s <= center < e:
            return e - s
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_csv", required=True)
    ap.add_argument("--ref_fasta", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--n", type=int, default=25000, help="uniform subset size (all positives forced in)")
    ap.add_argument("--left_bp", type=int, default=128)
    ap.add_argument("--right_bp", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="bpe_instability")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    import pyfaidx
    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    fa = pyfaidx.Fasta(a.ref_fasta)

    # DNABERT-2's tokenizer may or may not support return_offsets_mapping depending on
    # whether the installed build is the fast (Rust) tokenizer. Probe once; if unsupported,
    # reconstruct char offsets from the token strings (DNA BPE tokens are literal substrings,
    # so a left-to-right cursor over the decoded pieces recovers exact spans).
    def _supports_offsets():
        try:
            t = tok("ACGTACGT", return_offsets_mapping=True, add_special_tokens=False)
            return "offset_mapping" in t and len(t["offset_mapping"]) > 0
        except Exception:
            return False
    USE_OFFSETS = _supports_offsets()

    def tokenize_with_offsets(s):
        """Return (input_ids, [(start,end),...]) char spans, via fast path or reconstruction."""
        if USE_OFFSETS:
            e = tok(s, return_offsets_mapping=True, add_special_tokens=False)
            return e["input_ids"], [(a_, b_) for (a_, b_) in e["offset_mapping"] if b_ > a_]
        ids = tok(s, add_special_tokens=False)["input_ids"]
        pieces = tok.convert_ids_to_tokens(ids)
        offs, cur = [], 0
        for p in pieces:
            p = p.replace("##", "")                 # wordpiece-style continuation, if any
            L = len(p)
            offs.append((cur, cur + L)); cur += L
        return ids, offs
    print(f"[tokenizer] offset_mapping supported: {USE_OFFSETS}"
          + ("" if USE_OFFSETS else "  (using string-reconstruction fallback)"))

    ev = pd.read_csv(a.eval_csv)
    # uniform subset with ALL positives forced in
    rng = np.random.default_rng(a.seed)
    pos_idx = ev.index[ev.label == 1].to_numpy()
    neg_idx = ev.index[ev.label == 0].to_numpy()
    n_neg = max(0, a.n - len(pos_idx))
    keep = np.concatenate([pos_idx, rng.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False)])
    ev = ev.loc[keep].reset_index(drop=True)

    W = a.left_bp + a.right_bp + 1
    rows = []
    for _, r in ev.iterrows():
        chrom, pos = str(r["chr"]), int(r["pos"])
        s = pos - 1 - a.left_bp; e = pos - 1 + a.right_bp + 1
        if s < 0: continue
        try: seq = str(fa[chrom][s:e]).upper()
        except Exception: continue
        if len(seq) != W: continue
        c = a.left_bp
        if seq[c] != str(r["ref"]).upper(): continue          # ref-allele guardrail
        alt = seq[:c] + str(r["alt"]).upper() + seq[c+1:]
        ids_r, off_r = tokenize_with_offsets(seq)
        ids_a, off_a = tokenize_with_offsets(alt)
        rows.append(dict(
            snp=r["snp"], label=int(r["label"]),
            tok_differs=int(ids_r != ids_a),
            dlen=len(ids_a) - len(ids_r),
            disruption_span=changed_span_bp(off_r, off_a, c),
            center_tok_len=center_token_len(off_r, c),
        ))
    d = pd.DataFrame(rows)
    d.to_csv(f"{a.out}_perVariant.csv.gz", index=False, compression="gzip")

    # ---- summary stats, per class ----
    def summarize(sub, name):
        return (f"{name:9s} n={len(sub):6d}  tok_differs={sub.tok_differs.mean():.3f}  "
                f"dlen!=0={ (sub.dlen!=0).mean():.3f}  "
                f"span med={sub.disruption_span.median():.1f} p90={sub.disruption_span.quantile(.9):.1f}  "
                f"ctr_tok med={sub.center_tok_len.median():.1f}")
    print(summarize(d, "ALL"))
    print(summarize(d[d.label==1], "positive"))
    print(summarize(d[d.label==0], "negative"))

    # ---- 2-panel figure ----
    POS, NEG = "#c0392b", "#8fb0d0"
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.6, 3.8), dpi=300)
    dpos, dneg = d[d.label==1], d[d.label==0]

    # Panel A (headline): disruption-span distribution, pos vs neg
    hi = float(d.disruption_span.quantile(0.99)) or 1.0
    bins = np.linspace(0, hi, 40)
    axA.hist(dneg.disruption_span, bins=bins, density=True, color=NEG, alpha=0.7,
             label=f"ASB neg (n={len(dneg)})")
    axA.hist(dpos.disruption_span, bins=bins, density=True, histtype="step", lw=1.6,
             color=POS, label=f"ASB pos (n={len(dpos)})")
    axA.axvline(a.left_bp, color="#666", lw=0.8, ls=":")  # window edge reference
    axA.set_xlabel("disruption span: bp from SNV to farthest changed token boundary", fontsize=7.5)
    axA.set_ylabel("density", fontsize=8)
    axA.set_title("A  BPE disruption span (sizes the CNN receptive field)", fontsize=8, loc="left")
    axA.legend(fontsize=6.5, frameon=False); axA.tick_params(labelsize=6)
    for s in ("top","right"): axA.spines[s].set_visible(False)

    # Panel B: token-count-change |dlen| distribution, pos vs neg
    m = int(min(6, d.dlen.abs().max()))
    xs = np.arange(0, m+1)
    wpos = [ (dpos.dlen.abs()==k).mean() for k in xs ]
    wneg = [ (dneg.dlen.abs()==k).mean() for k in xs ]
    w = 0.38
    axB.bar(xs-w/2, wneg, w, color=NEG, label="ASB neg")
    axB.bar(xs+w/2, wpos, w, color=POS, label="ASB pos")
    axB.set_xlabel("|token-count change|  (|len(alt)-len(ref)|)", fontsize=7.5)
    axB.set_ylabel("fraction of variants", fontsize=8)
    axB.set_title(f"B  token-count change  (dlen!=0: {(d.dlen!=0).mean():.0%} of variants)",
                  fontsize=8, loc="left")
    axB.set_xticks(xs); axB.legend(fontsize=6.5, frameon=False); axB.tick_params(labelsize=6)
    for s in ("top","right"): axB.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{a.out}.png", bbox_inches="tight")
    print(f"[wrote] {a.out}.png and {a.out}_perVariant.csv.gz")

if __name__ == "__main__":
    main()
