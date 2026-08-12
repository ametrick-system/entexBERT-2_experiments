#!/usr/bin/env python3
"""
plot_delta_vs_sigma.py — |Delta| vs predicted sigma for ADASTRA CTCF variants,
colored by ASB label, from a sigma-run perVariant file.

Reads ctcf_adastra_entexbert2_sigma_perVariant.csv.gz (must have sigma/zscore cols,
i.e. produced by the sigma-aware scorer). Draws two panels:
  A) |Delta| vs sigma scatter, ASB-positive vs negative, with the z = |Delta|/sigma
     iso-contours (diagonal lines) so the reader sees what dividing by sigma does.
  B) balanced-AUROC bar: |Delta| vs |Delta|/sigma (leak-free if a 'leaky' col exists).

  python plot_delta_vs_sigma.py \
      --pervariant ctcf_adastra_entexbert2_sigma_perVariant.csv.gz \
      --out adastra_delta_vs_sigma.png
"""
import argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def balanced_auroc(score, label, seed=1, n_boot=1000):
    rng = np.random.default_rng(seed)
    pos = np.where(label == 1)[0]; neg = np.where(label == 0)[0]
    n = min(len(pos), len(neg))
    if n < 10: return np.nan, (np.nan, np.nan), 0
    take = np.concatenate([rng.choice(pos, n, replace=False),
                           rng.choice(neg, n, replace=False)])
    pt = roc_auc_score(label[take], score[take])
    boots = []
    for _ in range(n_boot):
        b = rng.choice(take, len(take), replace=True)
        if len(np.unique(label[b])) < 2: continue
        boots.append(roc_auc_score(label[b], score[b]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return pt, (lo, hi), n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pervariant", required=True)
    ap.add_argument("--leak_free", action="store_true",
                    help="restrict to leaky==False rows (needs a 'leaky' column)")
    ap.add_argument("--out", default="adastra_delta_vs_sigma.png")
    a = ap.parse_args()

    df = pd.read_csv(a.pervariant)
    assert "sigma" in df.columns and "zscore" in df.columns, \
        f"no sigma/zscore columns in {a.pervariant} — re-run with the sigma-aware scorer"
    if a.leak_free:
        assert "leaky" in df.columns, "no 'leaky' column; run the scorer with --train_coords --drop_leaky"
        df = df[~df["leaky"]].reset_index(drop=True)
    lab = df["label"].to_numpy()
    ad = df["abs_delta"].to_numpy(); sg = df["sigma"].to_numpy(); z = df["zscore"].to_numpy()

    POS, NEG = "#c0392b", "#8fb0d0"       # alarm hue reserved for the positive class
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.4, 3.9), dpi=300,
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    # ---- Panel A: |Delta| vs sigma, colored by ASB label ----
    neg = df[df.label == 0]; pos = df[df.label == 1]
    axA.scatter(neg["sigma"], neg["abs_delta"], s=5, alpha=0.25, linewidths=0,
                color=NEG, label="ASB negative", zorder=2)
    axA.scatter(pos["sigma"], pos["abs_delta"], s=7, alpha=0.55, linewidths=0,
                color=POS, label="ASB positive", zorder=3)
    # z = |Delta|/sigma iso-lines: |Delta| = z * sigma
    xs = np.linspace(sg.min(), sg.max(), 50)
    for zv in np.percentile(z, [50, 90, 99]):
        axA.plot(xs, zv * xs, color="#666666", lw=0.7, ls=":", zorder=1)
        axA.text(xs[-1], zv*xs[-1], f" z={zv:.1f}", fontsize=5.5, color="#666666",
                 va="center", ha="left")
    axA.set_xlabel("predicted $\\sigma$ (uncertainty)", fontsize=8)
    axA.set_ylabel("$|\\Delta|$ = |$\\mu_{alt}-\\mu_{ref}$|", fontsize=8)
    axA.set_title("A  |$\\Delta$| vs $\\sigma$ (dotted = $z=|\\Delta|/\\sigma$ iso-lines)",
                  fontsize=8, loc="left")
    axA.margins(0.04); axA.tick_params(labelsize=6)
    axA.legend(loc="upper left", fontsize=6.5, frameon=False, markerscale=1.6)
    for s in ("top", "right"): axA.spines[s].set_visible(False)

    # ---- Panel B: balanced AUROC, |Delta| vs |Delta|/sigma ----
    reg = "leak-free" if a.leak_free else "full-set"
    a_d, (lo_d, hi_d), n = balanced_auroc(ad, lab)
    a_z, (lo_z, hi_z), _ = balanced_auroc(z, lab)
    xlab = ["$|\\Delta|$", "$|\\Delta|/\\sigma$"]
    vals = [a_d, a_z]; los = [lo_d, lo_z]; his = [hi_d, hi_z]
    cols = ["#1f3a5f", "#c0392b"]
    xp = np.arange(2)
    for i in range(2):
        axB.plot([xp[i], xp[i]], [los[i], his[i]], color=cols[i], lw=1.4, zorder=2)
        axB.scatter([xp[i]], [vals[i]], s=55, color=cols[i], zorder=3)
        axB.text(xp[i], his[i]+0.006, f"{vals[i]:.3f}", ha="center", va="bottom",
                 fontsize=7.5, color=cols[i])
    axB.axhline(0.5, color="#999999", lw=0.8, ls="--", zorder=1)
    axB.text(1.45, 0.5, "chance", fontsize=6, color="#999999", va="center")
    axB.set_xticks(xp); axB.set_xticklabels(xlab, fontsize=8)
    axB.set_ylabel("balanced AUROC vs ASB", fontsize=8)
    axB.set_title(f"B  does $\\sigma$ help?  ({reg}, $n_{{pos}}$={n})", fontsize=8, loc="left", pad=10)
    # headroom so the value labels above the top CI don't collide with the title
    axB.set_ylim(min(0.49, min(los) - 0.01), max(his) + 0.03)
    axB.set_xlim(-0.5, 1.7); axB.tick_params(labelsize=6)
    for s in ("top", "right"): axB.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print(f"[{reg}] |Delta| AUROC={a_d:.4f} [{lo_d:.4f},{hi_d:.4f}] | "
          f"|Delta|/sigma AUROC={a_z:.4f} [{lo_z:.4f},{hi_z:.4f}] | n_pos={n}")
    print(f"[wrote] {a.out}")

if __name__ == "__main__":
    main()
