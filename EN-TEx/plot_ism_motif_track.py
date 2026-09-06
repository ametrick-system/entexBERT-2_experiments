#!/usr/bin/env python
"""
Per-window motif-track figure for the entexBERT-2 trunk, in the style of the DNABERT-1
attention-track panels but driven by base-resolution ISM importance instead of attention.

Top panel : ISM importance profile along the window (per-base; the model's own attribution).
Bottom    : JASPAR motif-hit track -- for each TF with a FIMO hit in this window, a colored
            bar spanning the hit, one row per TF. This is the ISM analogue of the
            "chr8:[...] ZNF580/EGR2/ZNF740/..." track in the reference figures.

FIMO scan uses memelite (ships with modisco-lite; no MEME binary). Runs LOCALLY on the ISM .npz.
Set NUMBA_CACHE_DIR before import.
"""
import os
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
import argparse, json, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

BASES = "ACGT"


def load_jaspar_motifs(path, restrict=None):
    """{name|id: (4,W) count matrix} for memelite.fimo. restrict = optional set of TF names."""
    db = json.load(open(path)); out = {}
    for k, v in db.items():
        nm = v.get("name", k)
        if restrict and nm.upper() not in restrict:
            continue
        out[f"{nm}|{v.get('id', k)}"] = np.array([v["pfm"][b] for b in BASES], dtype=np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ism", required=True)
    ap.add_argument("--jaspar", required=True)
    ap.add_argument("--window", type=int, default=0, help="which window index to draw")
    ap.add_argument("--smooth", type=int, default=5, help="rolling-mean width for the profile")
    ap.add_argument("--fimo_threshold", type=float, default=1e-4)
    ap.add_argument("--restrict", default=None,
                    help="comma-separated TF names to scan (default: discovered_motifs.json hits if given, else a CTCF+cofactor set)")
    ap.add_argument("--discovered", default=None,
                    help="discovered_motifs.json -> restrict the track to TFs MoDISco actually found")
    ap.add_argument("--max_tracks", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from memelite import fimo

    d = np.load(a.ism, allow_pickle=True)
    onehot, imp = d["onehot"], d["importance"]
    N, L, _ = onehot.shape
    w = a.window
    if not (0 <= w < N):
        raise SystemExit(f"--window {w} out of range [0,{N})")

    # restrict the TF set: MoDISco-discovered hits > explicit list > default CTCF+known cofactors
    restrict = None
    if a.discovered and os.path.exists(a.discovered):
        disc = json.load(open(a.discovered))
        restrict = {h["tf"].upper() for p in disc for h in p.get("top_hits", [])}
    elif a.restrict:
        restrict = {s.strip().upper() for s in a.restrict.split(",")}
    else:
        restrict = {"CTCF", "CTCFL", "ZNF143", "SP1", "KLF4", "YY1", "ZBTB33"}

    motifs = load_jaspar_motifs(a.jaspar, restrict=restrict)
    oh = onehot[w].T[None, :, :].astype(np.float32)          # (1,4,L)
    res = fimo(motifs, oh, threshold=a.fimo_threshold)

    # collect hits: {tf: [(start,end,strand,p)]}
    hits = {}
    for r in res:
        if len(r) == 0:
            continue
        for _, row in r.iterrows():
            tf = str(row["motif_name"]).split("|")[0]
            hits.setdefault(tf, []).append((int(row["start"]), int(row["end"]),
                                            str(row["strand"]), float(row["p-value"])))
    # order tracks by best (smallest) p-value, cap at max_tracks
    tf_order = sorted(hits, key=lambda t: min(h[3] for h in hits[t]))[:a.max_tracks]

    # smooth importance profile
    prof = imp[w].astype(np.float64)
    if a.smooth > 1:
        k = np.ones(a.smooth) / a.smooth
        prof = np.convolve(prof, k, mode="same")

    fig, (axp, axt) = plt.subplots(
        2, 1, figsize=(11, 4.8), sharex=True,
        gridspec_kw={"height_ratios": [2, max(1, len(tf_order) * 0.28)]})
    axp.plot(np.arange(L), prof, color="#7030a0", lw=1.3)
    axp.axhline(0, color="0.7", lw=0.7)
    axp.fill_between(np.arange(L), 0, prof, where=prof > 0, color="#7030a0", alpha=0.18)
    axp.set_ylabel("ISM importance")
    axp.set_title(f"entexBERT-2 trunk — window {w}: ISM importance + JASPAR motif hits", fontsize=11)

    cmap = mpl.colormaps["tab20"]
    for i, tf in enumerate(tf_order):
        y = len(tf_order) - i
        for (s, e, strand, p) in hits[tf]:
            axt.plot([s, e], [y, y], lw=5, solid_capstyle="butt", color=cmap(i % 20))
        axt.text(-2, y, tf, ha="right", va="center", fontsize=8)
    axt.set_ylim(0.3, len(tf_order) + 0.7)
    axt.set_yticks([]); axt.set_xlabel("position in window (bp)")
    axt.set_xlim(-0.02 * L, L)
    if not tf_order:
        axt.text(L / 2, 1, "no JASPAR hits at this threshold", ha="center", va="center", fontsize=9, color="0.5")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"[plot] window {w}: {len(tf_order)} TF tracks -> {a.out}")
    for tf in tf_order:
        best = min(hits[tf], key=lambda h: h[3])
        print(f"   {tf:10s} {len(hits[tf])} hit(s), best {best[0]}-{best[1]}({best[2]}) p={best[3]:.1e}")


if __name__ == "__main__":
    main()
