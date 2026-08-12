#!/usr/bin/env python3
"""
sigma_sanity.py — diagnose whether the test-time sigma head is actually surfacing,
and (if so) sanity-check that the learned variance VARIES and tracks prediction error.

Run from the repo root or the ADASTRA dir (needs entexbert2 + pyfaidx on PYTHONPATH):
  python sigma_sanity.py --checkpoint_dir <.../runs/reg> \
      --ref_fasta /home/asm242/reference_genome/hg38.fa \
      --eval_csv ctcf_adastra_evalset.csv[.gz] --n 3000
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--ref_fasta", required=True)
    ap.add_argument("--eval_csv", required=True)
    ap.add_argument("--n", type=int, default=3000, help="how many variants to score")
    ap.add_argument("--left_bp", type=int, default=128)
    ap.add_argument("--right_bp", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    # ---- 1) run_config check: does the checkpoint claim predict_sigma? ----
    rc_path = os.path.join(a.checkpoint_dir, "run_config.json")
    rc = json.load(open(rc_path)) if os.path.exists(rc_path) else {}
    print(f"[run_config] predict_sigma = {rc.get('predict_sigma')}  "
          f"sigma_logvar_clamp = {rc.get('sigma_logvar_clamp')}  "
          f"task = {rc.get('task')}  contrast_mode = {rc.get('contrast_mode')}")
    if not rc.get("predict_sigma"):
        print("  !! This checkpoint was NOT trained with predict_sigma=True. "
              "It is the plain/mu-only model. A sigma eval here is meaningless — "
              "point --checkpoint_dir at the SIGMA run's runs/reg.")

    # ---- 2) model_io check: is the installed package sigma-aware? ----
    from entexbert2 import model_io
    import inspect
    src = inspect.getsource(model_io.logits_and_embeddings)
    io_sigma_aware = "predict_sigma" in src and "log_var" in src
    print(f"[model_io] logits_and_embeddings is sigma-aware: {io_sigma_aware}")
    if not io_sigma_aware:
        print("  !! The entexbert2 on your PYTHONPATH has the OLD model_io "
              "(no sigma surfacing). Re-install/pull the repo with the sigma diff, "
              "then `pip install -e .` in the eb2 env.")

    # ---- 3) actually score a subset and inspect the returned logits shape ----
    from entexbert2.analyze import run_inference
    import pyfaidx
    fa = pyfaidx.Fasta(a.ref_fasta)
    ev = pd.read_csv(a.eval_csv)
    ev = ev.sample(min(a.n, len(ev)), random_state=1).reset_index(drop=True)

    W = a.left_bp + a.right_bp + 1
    pairs, keep = [], []
    for i, r in ev.iterrows():
        chrom, pos = str(r["chr"]), int(r["pos"])
        s = pos - 1 - a.left_bp; e = pos - 1 + a.right_bp + 1
        if s < 0: continue
        try: seq = str(fa[chrom][s:e]).upper()
        except Exception: continue
        if len(seq) != W: continue
        c = a.left_bp
        if seq[c] != str(r["ref"]).upper(): continue          # ref-allele guardrail
        alt = seq[:c] + str(r["alt"]).upper() + seq[c+1:]
        pairs.append([seq, alt]); keep.append(i)
    print(f"[windows] built {len(pairs)} ref/alt pairs (W={W})")

    logits, _emb, run_config = run_inference(
        a.checkpoint_dir, pairs, a.batch_size, a.device, None)
    arr = np.asarray(logits, dtype=float).reshape(len(pairs), -1)
    print(f"[inference] returned logits shape = {arr.shape}  (2 cols => sigma is surfacing)")

    if arr.shape[1] < 2:
        print("\n=== VERDICT: sigma NOT surfacing. delta-only. ===")
        print("Fix per the checks above (wrong checkpoint, or stale model_io), then re-run.")
        return

    # ---- 4) THE SANITY CHECK: does learned variance VARY and track error? ----
    delta = arr[:, 0]; log_var = arr[:, 1]; sigma = np.exp(0.5 * log_var)
    print(f"\n=== sigma sanity ===")
    print(f"log_var: std={log_var.std():.4f}  min={log_var.min():.3f} max={log_var.max():.3f}")
    if log_var.std() < 1e-3:
        print("  !! log_var is ~FLAT. The head did not learn per-sequence variance; "
              "|Delta|/sigma == |Delta|/const, so it cannot change the ranking. "
              "This is an informative negative — sigma-from-sequence carries no signal here.")
    else:
        print(f"  sigma varies: {np.exp(0.5*log_var.min()):.4f} .. {np.exp(0.5*log_var.max()):.4f} "
              f"(median {np.median(sigma):.4f})")
        # sigma should be LARGER where |delta| is large IF it is tracking prediction spread;
        # more informative: bin by sigma, look at |delta| dispersion per bin.
        from scipy.stats import spearmanr
        q = pd.qcut(sigma, 5, labels=False, duplicates="drop")
        tab = pd.DataFrame({"sigma_bin": q, "abs_delta": np.abs(delta)}) \
                .groupby("sigma_bin")["abs_delta"].agg(["median","std","count"])
        print("  |delta| dispersion by sigma quintile (high-sigma bins should be more spread):")
        print(tab.to_string())
        print(f"  Spearman(sigma, |delta|) = {spearmanr(sigma, np.abs(delta)).correlation:.4f}")

if __name__ == "__main__":
    main()
