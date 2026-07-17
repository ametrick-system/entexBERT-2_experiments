#!/usr/bin/env python
"""
diagnose_signal.py — is there learnable ASB signal, or is the model collapsing?

Answers three questions, cheaply, without retraining:
  A. k-mer FLOOR: can a linear logistic model on 6-mer counts beat chance on DEV?
     -> if YES, signal exists in ref_single and DNABERT-2 should at least match it.
     -> if NO (dev AUROC ~0.5), the ref_single framing itself lacks the information.
  B. CHECKPOINT eval: score a trained checkpoint on its own TRAIN set and on DEV.
     -> train AUROC ~0.5  => model can't even fit -> optimization broken OR label not a
                              function of input (imbalance is exonerated either way).
     -> train high, dev ~0.5 => overfitting / generalization, NOT imbalance.
     -> both moderate & tracking => learning fine, needs more steps / better LR.
     Also dumps the predicted-probability histogram: a spike at one value = collapse
     to a constant prediction (the AUROC=0.500 / MCC=0 fingerprint).
  C. VERDICT: prints the interpretation.

Usage:
  # A only (no GPU, no checkpoint needed) — run this FIRST, it's the fastest signal test:
  python diagnose_signal.py --data_dir <fold0_dir> --kmer_only

  # A + B (needs the trained checkpoint dir with run_config.json + weights):
  python diagnose_signal.py --data_dir <fold0_dir> --checkpoint_dir <runs/lupi or runs/baseline>

Notes:
- --data_dir is the folder holding train.csv / dev.csv (columns: sequence,label[,aux...]).
- For the k-mer floor, DEV AUROC is the signal indicator (train AUROC saturates to ~1.0
  because thousands of k-mer features memorize the train rows — expected, not informative).
- Part B reuses your own model_io.load_model_and_tokenizer + logits_and_embeddings, so the
  forward pass matches training exactly (eval mode -> dropout identity). Positive-class prob
  = softmax(logits)[:,1] for the binary (main_num_labels=2) head.
"""
import argparse, os, sys, json
import numpy as np
import pandas as pd


def kmer_floor(train_df, dev_df, k=6, seq_col="sequence", label_col="label", C=1.0, max_train=None):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    if max_train and len(train_df) > max_train:
        train_df = train_df.sample(max_train, random_state=0)
    vec = CountVectorizer(analyzer="char", ngram_range=(k, k), lowercase=False)
    Xtr = vec.fit_transform(train_df[seq_col].astype(str))
    Xdv = vec.transform(dev_df[seq_col].astype(str))
    ytr = train_df[label_col].astype(int).values
    ydv = dev_df[label_col].astype(int).values
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=C)
    clf.fit(Xtr, ytr)
    pdv = clf.predict_proba(Xdv)[:, 1]
    return dict(k=k, n_features=Xtr.shape[1],
                train_base_rate=float(ytr.mean()), dev_base_rate=float(ydv.mean()),
                dev_auroc=float(roc_auc_score(ydv, pdv)),
                dev_auprc=float(average_precision_score(ydv, pdv)))


def eval_checkpoint(data_dir, checkpoint_dir, batch_size=64, max_rows=8000, device=None):
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
    from entexbert2.model_io import load_model_and_tokenizer, logits_and_embeddings
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, run_config = load_model_and_tokenizer(checkpoint_dir, device=device)
    mml = int(run_config.get("model_max_length", 512))

    def score_split(fn):
        df = pd.read_csv(os.path.join(data_dir, fn))
        if max_rows and len(df) > max_rows:                    # class-stratified subsample
            df = (df.groupby("label", group_keys=False)
                    .apply(lambda g: g.sample(min(len(g), int(max_rows * len(g) / len(df))),
                                              random_state=0)))
        seqs = df["sequence"].astype(str).tolist()
        y = df["label"].astype(int).values
        probs = []
        for i in range(0, len(seqs), batch_size):
            enc = tok(seqs[i:i+batch_size], return_tensors="pt", padding="longest",
                      truncation=True, max_length=mml)
            ids = enc["input_ids"].to(device); mask = enc["attention_mask"].to(device)
            logits, _ = logits_and_embeddings(model, ids, mask)
            p = torch.softmax(logits, dim=-1)[:, 1] if logits.shape[-1] > 1 \
                else torch.sigmoid(logits[:, 0])
            probs.append(p.float().cpu().numpy())
        p = np.concatenate(probs)
        pred = (p >= 0.5).astype(int)
        # histogram of predicted probs -> collapse detector
        hist, edges = np.histogram(p, bins=10, range=(0, 1))
        return dict(n=len(y), base_rate=float(y.mean()),
                    auroc=float(roc_auc_score(y, p)) if y.min() != y.max() else float("nan"),
                    auprc=float(average_precision_score(y, p)) if y.min() != y.max() else float("nan"),
                    mcc=float(matthews_corrcoef(y, pred)),
                    prob_mean=float(p.mean()), prob_std=float(p.std()),
                    prob_hist=hist.tolist(), pred_pos_frac=float(pred.mean()))
    return {fn.split(".")[0]: score_split(fn) for fn in ("train.csv", "dev.csv")}, run_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder with train.csv / dev.csv")
    ap.add_argument("--checkpoint_dir", default=None, help="trained run dir (run_config.json + weights)")
    ap.add_argument("--kmer_only", action="store_true")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--kmer_max_train", type=int, default=60000)
    ap.add_argument("--eval_max_rows", type=int, default=8000)
    args = ap.parse_args()

    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    dev_df = pd.read_csv(os.path.join(args.data_dir, "dev.csv"))
    print(f"[data] train={len(train_df)} dev={len(dev_df)} "
          f"train_pos_rate={train_df['label'].mean():.4f} dev_pos_rate={dev_df['label'].mean():.4f}")

    print("\n" + "=" * 70 + "\nPART A — k-mer logistic-regression FLOOR (dev AUROC is the signal test)\n" + "=" * 70)
    a = kmer_floor(train_df, dev_df, k=args.k, max_train=args.kmer_max_train)
    print(json.dumps(a, indent=2))
    floor_has_signal = a["dev_auroc"] > 0.55
    print(f"\n-> k-mer floor {'FINDS' if floor_has_signal else 'does NOT find'} signal "
          f"(dev AUROC {a['dev_auroc']:.3f} vs 0.5 chance).")

    ck = None
    if not args.kmer_only and args.checkpoint_dir:
        print("\n" + "=" * 70 + "\nPART B — trained-checkpoint eval on TRAIN vs DEV\n" + "=" * 70)
        ck, rc = eval_checkpoint(args.data_dir, args.checkpoint_dir, max_rows=args.eval_max_rows)
        for split in ("train", "dev"):
            s = ck[split]
            print(f"\n[{split}] n={s['n']} base={s['base_rate']:.4f} "
                  f"AUROC={s['auroc']:.4f} AUPRC={s['auprc']:.4f} MCC={s['mcc']:.4f}")
            print(f"       pred prob: mean={s['prob_mean']:.4f} std={s['prob_std']:.4f} "
                  f"pred_pos_frac={s['pred_pos_frac']:.4f}")
            print(f"       prob hist [0..1, 10 bins]: {s['prob_hist']}")
            if s["prob_std"] < 0.02:
                print("       ** COLLAPSE: predictions nearly constant -> AUROC~0.5 is a "
                      "training-dynamics failure, not imbalance. **")

    print("\n" + "=" * 70 + "\nPART C — VERDICT\n" + "=" * 70)
    if ck is None:
        if floor_has_signal:
            print("A linear model finds ranking signal in ref_single that the network should match.\n"
                  "If DNABERT-2 sits at AUROC~0.5, the problem is OPTIMIZATION (LR/stability),\n"
                  "not imbalance and not a missing signal. Run Part B on a checkpoint to localize.")
        else:
            print("Even a k-mer linear model cannot beat chance on dev. Signal in ref_single is\n"
                  "very weak -> this points at the INPUT FRAMING (ref-only, no alt allele), the\n"
                  "Wave-B ref_alt_pair reframe -- NOT at class imbalance. Class weights won't help.")
    else:
        tr, dv = ck["train"]["auroc"], ck["dev"]["auroc"]
        collapse = ck["train"]["prob_std"] < 0.02 or ck["dev"]["prob_std"] < 0.02
        if collapse or (tr < 0.55 and dv < 0.55):
            print("Model is at chance on TRAIN too -> it isn't even fitting. This is optimization\n"
                  "(LR too high / gradient blow-up from 33x positive weighting) or the label isn't a\n"
                  "function of the ref-only input. Either way, IMBALANCE IS EXONERATED. Fix: lower LR,\n"
                  "or a gentler pos_weight (e.g. sqrt-inverse-freq / capped) rather than full balanced.")
        elif tr >= 0.6 and dv < 0.55:
            print("Model FITS train but not dev -> generalization/overfitting, not imbalance.\n"
                  "Fix: regularization, fewer epochs, or the input framing -- not class weights.")
        else:
            print("Model is learning (train & dev both above chance and tracking). It likely just\n"
                  "needs more steps at the right LR. Imbalance is being handled adequately.")
        if floor_has_signal and (tr < 0.55):
            print("\nNOTE: the k-mer floor DID find signal while DNABERT-2 did not fit it -> strong\n"
                  "evidence the failure is optimization, since a linear model already beats the net.")


if __name__ == "__main__":
    main()
