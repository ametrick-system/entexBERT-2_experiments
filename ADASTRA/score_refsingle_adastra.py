#!/usr/bin/env python3
"""
score_refsingle_adastra.py — score a PRE-REFACTOR ref_single 2-class ASB classifier
on the ADASTRA CTCF benchmark, by P(ASB) of the hg38 REFERENCE window alone.

WHY THIS IS SELF-CONTAINED (no `entexbert2` import): the checkpoint is a pre-refactor
LUPI-era model (run_config: main_num_labels=2, num_aux_tasks, contrast_mode). Rather than
depend on whatever version the cluster `entexbert2.model` is at, we rebuild the exact head
from the state-dict keys the user dumped:

    logvar_bias        (1,)        # vestigial sigma-head scalar; loaded then IGNORED for clf
    main_head.0.weight (768,768)   # Linear 768->768
    main_head.0.bias   (768,)
    main_head.3.weight (2,768)     # Linear 768->2   (indices 1,2 = gelu, dropout: no params)
    main_head.3.bias   (2,)

Pooling is center_mean width 5 (from run_config), reproduced BYTE-FOR-BYTE from model.py's
_pool: mean of `width` tokens centered at valid//2, valid = attention_mask.sum().

SCORING CONVENTION (region-propensity baseline, per user): feed ONLY the hg38 reference
window at each variant; score = softmax(main_head(pool(ref)))[1] = P(class=1) = P(ASB).
The alt allele is never seen — this measures how far REGION PROPENSITY alone gets you,
the honest ref_single baseline against the paired distance head (leak-free 0.7208).

LEAK-FREE: this checkpoint used the binsplit2 (salt=entexbert2_v1) 100kb bin split, which
is NOT the stem run's chr3+chr10. So the leak filter reads THIS checkpoint's own meta
sidecars (fold0/train.meta.csv + fold0/dev.meta.csv) — pass both via --train_coords. AUROC
is prevalence-invariant, so the number is comparable to 0.7208 even though the held-out sets
are not variant-identical.

WHAT YOU NEED ON THE CLUSTER (all small, one GPU):
  --checkpoint_dir  the ref_single run dir with pytorch_model.bin + run_config.json
                    (e.g. experiments/refsingle_ENC-002_CTCF_..._cw/runs/clf)
  --eval_csv        ctcf_adastra_evalset.csv.gz     (9071 pos + 242720 neg)
  --ref_fasta       /home/asm242/reference_genome/hg38.fa
  --reference_csv   ctcf_benchmark_reference_auroc.csv   (CTCF_AUROC column)
  --train_coords    .../fold0/train.meta.csv .../fold0/dev.meta.csv   (leak-free)
  --model_name_or_path  $HOME/entexBERT-2/DNABERT-2-117M-attention   (the backbone)

Run (eb2 env, GPU node):
  python score_refsingle_adastra.py \
    --checkpoint_dir /home/asm242/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/refsingle_ENC-002_CTCF_ctcf_ref_single_classification_binsplit2_cw/runs/clf \
    --model_name_or_path $HOME/entexBERT-2/DNABERT-2-117M-attention \
    --eval_csv ctcf_adastra_evalset.csv \
    --ref_fasta $HOME/reference_genome/hg38.fa \
    --reference_csv ctcf_benchmark_reference_auroc.csv \
    --train_coords /home/asm242/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/refsingle_ENC-002_CTCF_ctcf_ref_single_classification_binsplit2_cw/inputs/ENC-002__TF-ChIP-seq_CTCF/fold0/train.meta.csv \
                   /home/asm242/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/refsingle_ENC-002_CTCF_ctcf_ref_single_classification_binsplit2_cw/inputs/ENC-002__TF-ChIP-seq_CTCF/fold0/dev.meta.csv \
    --out refsingle_adastra
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import AutoModel, AutoTokenizer


# ----------------------------------------------------------------------
# Model: DNABERT-2 backbone + center_mean pool + 2-layer main_head -> 2 logits.
# Rebuilt from the checkpoint's own key layout; loads with 0 missing / 0 unexpected.
# ----------------------------------------------------------------------
class RefSingleClassifier(torch.nn.Module):
    def __init__(self, model_name_or_path, center_pool_width=5,
                 head_activation="gelu", head_dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        hidden = self.backbone.config.hidden_size          # 768
        self.center_pool_width = int(center_pool_width)
        act = {"gelu": torch.nn.GELU(), "relu": torch.nn.ReLU()}[head_activation]
        # main_head indices: 0 Linear(768,768), 1 act, 2 Dropout, 3 Linear(768,2)
        self.main_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            act,
            torch.nn.Dropout(head_dropout),
            torch.nn.Linear(hidden, 2),
        )
        # vestigial sigma-head scalar present in the checkpoint; loaded, never used for clf
        self.logvar_bias = torch.nn.Parameter(torch.zeros(1))

    def _pool(self, seq, attention_mask):
        # center_mean: mean of center_pool_width tokens around valid//2 (matches model.py._pool)
        bsz, max_len, _ = seq.shape
        half = self.center_pool_width // 2
        pooled = []
        for b in range(bsz):
            valid = int(attention_mask[b].sum().item()) if attention_mask is not None else max_len
            valid = max(valid, 1)
            center = valid // 2
            start = max(0, center - half)
            end = min(valid, center + half + 1)
            pooled.append(seq[b, start:end, :].mean(dim=0))
        return torch.stack(pooled, dim=0)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        seq = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        h = self._pool(seq, attention_mask)
        logits = self.main_head(h)                          # (B, 2)
        return logits


def load_checkpoint(model_name_or_path, ckpt_dir, run_config):
    model = RefSingleClassifier(
        model_name_or_path,
        center_pool_width=int(run_config.get("center_pool_width", 5)),
        head_activation=run_config.get("head_activation", "gelu"),
        head_dropout=float(run_config.get("head_dropout", 0.1)),
    )
    sd_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    sd = torch.load(sd_path, map_location="cpu")
    # Our submodule is named `backbone.`; the checkpoint should match (the pre-refactor class
    # also used self.backbone = AutoModel). If a checkpoint instead used a bare/`bert.` prefix,
    # remap so backbone.* keys line up rather than blow the seam assert. Head keys (main_head.*,
    # logvar_bias) are left untouched.
    want = {k for k in model.state_dict()}
    if not any(k.startswith("backbone.") for k in sd) and any(k.startswith("bert.") for k in sd):
        sd = {("backbone." + k[len("bert."):] if k.startswith("bert.") else k): v
              for k, v in sd.items()}
        print("[load] remapped 'bert.' backbone prefix -> 'backbone.'")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # the ONLY acceptable "missing" are none; the ONLY acceptable "unexpected" are none.
    missing = [k for k in missing]
    unexpected = [k for k in unexpected]
    print(f"[load] {sd_path}")
    print(f"[load] missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
    if missing:
        print("       missing:", missing[:12], "..." if len(missing) > 12 else "")
    if unexpected:
        print("       unexpected:", unexpected[:12], "..." if len(unexpected) > 12 else "")
    # HARD SEAM: head + backbone must both be fully populated. Allow ZERO missing/unexpected.
    assert not missing and not unexpected, (
        "state_dict load is not clean -- the head or backbone keys do not match. "
        "Refusing to score with a partially-random model (would give a garbage AUROC silently)."
    )
    model.eval()
    return model


# ----------------------------------------------------------------------
# Reference windows only (no alt): hg38 window centered on the SNV, length = win.
# ----------------------------------------------------------------------
def build_ref_windows(eval_df, ref_fasta, left_bp, right_bp):
    fa = Fasta(ref_fasta, sequence_always_upper=True)
    win = left_bp + 1 + right_bp
    seqs, keep = [], []
    n_oob = n_badchrom = n_refmismatch = 0
    for chrom, pos1, ref_a in zip(eval_df["chr"], eval_df["pos"], eval_df["ref"]):
        if chrom not in fa:
            seqs.append(""); keep.append(False); n_badchrom += 1; continue
        p0 = int(pos1) - 1
        start = p0 - left_bp
        end = p0 + right_bp + 1
        if start < 0 or end > len(fa[chrom]):
            seqs.append(""); keep.append(False); n_oob += 1; continue
        seq = str(fa[chrom][start:end])
        if len(seq) != win:
            seqs.append(""); keep.append(False); n_oob += 1; continue
        if seq[left_bp] != str(ref_a).upper():
            n_refmismatch += 1
        seqs.append(seq); keep.append(True)
    print(f"[windows] built {sum(keep)}/{len(eval_df)}  "
          f"(dropped: {n_oob} out-of-bounds, {n_badchrom} bad-chrom; "
          f"ref-base!=hg38 on {n_refmismatch} kept rows)")
    return seqs, np.array(keep, dtype=bool)


@torch.no_grad()
def score_windows(model, tokenizer, seqs, batch_size, device, max_len):
    model.to(device)
    probs = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        logits = model(input_ids, attn)                     # (B, 2)
        p = torch.softmax(logits.float(), dim=-1)[:, 1]     # P(class=1) = P(ASB)
        probs.append(p.cpu().numpy())
        if (i // batch_size) % 50 == 0:
            print(f"  [score] {i+len(chunk)}/{len(seqs)}", flush=True)
    return np.concatenate(probs)


def balanced_auroc(score, label, seed=1, n_boot=1000):
    score = np.asarray(score, float); label = np.asarray(label, int)
    pos = np.where(label == 1)[0]; neg = np.where(label == 0)[0]
    m = len(pos)
    rng = np.random.default_rng(seed)
    sel = rng.choice(neg, size=m, replace=False)
    idx = np.concatenate([pos, sel])
    y, s = label[idx], score[idx]
    pt = roc_auc_score(y, s); aupr = average_precision_score(y, s)
    boots = []
    for _ in range(n_boot):
        b = rng.integers(0, len(idx), len(idx))
        if len(np.unique(y[b])) < 2:
            continue
        boots.append(roc_auc_score(y[b], s[b]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return pt, aupr, (lo, hi), m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--model_name_or_path", required=True,
                    help="DNABERT-2 backbone dir (same one training used)")
    ap.add_argument("--eval_csv", required=True)
    ap.add_argument("--ref_fasta", required=True)
    ap.add_argument("--reference_csv", default=None)
    ap.add_argument("--train_coords", nargs="+", default=None,
                    help="fold0/train.meta.csv fold0/dev.meta.csv for the leak-free filter")
    ap.add_argument("--bin_size", type=int, default=100000)
    ap.add_argument("--left_bp", type=int, default=128)
    ap.add_argument("--right_bp", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="refsingle_adastra")
    args = ap.parse_args()

    with open(os.path.join(args.checkpoint_dir, "run_config.json")) as f:
        run_config = json.load(f)
    assert run_config.get("task") == "classification", \
        f"expected task=classification, got {run_config.get('task')}"
    print(f"[cfg] task={run_config.get('task')} main_num_labels={run_config.get('main_num_labels')} "
          f"pooling={run_config.get('pooling_mode')} width={run_config.get('center_pool_width')} "
          f"head_layers={run_config.get('head_num_layers')}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = load_checkpoint(args.model_name_or_path, args.checkpoint_dir, run_config)

    ev = pd.read_csv(args.eval_csv)
    print(f"[load] ADASTRA eval: {len(ev)} rows (pos={int((ev.label==1).sum())}, "
          f"neg={int((ev.label==0).sum())})")

    seqs, keep = build_ref_windows(ev, args.ref_fasta, args.left_bp, args.right_bp)
    ev = ev.loc[keep].reset_index(drop=True)
    ref_seqs = list(np.asarray(seqs)[keep])

    print(f"[score] ref-window P(ASB) on {len(ref_seqs)} variants (single-window, region propensity)...")
    ev["p_asb"] = score_windows(model, tokenizer, ref_seqs, args.batch_size, args.device, args.max_len)

    # leak-free filter against THIS checkpoint's own meta sidecars
    leaky = np.zeros(len(ev), dtype=bool)
    seen = set()
    for path in (args.train_coords or []):
        if not os.path.exists(path):
            print(f"[leakage] WARNING: {path} not found; skipping."); continue
        tc = pd.read_csv(path)
        chrom_col = "chr" if "chr" in tc.columns else tc.columns[0]
        pos_col = next((c for c in ("SNV", "pos", "anchor") if c in tc.columns), None)
        if pos_col is None:
            print(f"[leakage] {path}: no SNV/pos/anchor col ({list(tc.columns)[:6]}); skip."); continue
        before = len(seen)
        seen |= set(zip(tc[chrom_col].astype(str), (tc[pos_col].astype(int) // args.bin_size)))
        print(f"[leakage] {os.path.basename(path)}: +{len(seen)-before} bins (via '{pos_col}'), {len(seen)} total")
    if seen:
        ev_bins = list(zip(ev["chr"].astype(str), ((ev["pos"].astype(int) - 1) // args.bin_size)))
        leaky = np.array([b in seen for b in ev_bins])
        n_pos_leak = int(leaky[ev["label"].to_numpy() == 1].sum())
        print(f"[leakage] {leaky.sum()}/{len(ev)} eval variants ({100*leaky.mean():.2f}%) in a seen bin; "
              f"{n_pos_leak} ASB-positive. Dropped for leak_free.")
    else:
        print("[leakage] no --train_coords given; full-set number only.")

    def report(tag, sub):
        pt, aupr, (lo, hi), m = balanced_auroc(sub["p_asb"].to_numpy(), sub["label"].to_numpy())
        print(f"[AUROC:{tag}] balanced {m} pos + {m} neg  AUROC={pt:.4f}  "
              f"95%CI[{lo:.4f},{hi:.4f}]  AUPRC={aupr:.4f}")
        return {"regime": tag, "auroc": pt, "auroc_lo": lo, "auroc_hi": hi, "auprc": aupr, "n_pos": int(m)}

    results = [report("full", ev)]
    if leaky.any():
        results.append(report("leak_free", ev.loc[~leaky]))

    if args.reference_csv and os.path.exists(args.reference_csv):
        ref = pd.read_csv(args.reference_csv).sort_values("CTCF_AUROC", ascending=False)
        eb2 = results[-1]["auroc"]
        print(f"\n=== ref_single (region-propensity) vs benchmark models (CTCF_AUROC, {len(ref)}) ===")
        placed = False
        for _, r in ref.iterrows():
            if not placed and eb2 >= r["CTCF_AUROC"]:
                print(f"  >>> entexBERT-2 ref_single (this run)   {eb2:.4f}  <<<"); placed = True
            print(f"      {r['model']:20s} {r['family']:9s} {r['CTCF_AUROC']:.4f}")
        if not placed:
            print(f"  >>> entexBERT-2 ref_single (this run)   {eb2:.4f}  (below all listed) <<<")

    ev["leaky"] = leaky
    ev[["chr", "pos", "ref", "alt", "snp", "label", "p_asb", "leaky"]].to_csv(
        f"{args.out}_perVariant.csv.gz", index=False, compression="gzip")
    with open(f"{args.out}_metrics.json", "w") as f:
        json.dump({"results": results, "checkpoint_dir": args.checkpoint_dir,
                   "scoring": "ref_window_only_P(ASB)", "n_scored": int(len(ev)),
                   "leaky_excluded": bool(leaky.any())}, f, indent=2)
    print(f"\n[done] wrote {args.out}_metrics.json + {args.out}_perVariant.csv.gz")


if __name__ == "__main__":
    main()
