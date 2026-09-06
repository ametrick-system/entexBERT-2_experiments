#!/usr/bin/env python3
"""
staircase_probe.py — the DNABERT-2 "staircase" probe for CTCF.

ONE protocol, applied to THREE trunk states x TWO window sets, so every rung differs
from its neighbour by exactly one controlled thing. The trunk is always FROZEN; only the
small probe on top is trained. This isolates what the pretrained weights themselves encode,
before any fine-tuning — the missing rungs below the already-verified fine-tuned numbers.

TRUNK STATES (all loaded via AutoModel, so no dependency on the entexbert2 refactor version):
  random   : DNABERT-2 architecture, RANDOM weights (fixed seed). The "no pretraining" floor.
  pretrained: DNABERT-2-117M as published. Pretraining alone.
  finetuned : + the user's CTCF Stage-1 binding trunk overlaid on the backbone. Pretraining
              + binding supervision. (Backbone keys only; any head keys are ignored.)

WINDOW SETS:
  binding : peak-vs-background windows from a multi_tissue_peak build (run_experiment).
            y = (feature_type == "peak").  Single window per site (input_mode single -> 'sequence').
            Two questions scored:
              (1) peak-vs-background AUROC        -- "does the trunk SEE binding sites?"
              (2) peak-strength Spearman           -- ridge on pooled emb -> binding_label_raw,
                                                      Spearman among test peaks.
            Two probe styles on the SAME frozen embeddings:
              linear : logistic regression on the POOLED (center_mean w=5) vector -- strict floor.
              cnn    : BEND-style 2-layer Conv1d over the PER-TOKEN embedding sequence
                       (sees local structure the pooled vector discards).
  asb     : ADASTRA ref/alt pairs (sequence1=ref window, sequence2=alt window). Twin-probe:
            freeze trunk, learn a projection P (768->d) + logistic-on-distance a,b, train on
            EN-TEx Stage-2 ASB pairs, score ADASTRA leak-free (balanced bootstrap AUROC).
            Plus a training-free ZERO-SHOT rung: s = ||pool(ref) - pool(alt)|| ranked directly
            (pretrained trunk only) -- the untrained twin, BPE-agnostic.

The already-verified fine-tuned ASB rungs (ref_single 0.5796, paired twin 0.7208, +stem 0.7271)
are NOT recomputed here; they are merged into the figure downstream from their saved numbers.

OUTPUT: <out>_staircase_metrics.json  — every rung with AUROC/CI/n (+ Spearman for binding).

Run (eb2 env, one GPU):
  python staircase_probe.py \
    --backbone_path   $HOME/entexBERT-2/DNABERT-2-117M-attention \
    --finetuned_trunk .../binding_reg_ENC-002_CTCF_.../runs/reg \
    --binding_dir     .../staircase/binding_windows \
    --asb_train_dir   .../stage2_ctcf_clf_stem/inputs_clf \
    --adastra_pairs   .../staircase/adastra_pairs.csv \
    --adastra_eval_csv ctcf_adastra_evalset.csv.gz \
    --train_coords    .../fold0/train.meta.csv .../fold0/dev.meta.csv \
    --reference_csv   ctcf_benchmark_reference_auroc.csv \
    --out             staircase_ctcf
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
from transformers import AutoModel, AutoConfig, AutoTokenizer

CENTER_POOL_WIDTH = 5


# ----------------------------------------------------------------------
# Trunk loading — three states, all as a bare AutoModel backbone.
# ----------------------------------------------------------------------
def load_trunk(state, backbone_path, finetuned_trunk=None, seed=0, device="cuda"):
    if state == "random":
        cfg = AutoConfig.from_pretrained(backbone_path, trust_remote_code=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = AutoModel.from_config(cfg, trust_remote_code=True)
        print(f"[trunk:random] fresh AutoModel.from_config (seed={seed}) — NO pretrained weights")
    elif state == "pretrained":
        model = AutoModel.from_pretrained(backbone_path, trust_remote_code=True)
        print(f"[trunk:pretrained] AutoModel.from_pretrained({os.path.basename(backbone_path)})")
    elif state == "finetuned":
        model = AutoModel.from_pretrained(backbone_path, trust_remote_code=True)
        sd = torch.load(os.path.join(finetuned_trunk, "pytorch_model.bin"), map_location="cpu")
        # Overlay ONLY backbone weights. The finetuned checkpoint prefixes the trunk as
        # 'backbone.' (entexbert2) or 'bert.' (pre-refactor); strip either to bare keys that
        # match this AutoModel's own state_dict. Head keys (main_head.*, proj.*, dist_*, logvar_*)
        # are dropped — we only want the trunk.
        want = set(model.state_dict().keys())
        overlay, dropped = {}, []
        for k, v in sd.items():
            bare = k
            for pre in ("backbone.", "bert."):
                if bare.startswith(pre):
                    bare = bare[len(pre):]
                    break
            if bare in want:
                overlay[bare] = v
            else:
                dropped.append(k)
        missing, unexpected = model.load_state_dict(overlay, strict=False)
        # Every backbone tensor must be covered by the overlay; missing backbone keys = silent
        # partial-random trunk, which we refuse.
        real_missing = [k for k in missing]
        print(f"[trunk:finetuned] overlaid {len(overlay)}/{len(want)} backbone tensors "
              f"(dropped {len(dropped)} head keys); missing={len(real_missing)} unexpected={len(unexpected)}")
        assert not real_missing, (
            f"finetuned overlay left {len(real_missing)} backbone tensors random: {real_missing[:8]}. "
            "Refusing to probe a partially-random trunk.")
    else:
        raise ValueError(state)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ----------------------------------------------------------------------
# Embedding: per-token hidden states + center_mean pooled vector.
# ----------------------------------------------------------------------
def token_maxlen(tokenizer, seqs, cap, step=1024):
    """One CPU-only tokenization pass to find the true max token length (<= cap). Padding the
    memmap to this instead of `cap` roughly halves disk/IO when BPE tokens < cap."""
    L = 1
    for i in range(0, len(seqs), step):
        enc = tokenizer(seqs[i:i + step], truncation=True, max_length=cap)
        L = max(L, max(len(x) for x in enc["input_ids"]))
    return int(L)


@torch.no_grad()
def embed(model, tokenizer, seqs, batch_size, device, max_len):
    """Pooled (N,H) float32 only. center_mean pool of CENTER_POOL_WIDTH tokens at valid//2.
    Used for the pooled-vector probes (linear floor, twin, zero-shot) — all small in RAM."""
    pooled_all = []
    half = CENTER_POOL_WIDTH // 2
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=attn, return_dict=True)
        seq = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]  # (B,L,H)
        B = seq.shape[0]
        pooled = []
        for b in range(B):
            valid = max(int(attn[b].sum().item()), 1)
            c = valid // 2
            s = max(0, c - half); e = min(valid, c + half + 1)
            pooled.append(seq[b, s:e, :].mean(dim=0))
        pooled_all.append(torch.stack(pooled).float().cpu().numpy())
        if (i // batch_size) % 50 == 0:
            print(f"  [embed] {i+len(chunk)}/{len(seqs)}", flush=True)
    return np.concatenate(pooled_all, 0)


@torch.no_grad()
def embed_tokens_memmap(model, tokenizer, seqs, batch_size, device, L, hidden, prefix):
    """ONE forward pass that (a) streams per-token embeddings (N,L,H) fp16 + mask (N,L) int8 to
    DISK memmaps and (b) returns pooled (N,H) float32 in RAM. This bounds host RAM for the CNN
    probe: the full per-token tensor never lives in memory (that was the OOM). The CNN reads
    mini-batches back from the memmap; on a node with adequate RAM the OS page-caches it after
    the first epoch, so training stays fast while process RSS stays low (page cache is evictable)."""
    N = len(seqs)
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    tokf, maskf = prefix + ".tok.f16", prefix + ".mask.i8"
    tok_mm = np.memmap(tokf, dtype=np.float16, mode="w+", shape=(N, L, hidden))
    mask_mm = np.memmap(maskf, dtype=np.int8, mode="w+", shape=(N, L))
    pooled = np.empty((N, hidden), dtype=np.float32)
    half = CENTER_POOL_WIDTH // 2
    for i in range(0, N, batch_size):
        chunk = seqs[i:i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=L)
        ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=attn, return_dict=True)
        seq = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]  # (B,L,H)
        b1 = i + seq.shape[0]
        tok_mm[i:b1] = seq.half().cpu().numpy()
        mask_mm[i:b1] = attn.to(torch.int8).cpu().numpy()
        for b in range(seq.shape[0]):
            valid = max(int(attn[b].sum().item()), 1)
            c = valid // 2
            s = max(0, c - half); e = min(valid, c + half + 1)
            pooled[i + b] = seq[b, s:e, :].mean(dim=0).float().cpu().numpy()
        if (i // batch_size) % 50 == 0:
            print(f"  [embed-mm] {b1}/{N}", flush=True)
    tok_mm.flush(); mask_mm.flush()
    return pooled, tok_mm, mask_mm, (tokf, maskf)


# ----------------------------------------------------------------------
# Probes
# ----------------------------------------------------------------------
class CNNProbe(nn.Module):
    """BEND-style 2-layer Conv1d over the per-token embedding sequence -> masked mean -> logit."""
    def __init__(self, hidden, c1=128, c2=64, k=3):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden, c1, k, padding=k // 2)
        self.conv2 = nn.Conv1d(c1, c2, k, padding=k // 2)
        self.act = nn.ReLU()
        self.head = nn.Linear(c2, 1)

    def forward(self, tok, mask):
        # tok (B,L,H) -> (B,H,L)
        x = tok.transpose(1, 2)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))            # (B,c2,L)
        m = mask.unsqueeze(1).float()          # (B,1,L)
        x = (x * m).sum(-1) / m.sum(-1).clamp(min=1.0)   # masked mean -> (B,c2)
        return self.head(x).squeeze(-1)        # (B,)


def train_cnn_probe(tok_tr, mask_tr, y_tr, tok_te, mask_te, hidden, device,
                    epochs=15, bs=128, lr=1e-3, seed=0):
    """tok_tr/mask_tr/tok_te/mask_te are DISK memmaps (N,L,H)/(N,L). Mini-batches are pulled
    from disk per step, so RAM stays bounded regardless of N. Indices are sorted within a batch
    for sequential-ish memmap reads; the epoch order is still shuffled across batches."""
    torch.manual_seed(seed)
    probe = CNNProbe(hidden).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    ytr = torch.tensor(y_tr, dtype=torch.float32)
    n = tok_tr.shape[0]
    for ep in range(epochs):
        probe.train()
        perm = np.random.default_rng(seed + ep).permutation(n)
        for j in range(0, n, bs):
            idx = np.sort(perm[j:j + bs])                       # sorted -> faster memmap read
            xb = torch.from_numpy(np.ascontiguousarray(tok_tr[idx])).float().to(device)
            mb = torch.from_numpy(np.ascontiguousarray(mask_tr[idx])).float().to(device)
            logit = probe(xb, mb)
            loss = lossf(logit, ytr[torch.from_numpy(idx)].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    scores = []
    with torch.no_grad():
        for j in range(0, tok_te.shape[0], bs):
            xb = torch.from_numpy(np.ascontiguousarray(tok_te[j:j + bs])).float().to(device)
            mb = torch.from_numpy(np.ascontiguousarray(mask_te[j:j + bs])).float().to(device)
            scores.append(torch.sigmoid(probe(xb, mb)).cpu().numpy())
    return np.concatenate(scores)


class TwinProbe(nn.Module):
    """Frozen-trunk twin: projection P(768->d), s=||P(z1)-P(z2)||, p=sigma(a*s+b)."""
    def __init__(self, hidden, d=128):
        super().__init__()
        self.proj = nn.Linear(hidden, d)
        self.a = nn.Parameter(torch.tensor(0.5413))   # softplus(a)~1 init, matches model.py
        self.b = nn.Parameter(torch.tensor(0.0))

    def dist(self, z1, z2):
        p1 = self.proj(z1); p2 = self.proj(z2)
        return torch.linalg.vector_norm(p1 - p2, dim=-1)

    def forward(self, z1, z2):
        s = self.dist(z1, z2)
        return torch.nn.functional.softplus(self.a) * s + self.b


def train_twin_probe(z1_tr, z2_tr, y_tr, z1_te, z2_te, hidden, device,
                     d=128, epochs=30, bs=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    probe = TwinProbe(hidden, d).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    z1t = torch.tensor(z1_tr, dtype=torch.float32); z2t = torch.tensor(z2_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    n = z1t.shape[0]
    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(n)
        for j in range(0, n, bs):
            idx = perm[j:j + bs]
            logit = probe(z1t[idx].to(device), z2t[idx].to(device))
            loss = lossf(logit, yt[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        s = probe(torch.tensor(z1_te, dtype=torch.float32).to(device),
                  torch.tensor(z2_te, dtype=torch.float32).to(device))
    return torch.sigmoid(s).cpu().numpy()


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def balanced_auroc(score, label, seed=1, n_boot=1000):
    score = np.asarray(score, float); label = np.asarray(label, int)
    pos = np.where(label == 1)[0]; neg = np.where(label == 0)[0]
    m = min(len(pos), len(neg))
    rng = np.random.default_rng(seed)
    p_sel = rng.choice(pos, size=m, replace=False) if len(pos) > m else pos
    n_sel = rng.choice(neg, size=m, replace=False) if len(neg) > m else neg
    idx = np.concatenate([p_sel, n_sel])
    y, s = label[idx], score[idx]
    pt = roc_auc_score(y, s); aupr = average_precision_score(y, s)
    boots = []
    for _ in range(n_boot):
        b = rng.integers(0, len(idx), len(idx))
        if len(np.unique(y[b])) < 2:
            continue
        boots.append(roc_auc_score(y[b], s[b]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return float(pt), float(aupr), (float(lo), float(hi)), int(m)


def rung(name, score, label):
    pt, aupr, (lo, hi), m = balanced_auroc(score, label)
    print(f"    [{name}] AUROC={pt:.4f} 95%CI[{lo:.4f},{hi:.4f}] AUPRC={aupr:.4f} (balanced {m}+{m})")
    return {"rung": name, "auroc": pt, "auroc_lo": lo, "auroc_hi": hi, "auprc": aupr, "n_pos": m}


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_binding(binding_dir):
    """train/test minimal (sequence,label unused) + meta (feature_type, binding_label_raw)."""
    out = {}
    for split in ("train", "test"):
        mini = pd.read_csv(os.path.join(binding_dir, f"{split}.csv"))
        meta = pd.read_csv(os.path.join(binding_dir, f"{split}.meta.csv"))
        assert len(mini) == len(meta), f"{split}: minimal/meta row mismatch {len(mini)} vs {len(meta)}"
        seqcol = "sequence" if "sequence" in mini.columns else mini.columns[0]
        df = pd.DataFrame({
            "sequence": mini[seqcol].astype(str),
            "y_peak": (meta["feature_type"].astype(str) == "peak").astype(int),
            "strength": meta["binding_label_raw"].astype(float) if "binding_label_raw" in meta else np.nan,
        })
        out[split] = df.reset_index(drop=True)
        print(f"[binding:{split}] {len(df)} rows, peaks={int(df.y_peak.sum())}, bg={int((df.y_peak==0).sum())}")
    return out


def load_asb_train(asb_train_dir):
    dfs = []
    for split in ("train", "dev"):
        p = os.path.join(asb_train_dir, f"{split}.csv")
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    df = pd.concat(dfs, ignore_index=True)
    assert {"sequence1", "sequence2", "label"} <= set(df.columns), \
        f"ASB train needs sequence1/sequence2/label; got {list(df.columns)[:8]}"
    df = df[["sequence1", "sequence2", "label"]].dropna().reset_index(drop=True)
    print(f"[asb:train] {len(df)} pairs, pos={int((df.label==1).sum())}")
    return df


def leakfree_mask(pairs, train_coords, bin_size):
    """Bin-level leak filter on the ADASTRA pairs using the fine-tuned checkpoint's meta sidecars."""
    seen = set()
    for path in (train_coords or []):
        if not os.path.exists(path):
            print(f"[leak] WARNING {path} missing; skip"); continue
        tc = pd.read_csv(path)
        cc = "chr" if "chr" in tc.columns else tc.columns[0]
        pc = next((c for c in ("SNV", "pos", "anchor") if c in tc.columns), None)
        if pc is None:
            print(f"[leak] {path}: no pos col; skip"); continue
        seen |= set(zip(tc[cc].astype(str), tc[pc].astype(int) // bin_size))
    if not seen:
        print("[leak] no coords -> full-set only")
        return np.zeros(len(pairs), bool)
    bins = list(zip(pairs["chr"].astype(str), (pairs["pos"].astype(int) - 1) // bin_size))
    leaky = np.array([b in seen for b in bins])
    print(f"[leak] {leaky.sum()}/{len(pairs)} ({100*leaky.mean():.1f}%) in a seen bin; "
          f"{int(leaky[pairs['label'].to_numpy()==1].sum())} positive")
    return leaky


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone_path", required=True)
    ap.add_argument("--finetuned_trunk", required=True, help="CTCF Stage-1 run dir with pytorch_model.bin")
    ap.add_argument("--binding_dir", required=True)
    ap.add_argument("--asb_train_dir", required=True)
    ap.add_argument("--adastra_pairs", required=True, help="CSV: sequence1,sequence2,label,chr,pos")
    ap.add_argument("--train_coords", nargs="+", default=None)
    ap.add_argument("--reference_csv", default=None)
    ap.add_argument("--bin_size", type=int, default=100000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--proj_dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="staircase_ctcf")
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.backbone_path, trust_remote_code=True)
    hidden = AutoConfig.from_pretrained(args.backbone_path, trust_remote_code=True).hidden_size

    binding = load_binding(args.binding_dir)
    asb_tr = load_asb_train(args.asb_train_dir)
    pairs = pd.read_csv(args.adastra_pairs)
    assert {"sequence1", "sequence2", "label"} <= set(pairs.columns), \
        f"ADASTRA pairs need sequence1/sequence2/label; got {list(pairs.columns)[:8]}"
    have_coords = {"chr", "pos"} <= set(pairs.columns)
    print(f"[asb:adastra] {len(pairs)} pairs, pos={int((pairs.label==1).sum())} "
          f"(chr/pos present={have_coords})")
    # Leak filtering: if the pairs carry chr/pos AND --train_coords is given, filter here (one place,
    # transparent). Otherwise the pairs are assumed already leak-free (built with --drop_leaky against
    # the twin probe's own EN-TEx training bins) and every row is scored as-is.
    if have_coords and args.train_coords:
        leaky = leakfree_mask(pairs, args.train_coords, args.bin_size)
    else:
        leaky = np.zeros(len(pairs), bool)
        print("[leak] pairs assumed pre-filtered leak-free (no chr/pos or no --train_coords)")

    metrics = {"binding": {}, "asb": {}, "config": vars(args)}

    for state in ("random", "pretrained", "finetuned"):
        print(f"\n===== TRUNK: {state} =====")
        model = load_trunk(state, args.backbone_path,
                           finetuned_trunk=args.finetuned_trunk, seed=args.seed, device=dev)

        # ---------- BINDING ----------
        # Per-token embeddings stream to DISK memmaps (bounds RAM; this was the OOM). Size the
        # memmap to the true max BPE token length (<= max_len) found in one CPU pass.
        Ltr = token_maxlen(tok, binding["train"].sequence.tolist(), args.max_len)
        Lte = token_maxlen(tok, binding["test"].sequence.tolist(), args.max_len)
        print(f"  [binding] token maxlen train={Ltr} test={Lte} (cap {args.max_len}); streaming to memmap")
        mmdir = os.path.join(os.path.dirname(args.out) or ".", f"_mm_{state}")
        ptr, ttr, mtr, ftr = embed_tokens_memmap(model, tok, binding["train"].sequence.tolist(),
                                                 args.batch_size, dev, Ltr, hidden, os.path.join(mmdir, "tr"))
        pte, tte, mte, fte = embed_tokens_memmap(model, tok, binding["test"].sequence.tolist(),
                                                 args.batch_size, dev, Lte, hidden, os.path.join(mmdir, "te"))
        ytr = binding["train"].y_peak.to_numpy(); yte = binding["test"].y_peak.to_numpy()

        # linear floor (pooled)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(ptr, ytr)
        s_lin = clf.predict_proba(pte)[:, 1]
        r_lin = rung(f"binding/{state}/linear", s_lin, yte)
        # BEND CNN (per-token, read from memmap)
        s_cnn = train_cnn_probe(ttr, mtr, ytr, tte, mte, hidden, dev, seed=args.seed)
        r_cnn = rung(f"binding/{state}/cnn", s_cnn, yte)
        # free the memmaps for this trunk state before the next
        del ttr, mtr, tte, mte
        for f in (*ftr, *fte):
            try:
                os.remove(f)
            except OSError:
                pass
        # peak-strength Spearman (ridge on pooled -> binding_label_raw among test peaks)
        str_tr = binding["train"].strength.to_numpy(); str_te = binding["test"].strength.to_numpy()
        spear = None
        pk = (yte == 1) & np.isfinite(str_te)
        if np.isfinite(str_tr).sum() > 50 and pk.sum() > 20:
            rg = Ridge(alpha=1.0).fit(ptr[np.isfinite(str_tr)], str_tr[np.isfinite(str_tr)])
            pred = rg.predict(pte[pk])
            rho, _ = spearmanr(pred, str_te[pk])
            spear = float(rho)
            print(f"    [binding/{state}/strength] Spearman={spear:.4f} (test peaks n={int(pk.sum())})")
        metrics["binding"][state] = {"linear": r_lin, "cnn": r_cnn, "strength_spearman": spear}

        # ---------- ASB ----------
        # zero-shot untrained twin distance (pretrained only): rank ||pool(ref)-pool(alt)||
        if state == "pretrained":
            print(f"  [asb] zero-shot embedding distance (no training)")
            zr = embed(model, tok, pairs.sequence1.tolist(), args.batch_size, dev, args.max_len)
            za = embed(model, tok, pairs.sequence2.tolist(), args.batch_size, dev, args.max_len)
            d0 = np.linalg.norm(zr - za, axis=1)
            zmask = ~leaky if leaky.any() else np.ones(len(pairs), bool)
            metrics["asb"]["zeroshot_distance"] = rung("asb/pretrained/zeroshot_dist",
                                                       d0[zmask], pairs.label.to_numpy()[zmask])

        # frozen twin-probe (random + pretrained; finetuned twin == the verified paired rung)
        if state in ("random", "pretrained"):
            print(f"  [asb] frozen twin-probe (train EN-TEx pairs -> score ADASTRA)")
            z1tr = embed(model, tok, asb_tr.sequence1.tolist(), args.batch_size, dev, args.max_len)
            z2tr = embed(model, tok, asb_tr.sequence2.tolist(), args.batch_size, dev, args.max_len)
            z1te = embed(model, tok, pairs.sequence1.tolist(), args.batch_size, dev, args.max_len)
            z2te = embed(model, tok, pairs.sequence2.tolist(), args.batch_size, dev, args.max_len)
            s_tw = train_twin_probe(z1tr, z2tr, asb_tr.label.to_numpy(), z1te, z2te,
                                    hidden, dev, d=args.proj_dim, seed=args.seed)
            zmask = ~leaky if leaky.any() else np.ones(len(pairs), bool)
            metrics["asb"][f"twin_{state}"] = rung(f"asb/{state}/twin_probe",
                                                   s_tw[zmask], pairs.label.to_numpy()[zmask])

        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    with open(f"{args.out}_staircase_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[done] wrote {args.out}_staircase_metrics.json")


if __name__ == "__main__":
    main()
