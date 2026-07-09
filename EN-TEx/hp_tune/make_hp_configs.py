#!/usr/bin/env python3
"""
make_hp_configs.py — sample N reproducible random hyperparameter configs for the ref_single
BASELINE (num_aux_tasks=0) tuning search (Bergstra & Bengio 2012: random > grid at equal budget).

Core space (the knobs that move DNABERT-2 fine-tuning; LoRA off = full fine-tune):
  learning_rate                 log-uniform in [1e-5, 5e-4]
  per_device_train_batch_size   {8, 16, 32}
  num_train_epochs              {2, 3, 4}
  weight_decay                  {0.0, 0.01, 0.1}
  warmup_steps                  {0, 50, 100}
  head_num_layers               {1, 2}       # 1 = linear head, 2 = MLP head

A fixed --sampler_seed makes the config list byte-reproducible. Writes a TSV, one config per line:
  config_id  learning_rate  batch_size  epochs  weight_decay  warmup_steps  head_num_layers

Usage:
  python make_hp_configs.py --n 20 --sampler_seed 0 --out hp/hp_configs.tsv
"""
import argparse, math, random

COLS = ["config_id", "learning_rate", "batch_size", "epochs",
        "weight_decay", "warmup_steps", "head_num_layers"]


def sample_config(rng):
    lr = 10 ** rng.uniform(math.log10(1e-5), math.log10(5e-4))   # log-uniform
    return {
        "learning_rate": f"{lr:.3e}",
        "batch_size": rng.choice([8, 16, 32]),
        "epochs": rng.choice([2, 3, 4]),
        "weight_decay": rng.choice([0.0, 0.01, 0.1]),
        "warmup_steps": rng.choice([0, 50, 100]),
        "head_num_layers": rng.choice([1, 2]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of random configs")
    ap.add_argument("--sampler_seed", type=int, default=0, help="seed for the config sampler (reproducible)")
    ap.add_argument("--out", default="hp_configs.tsv")
    args = ap.parse_args()

    rng = random.Random(args.sampler_seed)
    seen, rows = set(), []
    # sample distinct configs (retry on collision, bounded)
    tries = 0
    while len(rows) < args.n and tries < args.n * 100:
        tries += 1
        c = sample_config(rng)
        key = tuple(sorted(c.items()))
        if key in seen:
            continue
        seen.add(key)
        c["config_id"] = f"cfg{len(rows):03d}"
        rows.append(c)

    with open(args.out, "w") as f:
        f.write("\t".join(COLS) + "\n")
        for c in rows:
            f.write("\t".join(str(c[k]) for k in COLS) + "\n")
    print(f"wrote {len(rows)} configs to {args.out} (sampler_seed={args.sampler_seed})")


if __name__ == "__main__":
    main()
