import torch, collections
sd = torch.load("/home/asm242/palmer_scratch/entexBERT-2_experiments/EN-TEx/experiments/refsingle_ENC-002_CTCF_ctcf_ref_single_classification_binsplit2_cw/runs/clf/pytorch_model.bin", map_location="cpu")
# show only the NON-backbone keys (the head) + their shapes
for k, v in sd.items():
    if not k.startswith("backbone.") and not k.startswith("bert."):
        print(k, tuple(v.shape))