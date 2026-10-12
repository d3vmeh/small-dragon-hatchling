"""Sparse-head encoder-weight tail (Appendix A9): for each model, the 99.9th percentile of the encoder weights
and the mean of the largest 0.1 % of them, per head. The sparse head is the lowest-density head of Table 9
(density_training.txt). CPU only: python weight_tail.py > weight_tail.txt"""
import torch, json
MODELS = [("25M bytes", "../lab4090/ckpt_scale_bytes_d256.pt", "25m"), ("50M bytes", "../lab4090/ckpt_scale_bytes_d384.pt", "50m"),
          ("100M bytes", "ckpt100m/ckpt_h100.pt", "100m"), ("25M BPE", "../lab4090/ckpt_scale_bpe_d256.pt", "bpe25m"),
          ("50M BPE", "../lab4090/ckpt_scale_bpe_d384.pt", "bpe50m"), ("100M BPE", "ckpt100m/ckpt_100m_bpe.pt", "bpe100m")]
def tensor(sd, name):
    for k, v in sd.items():
        if k.endswith(name): return v.float()
    raise KeyError(name)
print("model | sparse head (0-indexed, lowest density) | 99.9th percentile per head | mean of top 0.1 % per head")
for label, path, tag in MODELS:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd["model"] if "model" in sd else sd
    enc = tensor(sd, "encoder")                       # (heads, D, N)
    dens = json.load(open(f"scope_{tag}.json"))["sparsity"]
    head_mean = [sum(r[h] for r in dens) / len(dens) for h in range(len(dens[0]))]
    sparse = min(range(len(head_mean)), key=lambda h: head_mean[h])
    p999, top = [], []
    for h in range(enc.shape[0]):
        w = enc[h].flatten()
        p999.append(torch.quantile(w, 0.999).item())
        k = max(1, int(w.numel() * 0.001)); top.append(torch.topk(w, k).values.mean().item())
    print(f"{label} | h{sparse} | " + " ".join(f"{x:.3f}" for x in p999) + " | " + " ".join(f"{x:.3f}" for x in top))
