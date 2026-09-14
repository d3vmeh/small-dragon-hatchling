# Matched-TEXT BPE-100M: same architecture as the H100 byte baseline
# (D=512), same text exposure (~5.4B bytes' worth = ~1.41B tokens), BPE-8k
# tokens. About 22 h on one RTX 4090. SMOKE=1 = 10-minute rehearsal.
# Pause-friendly: atomic checkpoints every 2k steps; rerun same command to
# resume (schedule continues from the saved step).
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u train_100m.py
import math
import os
import time

import bdh
import numpy as np
import torch

assert torch.cuda.is_available()
device = torch.device("cuda")
ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = bool(os.environ.get("SMOKE"))
DATA = os.path.join(HERE, "corpus_mix_bpe.bin")
CKPT = os.path.join(HERE, "ckpt_100m_bpe.pt")

CONFIG = bdh.BDHConfig(n_layer=6, n_embd=512, n_head=4,
                       mlp_internal_dim_multiplier=128, vocab_size=8192)
BLOCK, MICRO, ACCUM = 512, 4, 8         # 32 seqs/step on 24GB at D=512
MAX_STEPS = 300 if SMOKE else 86_000    # 86k x 16,384 tokens ~ 1.41B tokens
PEAK_LR, MIN_LR, WARMUP = 1e-3, 1e-4, 60 if SMOKE else 1500
LOG, CKPT_FREQ = 50 if SMOKE else 200, 100 if SMOKE else 2000

data = np.memmap(DATA, dtype=np.uint16, mode="r")
n_val = len(data) // 100
train_data, val_data = data[:-n_val], data[-n_val:]
print(f"corpus: {len(data)/1e6:.0f}M tokens", flush=True)


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK - 1, (MICRO,))
    x = torch.stack([torch.from_numpy(d[i:i + BLOCK].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i + 1:i + 1 + BLOCK].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def lr_at(step):
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    p = min((step - WARMUP) / max(MAX_STEPS - WARMUP, 1), 1.0)
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * p))


model = bdh.BDH(CONFIG).to(device)
print(f"model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params", flush=True)
model = torch.compile(model)
opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=0.1)

start = 0
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    start = ck["step"] + 1
    print(f"RESUMED from step {ck['step']}", flush=True)


def save_ckpt(step):
    tmp = CKPT + ".tmp"
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "step": step, "config": CONFIG.__dict__}, tmp)
    os.replace(tmp, CKPT)          # atomic replace


@torch.no_grad()
def val_loss():
    model.eval()
    losses = []
    for _ in range(10):
        x, y = get_batch("val")
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


t0, last = time.time(), time.time()
for step in range(start, MAX_STEPS):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    opt.zero_grad(set_to_none=True)
    for _ in range(ACCUM):
        x, y = get_batch("train")
        with ctx:
            _, loss = model(x, y)
        (loss / ACCUM).backward()
    opt.step()
    if step % LOG == 0:
        rate = LOG / max(time.time() - last, 1e-9)
        last = time.time()
        print(f"step {step:6d}/{MAX_STEPS}  loss {loss.item():.4f}  "
              f"{rate:.2f} it/s  eta {(MAX_STEPS-step)/max(rate,1e-9)/3600:.1f}h",
              flush=True)
    if step > start and step % CKPT_FREQ == 0:
        save_ckpt(step)
        print(f"step {step:6d}  VAL {val_loss():.4f}  [ckpt]", flush=True)

save_ckpt(MAX_STEPS - 1)
print(f"FINAL val {val_loss():.4f}, {(time.time()-t0)/3600:.2f}h", flush=True)
