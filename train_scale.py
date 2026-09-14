# Scaling series: the byte-100M recipe (train_100m.py / train_h100.py) scaled by width.
# Env: D=256|384 (width), BPP=54 (bytes per parameter -> total bytes), MICRO.
# MODEL=gpt -> parameter-matched transformer baseline (gpt.py): D = n_embd
# (n_layer=8, n_head=D//64; D=512 -> 25.4M, matches the D=256 BDH's 25.3M),
# checkpoint ckpt_scale_gpt_{ARM}_d{D}.pt with "arch": "gpt".
# Data: corpus_mix.bin (uint8 bytes, the matched-text corpus). SMOKE=1 = rehearsal.
# Pause-friendly: atomic checkpoints every 2k steps; rerun same command to
# resume (schedule continues from the saved step).
#   D=256 ARM=bytes PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u train_scale.py
import math
import os
import time

import bdh
import gpt
import numpy as np
import torch

assert torch.cuda.is_available()
device = torch.device("cuda")
ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
SEED = int(os.environ.get("SEED", "1337")); torch.manual_seed(SEED)   # SEED=N for a second training seed (ckpt gets _sN)
torch.backends.cuda.matmul.allow_tf32 = True

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = bool(os.environ.get("SMOKE"))
D = int(os.environ.get("D", "256"))
BPP = float(os.environ.get("BPP", "54"))          # bytes of TEXT per parameter (both arms)
ARM = os.environ.get("ARM", "bytes")               # bytes | bpe  (matched TEXT: same byte budget)
MODEL = os.environ.get("MODEL", "bdh")             # bdh | gpt (transformer baseline, gpt.py)
BYTES_PER_TOKEN = 3.83                             # corpus_mix_bpe measured ratio
DATA = os.path.join(HERE, "corpus_mix.bin" if ARM == "bytes" else "corpus_mix_bpe.bin")
CKPT = os.path.join(HERE, f"ckpt_scale_{'gpt_' if MODEL == 'gpt' else ''}{ARM}_d{D}{'' if os.environ.get('LAYERS', '6') == '6' else '_L' + os.environ['LAYERS']}{'' if os.environ.get('SEED', '1337') == '1337' else '_s' + os.environ['SEED']}.pt")

if MODEL == "gpt":
    CONFIG = gpt.GPTConfig(n_layer=8, n_embd=D, n_head=D // 64, block_size=512,
                           vocab_size=256 if ARM == "bytes" else 8192)
else:
    CONFIG = bdh.BDHConfig(n_layer=int(os.environ.get("LAYERS", 6)), n_embd=D, n_head=4,
                           mlp_internal_dim_multiplier=128, vocab_size=256 if ARM == "bytes" else 8192)
MICRO = int(os.environ.get("MICRO", "8"))
BLOCK, ACCUM = 512, 32 // MICRO         # 32 seqs/step = 16,384 bytes/step
_n_params = None                        # set after model build; MAX_STEPS derived below
MAX_STEPS = 300 if SMOKE else int(os.environ.get("MAX_STEPS", "0"))
PEAK_LR, MIN_LR, WARMUP = 1e-3, 1e-4, 60 if SMOKE else 1500
LOG, CKPT_FREQ = 50 if SMOKE else 200, 100 if SMOKE else 2000

data = np.memmap(DATA, dtype=np.uint8 if ARM == "bytes" else np.uint16, mode="r")
n_val = len(data) // 100
train_data, val_data = data[:-n_val], data[-n_val:]
print(f"corpus: {len(data)/1e9:.2f}G atoms ({ARM})", flush=True)


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


model = (gpt.GPT(CONFIG) if MODEL == "gpt" else bdh.BDH(CONFIG)).to(device)
_n_params = sum(p.numel() for p in model.parameters())
# parameter count for the byte-budget uses the NON-embedding core so both arms get the same text
_core = _n_params - (CONFIG.vocab_size * D * 2 if ARM == "bpe" else 0)
_text_bytes = _core * BPP
_atoms_per_step = 32 * BLOCK
if not MAX_STEPS:
    MAX_STEPS = int(_text_bytes / _atoms_per_step / (1 if ARM == "bytes" else BYTES_PER_TOKEN))
print(f"model: {_n_params/1e6:.1f}M params ({_n_params:,}; {MODEL}, D={D}, {ARM}) | {BPP} bytes/param on {_core/1e6:.1f}M core -> "
      f"{_text_bytes/1e9:.2f}G bytes of text = {MAX_STEPS} steps x {_atoms_per_step} atoms", flush=True)
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
                "step": step, "config": CONFIG.__dict__, "arch": MODEL}, tmp)
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
