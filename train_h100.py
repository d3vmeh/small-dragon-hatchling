# BDH phase-B run: ~100M params on FineWeb-Edu + TinyStories, tuned for one H100 80GB.
# Self-contained: streams its own data, checkpoints and resumes, and uploads
# checkpoints to a private HF repo so a lost pod costs only minutes.
#
# On the pod (RunPod "PyTorch" template, CUDA):
#   export HF_TOKEN=hf_...              # write token  (required for backup)
#   export HF_CKPT_REPO=YOURNAME/bdh-100m-run
#   python -u train_h100.py 2>&1 | tee train.log
import math
import os
import threading
import time

import bdh
import numpy as np
import requests
import torch

assert torch.cuda.is_available(), "refusing to start: no GPU"
device = torch.device("cuda")
ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)

# ---- config: ~100M params ----
CONFIG = bdh.BDHConfig(n_layer=6, n_embd=512, n_head=4,
                       mlp_internal_dim_multiplier=128, vocab_size=256)
BLOCK = 512
MICRO_BATCH = 16          # per forward pass; lower to 8 if OOM
ACCUM = 4                 # effective batch = 64 sequences = 65,536 bytes/step
MAX_STEPS = 180_000       # ~11.8B bytes total
PEAK_LR = 1e-3
WARMUP = 2_000
MIN_LR = 1e-4
WEIGHT_DECAY = 0.1
LOG_FREQ = 100
CKPT_FREQ = 2_000
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt_h100.pt")
DATA = os.path.join(HERE, "corpus.bin")
TARGET_BYTES = 13_000_000_000        # ~10.8B FineWeb-Edu + ~2.2B TinyStories

# SMOKE=1: a 15-minute rehearsal of the full pipeline on a small corpus,
# writing to separate files.
if os.environ.get("SMOKE"):
    MAX_STEPS, CKPT_FREQ, LOG_FREQ, WARMUP = 300, 100, 20, 50
    TARGET_BYTES = 300_000_000
    CKPT = os.path.join(HERE, "ckpt_smoke.pt")
    DATA = os.path.join(HERE, "corpus_smoke.bin")

TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"


def build_corpus():
    if os.path.exists(DATA):
        print(f"corpus exists: {os.path.getsize(DATA)/1e9:.1f}GB", flush=True)
        return
    t0 = time.time()
    with open(DATA + ".tmp", "wb") as f:
        print("downloading TinyStories...", flush=True)
        with requests.get(TINYSTORIES_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
        print(f"tinystories done ({f.tell()/1e9:.1f}GB); streaming FineWeb-Edu...",
              flush=True)
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        for doc in ds:
            f.write(doc["text"].encode("utf-8"))
            f.write(b"\n<|endoftext|>\n")
            if f.tell() >= TARGET_BYTES:
                break
            if f.tell() % (1 << 30) < 4096:
                print(f"  corpus: {f.tell()/1e9:.1f}GB "
                      f"({(time.time()-t0)/60:.0f} min)", flush=True)
    os.rename(DATA + ".tmp", DATA)
    print(f"corpus built: {os.path.getsize(DATA)/1e9:.1f}GB "
          f"in {(time.time()-t0)/60:.0f} min", flush=True)


build_corpus()
data = np.memmap(DATA, dtype=np.uint8, mode="r")
n_val = len(data) // 100
train_data, val_data = data[:-n_val], data[-n_val:]
print(f"corpus: {len(data)/1e9:.1f}GB ({n_val/1e6:.0f}MB held out)", flush=True)


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK - 1, (MICRO_BATCH,))
    x = torch.stack([torch.from_numpy(d[i:i + BLOCK].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i + 1:i + 1 + BLOCK].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def lr_at(step):
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    p = (step - WARMUP) / max(MAX_STEPS - WARMUP, 1)
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * p))


model = bdh.BDH(CONFIG).to(device)
print(f"model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params", flush=True)
model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR,
                              weight_decay=WEIGHT_DECAY)

start = 0
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    start = ck["step"] + 1
    print(f"RESUMED from step {ck['step']}", flush=True)

# ---- off-site checkpoint backup (background, never blocks training) ----
HF_REPO = os.environ.get("HF_CKPT_REPO")
_busy = threading.Event()


def _upload(step):
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(HF_REPO, private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=CKPT, path_in_repo="ckpt_h100.pt",
                        repo_id=HF_REPO, commit_message=f"step {step}")
        print(f"        [ckpt @ step {step} -> {HF_REPO}]", flush=True)
    except Exception as e:
        print(f"        [upload failed: {e}]", flush=True)
    finally:
        _busy.clear()


def save_and_ship(step):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "config": CONFIG.__dict__}, CKPT)
    if HF_REPO and not _busy.is_set():
        _busy.set()
        threading.Thread(target=_upload, args=(step,), daemon=True).start()


@torch.no_grad()
def val_loss():
    model.eval()
    losses = []
    for _ in range(20):
        x, y = get_batch("val")
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


t0, last_log = time.time(), time.time()
for step in range(start, MAX_STEPS):
    lr = lr_at(step)
    for g in optimizer.param_groups:
        g["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    for _ in range(ACCUM):
        x, y = get_batch("train")
        with ctx:
            _, loss = model(x, y)
        (loss / ACCUM).backward()
    optimizer.step()

    if step % LOG_FREQ == 0:
        rate = LOG_FREQ / max(time.time() - last_log, 1e-9)
        last_log = time.time()
        eta_h = (MAX_STEPS - step) / max(rate, 1e-9) / 3600
        print(f"step {step:6d}/{MAX_STEPS}  loss {loss.item():.4f}  lr {lr:.2e}  "
              f"{rate:.2f} it/s  eta {eta_h:.1f}h", flush=True)
    if step > start and step % CKPT_FREQ == 0:
        print(f"step {step:6d}  VAL {val_loss():.4f}", flush=True)
        save_and_ship(step)

save_and_ship(MAX_STEPS - 1)
while _busy.is_set():
    time.sleep(5)
_upload(MAX_STEPS - 1)     # final upload, synchronous
print(f"FINAL val {val_loss():.4f}, {(time.time()-t0)/3600:.2f}h", flush=True)
