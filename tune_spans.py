# Instruction-tune a BDH base checkpoint on span-masked examples.
#   examples pkl: [(utf8_bytes, [(start, end), ...]), ...]; loss only on the spans
#   (one span per assistant reply, end marker included; user turns never trained).
# Usage (env): BASE=ckpt_h100.pt OUT=ckpt_h100_chat_v2.pt EXAMPLES=curriculum_examples.pkl
#              ITERS=2000 BATCH=32 MICRO=8 LR=1e-4 SMOKE=1 (20 iters)  HF_CKPT_REPO=... (optional upload)
#              TOKENIZER=bpe8k.json -> BPE arm: byte spans are converted to token spans via offsets
import os
import pickle
import random
import time

import bdh
import torch

device = torch.device("cuda")
ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
torch.manual_seed(7)
HERE = os.path.dirname(os.path.abspath(__file__))
E = os.environ.get
BASE = os.path.join(HERE, E("BASE", "ckpt_h100.pt"))
OUT = os.path.join(HERE, E("OUT", "ckpt_h100_chat_v2.pt"))
EXAMPLES = os.path.join(HERE, E("EXAMPLES", "curriculum_examples.pkl"))
SMOKE = E("SMOKE", "0") == "1"
ITERS = 20 if SMOKE else int(E("ITERS", "2000"))
BATCH, MICRO, LR = int(E("BATCH", "32")), int(E("MICRO", "8")), float(E("LR", "1e-4"))
assert BATCH % MICRO == 0
ACCUM = BATCH // MICRO

ck = torch.load(BASE, map_location="cpu", weights_only=False)
cfg = bdh.BDHConfig(**ck["config"])
model = bdh.BDH(cfg)
model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
model = model.to(device)
print(f"loaded {os.path.basename(BASE)} (step {ck.get('step', '?')}) | {ITERS} iters x batch {BATCH} (micro {MICRO})", flush=True)

with open(EXAMPLES, "rb") as f:
    examples = pickle.load(f)
# older format (bytes, mask_from) -> single span
examples = [(b, m if isinstance(m, list) else [(m, len(b))]) for b, m in examples]
TOKENIZER = E("TOKENIZER")
if TOKENIZER:                                  # BPE arm: atoms are token ids, spans in token space
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(HERE, TOKENIZER))
    conv = []
    for b, spans in examples:
        text = b.decode("utf-8", errors="ignore")
        enc = tok.encode(text)
        cspans = [(len(b[:s].decode("utf-8", errors="ignore")), len(b[:e].decode("utf-8", errors="ignore"))) for s, e in spans]
        tspans = []
        for cs, ce in cspans:
            # the reply's leading space belongs to the answer token: start at cs-1
            ts = next((i for i, (o0, o1) in enumerate(enc.offsets) if o1 > max(cs - 1, 0)), len(enc.ids))
            te = next((i for i, (o0, o1) in enumerate(enc.offsets) if o0 >= ce), len(enc.ids))
            if te > ts: tspans.append((ts, te))
        if tspans: conv.append((enc.ids, tspans))
    examples = conv
    print(f"BPE arm: {len(examples)} examples converted to token spans", flush=True)
rng = random.Random(0)
rng.shuffle(examples)
val_ex, train_ex = examples[:500], examples[500:]
print(f"{len(train_ex)} train / {len(val_ex)} val examples", flush=True)


def get_batch(pool, n):
    batch = [pool[rng.randrange(len(pool))] for _ in range(n)]
    L = max(len(b) for b, _ in batch) - 1
    x = torch.zeros(n, L, dtype=torch.long)
    y = torch.full((n, L), -100, dtype=torch.long)
    for r, (b, spans) in enumerate(batch):
        seq = torch.tensor(list(b), dtype=torch.long)   # bytes or token ids
        x[r, : len(b) - 1] = seq[:-1]
        for s, e in spans:                       # predict bytes s..e-1 -> targets at positions s-1..e-2
            y[r, max(s - 1, 0): e - 1] = seq[max(s, 1): e]
    return x.to(device), y.to(device)


opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)


@torch.no_grad()
def val_loss():
    model.eval(); tot = 0.0
    for _ in range(8):
        x, y = get_batch(val_ex, MICRO)
        with ctx:
            _, loss = model(x, y)
        tot += loss.item()
    model.train(); return tot / 8


print(f"val before: {val_loss():.4f}", flush=True)
t0 = time.time()
for it in range(ITERS):
    opt.zero_grad(set_to_none=True)
    tot = 0.0
    for _ in range(ACCUM):
        x, y = get_batch(train_ex, MICRO)
        with ctx:
            _, loss = model(x, y)
        (loss / ACCUM).backward(); tot += loss.item() / ACCUM
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if it % 100 == 0 or SMOKE:
        print(f"tune {it:4d}  loss {tot:.4f}  {(time.time()-t0)/(it+1):.2f} s/it", flush=True)

torch.save({"model": model.state_dict(), "config": cfg.__dict__, "step": ITERS,
            "examples": os.path.basename(EXAMPLES)}, OUT)
print(f"val after: {val_loss():.4f}  ({(time.time()-t0)/60:.1f} min) -> {OUT}", flush=True)
repo = os.environ.get("HF_CKPT_REPO")
if repo and not SMOKE:
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=OUT, path_in_repo=os.path.basename(OUT), repo_id=repo, commit_message="chat-tuned")
    print("uploaded", flush=True)
