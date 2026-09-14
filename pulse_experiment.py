# Does a BDH do better with more layer passes at inference? (No training involved.)
# The 6 "layers" share weights, so the SAME trained model can be run with any
# pulse count at inference. Measures val loss + generates a sample at each.
#   CKPT= TAG= DATA= [TOK=1] [PULSES=5,7] FULL=1 python -u pulse_experiment.py
import os

import bdh
import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16) \
    if device.type == "cuda" else __import__("contextlib").nullcontext()
torch.manual_seed(1337)
HERE = os.path.dirname(os.path.abspath(__file__))

CKPT = os.environ.get("CKPT", os.path.join(HERE, "checkpoints", "bdh-100m-bytes.pt"))
TAG = os.environ.get("TAG", "bytes_100m")
DATA = os.environ.get("DATA", os.path.join(HERE, "eval_text_100m.bin"))   # evaluation text (raw bytes)
ck = torch.load(CKPT, map_location="cpu",
                weights_only=False)
state = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}

if os.environ.get("TOK"):                     # TOK=1: BPE checkpoint; tokenize the text file with bpe8k
    from tokenizers import Tokenizer
    _tok = Tokenizer.from_file(os.path.join(HERE, "bpe8k.json"))
    _txt = open(DATA, "rb").read().decode("utf-8", errors="ignore")
    data = np.array(_tok.encode(_txt).ids, dtype=np.int64)
else:
    data = np.memmap(DATA, dtype=np.uint8, mode="r")
FULL = bool(os.environ.get("FULL"))          # FULL=1: score the WHOLE file sequentially in 512-blocks
val = data if FULL else data[-len(data) // 20:]
BLOCK, BATCH, EVAL_BATCHES = 512, 32, 40
PULSES = [int(x) for x in os.environ.get("PULSES", "2,3,4,6,8,12,16").split(",")]   # 6 = the trained depth; PULSES= overrides the list

PROMPT = "Once upon a time, there was a little dragon named "


def get_batch():
    ix = torch.randint(len(val) - BLOCK - 1, (BATCH,))
    x = torch.stack([torch.from_numpy(val[i:i + BLOCK].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(val[i + 1:i + 1 + BLOCK].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def sample(model, n=220):
    idx = torch.tensor([_tok.encode(PROMPT).ids if os.environ.get("TOK") else list(PROMPT.encode())], dtype=torch.long, device=device)
    out = []
    for _ in range(n):
        with ctx:
            logits, _ = model(idx[:, -BLOCK:])
        logits = logits[0, -1, :] / 0.8
        v, _ = torch.topk(logits, 40)
        logits[logits < v[-1]] = float("-inf")
        b = int(torch.multinomial(F.softmax(logits.float(), -1), 1))
        out.append(b)
        idx = torch.cat([idx, torch.tensor([[b]], device=device)], dim=1)
    return _tok.decode(out) if os.environ.get("TOK") else bytes(out).decode("utf-8", errors="ignore")


results = [f"=== pulse-count experiment: {os.path.basename(CKPT)} (trained at 6); data {os.path.basename(DATA)}"
           f"{' FULL sequential' if FULL else ' last-5% random batches'}; nats/byte ==="]
torch.manual_seed(7)
if FULL:
    fixed_batches = []
    starts = list(range(0, len(val) - BLOCK - 1, BLOCK))
    for b in range(0, len(starts), BATCH):
        ss = starts[b:b + BATCH]
        x = torch.stack([torch.from_numpy(val[i:i + BLOCK].astype(np.int64)) for i in ss])
        y = torch.stack([torch.from_numpy(val[i + 1:i + 1 + BLOCK].astype(np.int64)) for i in ss])
        fixed_batches.append((x.to(device), y.to(device)))
else:
    fixed_batches = [get_batch() for _ in range(EVAL_BATCHES)]

for p in PULSES:
    cfg = bdh.BDHConfig(**{**ck["config"], "n_layer": p})
    model = bdh.BDH(cfg)
    model.load_state_dict(state)         # same weights, different pulse count
    model = model.to(device).eval()
    with torch.no_grad():
        losses = []
        for x, y in fixed_batches:
            with ctx:
                _, loss = model(x, y)
            losses.append(loss.item())
    vl = sum(losses) / len(losses)
    line = f"pulses={p:2d}  val_loss {vl:.4f} nats/byte"
    print(line, flush=True)
    results.append(line)
    story = sample(model)
    results.append(f"    sample: {story[:200]!r}")
    print(f"    sample: {story[:120]!r}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

with open(os.path.join(HERE, f"pulse_results_{TAG}.txt"), "w") as f:
    f.write("\n".join(results) + "\n")
print("DONE -> pulse_results.txt", flush=True)
