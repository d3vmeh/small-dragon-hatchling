# Planted-fact probe for the transformer baseline (gpt.py): same words,
# filler, distances, scoring and output format as needle.py, but scored by full
# recompute over the model's 512-atom block instead of the streaming grids
# (a transformer has no separate state; its memory is the context window).
#   CKPT=checkpoints/gpt-25m-bytes.pt TAG=gpt25m NEEDLES=a,b,.. SEED=n python -u needle_gpt.py
# Each target atom is predicted from the last min(len, 512) atoms before it, so
# at distances >= ~480 the needle sentence has fallen OUT of the window: the
# transformer's "window 512" row is the analogue of BDH's window-480 row,
# and there is no "inf" row (nothing outlives the block).
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from load_any import load_model  # noqa: E402

NEEDLES = os.environ.get("NEEDLES", "banana,purple,rocket").split(",")   # env NEEDLES=a,b,c
SEED = int(os.environ.get("SEED", "0"))                                    # picks a different filler slice
DISTS = [0, 64, 128, 256, 384, 600, 1024]
CKPT = os.environ["CKPT"]
TAG = os.environ.get("TAG", "gpt25m")
OUT = os.path.join(HERE, f"needle_results_{TAG}.txt" if SEED == 0 and len(NEEDLES) == 3 else f"needle_results_{TAG}_n{len(NEEDLES)}_s{SEED}.txt")

_off = 5000 + SEED * 40000
raw = open(os.path.join(HERE, "data", "input.txt"), "rb").read()[_off:_off + 15000].decode(errors="ignore")
filler = " ".join(raw.split())
assert all(n not in filler for n in NEEDLES), "filler contains a needle"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, ck = load_model(CKPT)
model = model.to(device, torch.float32).eval()
cfg = model.config
BLOCK = getattr(cfg, "block_size", 512)
WINDOWS = [BLOCK]
if cfg.vocab_size == 256:
    encode = lambda s: list(s.encode("utf-8")); ALPHABET = "bytes"
else:
    from tokenizers import Tokenizer
    _tok = Tokenizer.from_file(os.path.join(HERE, "bpe8k.json"))
    encode = lambda s: _tok.encode(s).ids; ALPHABET = "bpe8k"


@torch.no_grad()
def score(prefix_text: str, target_text: str):
    """-log p (nats) of target given prefix, and whether greedy decode hits it.
    Mirrors needle.py's score(): each target atom is predicted from everything
    before it, here truncated to the last BLOCK atoms (full recompute per atom)."""
    if ALPHABET == "bytes":
        prefix, target = encode(prefix_text), encode(target_text)
    else:
        enc = _tok.encode(prefix_text + target_text)
        cut = next(i for i, (o0, o1) in enumerate(enc.offsets) if o0 >= len(prefix_text))
        prefix, target = enc.ids[:cut], enc.ids[cut:]
    seq = list(prefix)
    nats, greedy = 0.0, []
    for t in target:
        x = torch.tensor([seq[-BLOCK:]], dtype=torch.long, device=device)
        lg = model(x)[0][0, -1].float()
        logp = torch.log_softmax(lg, -1)
        nats -= logp[t].item()
        greedy.append(int(lg.argmax()))
        seq.append(t)
    return nats, greedy == target


lines = [f"# needle probe, {TAG} ({os.path.basename(CKPT)}, {ALPHABET}, distances in characters), fp32 {device}; nats = -log p of ' <word>.' "
         f"after 'The secret word is' ; baseline = same filler, no needle; NEEDLES={','.join(NEEDLES)} SEED={SEED}"
         f"  [transformer: full recompute, block {BLOCK}]",
         f"{'window':>8} {'dist':>5} {'needle nats':>12} {'baseline':>9} {'gain':>6} {'top1 recall':>12}"]
print("\n".join(lines), flush=True)
for w in WINDOWS:
    for d in DISTS:
        n_sum = b_sum = hits = 0
        for word in NEEDLES:
            fill = filler[:d].rstrip() + (" " if d else "")
            query = "The secret word is"
            target = f" {word}."
            nats, hit = score(f"The secret word is {word}. {fill}{query}", target)
            base, _ = score(f"{fill}{query}", target)
            n_sum += nats; b_sum += base; hits += hit
        k = len(NEEDLES)
        line = (f"{str(w):>8} {d:>5} {n_sum/k:>12.2f} {b_sum/k:>9.2f} "
                f"{(b_sum-n_sum)/k:>6.2f} {hits:>7}/{k}")
        print(line, flush=True); lines.append(line)
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"saved -> {OUT}")
