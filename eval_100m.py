# Byte vs BPE model, bits per character on the same
# external held-out text (eval_text_100m.bin; see build_eval_text.py), scored
# separately on the FineWeb and TinyStories portions and combined.
#   BYTES_CKPT= BPE_CKPT= OUT= python -u eval_100m.py
# SMOKE=1 -> 100k chars. Env BYTES_CKPT / BPE_CKPT override paths.
import math
import os

from load_any import load_model
import torch
from tokenizers import Tokenizer

device = torch.device("cuda")
ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = bool(os.environ.get("SMOKE"))
BLOCK = 512
BYTES_CKPT = os.environ.get("BYTES_CKPT", os.path.join(HERE, "checkpoints", "bdh-100m-bytes.pt"))
BPE_CKPT = os.environ.get("BPE_CKPT", os.path.join(HERE, "checkpoints", "bdh-100m-bpe8k.pt"))
raw = open(os.path.join(HERE, "eval_text_100m.bin"), "rb").read()
FW = 2_500_000
parts = {"fineweb": raw[:FW].decode("utf-8", errors="ignore"),
         "tinystories": raw[FW:].decode("utf-8", errors="ignore")}
if SMOKE:
    parts = {k: v[:60_000] for k, v in parts.items()}


def load(path):
    m, ck = load_model(path)          # BDH or GPT by ck["arch"] (load_any.py)
    return m.to(device).eval(), ck


@torch.no_grad()
def bits_over(model, ids, n_chars):
    """total bits over full blocks; returns (bits, chars covered)."""
    bits = n = 0.0
    for i in range(0, len(ids) - BLOCK - 1, BLOCK):
        x = torch.tensor([ids[i:i + BLOCK]], dtype=torch.long, device=device)
        y = torch.tensor([ids[i + 1:i + 1 + BLOCK]], dtype=torch.long, device=device)
        with ctx:
            _, loss = model(x, y)
        bits += loss.item() * BLOCK / math.log(2); n += BLOCK
    return bits, n * (n_chars / len(ids))


tok = Tokenizer.from_file(os.path.join(HERE, "bpe8k.json"))
res = {}
for arm, path, enc in (("bytes", BYTES_CKPT, lambda s: list(s.encode("utf-8"))),
                       ("bpe", BPE_CKPT, lambda s: tok.encode(s).ids)):
    m, ck = load(path)
    res[arm] = {"step": ck.get("step")}
    tb = tc = 0.0
    for name, text in parts.items():
        b, c = bits_over(m, enc(text), len(text))
        res[arm][name] = b / c; tb += b; tc += c
        print(f"{arm:5s} {name:12s} {b/c:.4f} bpc", flush=True)
    res[arm]["combined"] = tb / tc
    del m; torch.cuda.empty_cache()

lines = [f"=== bits per character on external held-out text "
         f"(FineWeb-Edu shard 013 + TinyStories validation){' [SMOKE]' if SMOKE else ''} ===",
         f"# bytes: {os.path.basename(BYTES_CKPT)}   bpe: {os.path.basename(BPE_CKPT)}",
         f"{'':10s}{'bytes':>12s}{'bpe':>12s}{'delta':>10s}"]
for k in ("fineweb", "tinystories", "combined"):
    d = res["bytes"][k] - res["bpe"][k]
    lines.append(f"{k:10s}{res['bytes'][k]:>12.4f}{res['bpe'][k]:>12.4f}{d:>+10.4f}")
d = res["bytes"]["combined"] - res["bpe"]["combined"]
lines.append(f"(steps: bytes {res['bytes']['step']}, bpe {res['bpe']['step']})")
out = "\n".join(lines); print(out, flush=True)
if not SMOKE:
    open(os.path.join(HERE, os.environ.get("OUT", "results_100m.txt")), "w").write(out + "\n")
