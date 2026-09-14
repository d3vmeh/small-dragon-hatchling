# Context-use curve: bits per character on the external held-out text when each
# position may see only a bounded context. Tier k scores positions whose context
# length lies in [k, 2k): blocks of length 2k, loss counted on the last k positions.
# Used to test whether the alphabet gap comes from context span.
#   MODELS=bytes_100m,bpe_100m python -u context_use.py
# Env: SMOKE=1 (200k chars), CHARS=N (chars per genre, default all), MODELS=comma list.
import math, os, sys
import torch
from load_any import load_model
from tokenizers import Tokenizer

device = torch.device("cuda")
ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = bool(os.environ.get("SMOKE"))
TIERS = [8, 16, 32, 64, 128, 256]          # context in [k, 2k); 2k <= 512 block
# name -> (checkpoint under checkpoints/, alphabet); entries whose file is absent are skipped
CK = os.path.join(HERE, "checkpoints")
MODELS = {
    "bytes_25m": (os.path.join(CK, "bdh-25m-bytes.pt"), "bytes"),
    "bytes_50m": (os.path.join(CK, "bdh-50m-bytes.pt"), "bytes"),
    "bytes_100m": (os.path.join(CK, "bdh-100m-bytes.pt"), "bytes"),
    "bpe_25m": (os.path.join(CK, "bdh-25m-bpe8k.pt"), "bpe"),
    "bpe_50m": (os.path.join(CK, "bdh-50m-bpe8k.pt"), "bpe"),
    "bpe_100m": (os.path.join(CK, "bdh-100m-bpe8k.pt"), "bpe"),
    "gpt_25m": (os.path.join(CK, "gpt-25m-bytes.pt"), "bytes"),   # transformer baselines
    "gpt_50m": (os.path.join(CK, "gpt-50m-bytes.pt"), "bytes"),
    "gpt_100m": (os.path.join(CK, "gpt-100m-bytes.pt"), "bytes"),
    "gpt_bpe_25m": (os.path.join(CK, "gpt-25m-bpe8k.pt"), "bpe"),
    "gpt_bpe_50m": (os.path.join(CK, "gpt-50m-bpe8k.pt"), "bpe"),
    "gpt_bpe_100m": (os.path.join(CK, "gpt-100m-bpe8k.pt"), "bpe"),
}
MODELS = {k: v for k, v in MODELS.items() if os.path.exists(v[0])}
if os.environ.get("MODELS"):
    MODELS = {k: MODELS[k] for k in os.environ["MODELS"].split(",")}
raw = open(os.path.join(HERE, "eval_text_100m.bin"), "rb").read()
FW = 2_500_000
parts = {"fineweb": raw[:FW].decode("utf-8", errors="ignore"),
         "tinystories": raw[FW:].decode("utf-8", errors="ignore")}
lim = 200_000 if SMOKE else int(os.environ.get("CHARS", "0")) or None
if lim:
    parts = {k: v[:lim] for k, v in parts.items()}
tok = Tokenizer.from_file(os.path.join(HERE, "bpe8k.json"))
enc = {"bytes": lambda s: list(s.encode("utf-8")), "bpe": lambda s: tok.encode(s).ids}


def load(path):
    m, ck = load_model(path)          # BDH or GPT by ck["arch"] (load_any.py)
    return m.to(device).eval()


@torch.no_grad()
def tier_bits(model, ids, k, batch=16):
    """sum of bits over scored positions (context in [k,2k)) and count of scored positions."""
    L = 2 * k
    starts = list(range(0, len(ids) - L - 1, k))       # stride k so every position scored once
    bits = n = 0.0
    for b in range(0, len(starts), batch):
        ss = starts[b:b + batch]
        x = torch.tensor([ids[s:s + L] for s in ss], dtype=torch.long, device=device)
        y = torch.tensor([ids[s + 1:s + 1 + L] for s in ss], dtype=torch.long, device=device)
        with ctx:
            logits, _ = model(x)
        lp = torch.log_softmax(logits[:, k:].float(), -1)
        nll = -lp.gather(-1, y[:, k:, None]).squeeze(-1)  # (B, k)
        bits += nll.sum().item() / math.log(2); n += nll.numel()
    return bits, n


out = [f"# context-use curve: bpc on external held-out text, positions scored with context in [k,2k){' [SMOKE]' if SMOKE else ''}",
       f"# chars per genre: {[len(v) for v in parts.values()]}",
       "model       " + "".join(f"{f'[{k},{2*k})':>12s}" for k in TIERS)]
print("\n".join(out), flush=True)
for name, (path, arm) in MODELS.items():
    m = load(path)
    ids = {g: enc[arm](t) for g, t in parts.items()}
    row = []
    for k in TIERS:
        tb = tc = 0.0
        for g, t in parts.items():
            b, n = tier_bits(m, ids[g], k)
            tb += b; tc += n * (len(t) / len(ids[g]))    # positions -> characters
        row.append(tb / tc)
    line = f"{name:12s}" + "".join(f"{v:12.4f}" for v in row)
    out.append(line); print(line, flush=True)
    del m; torch.cuda.empty_cache()
if not SMOKE:
    open(os.path.join(HERE, os.environ.get("OUT", "results_context_use.txt")), "w").write("\n".join(out) + "\n")
