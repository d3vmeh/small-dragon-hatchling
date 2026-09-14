# Neuron density on HELD-OUT text, and on an untrained model of the same shape.
#   CKPT= TAG= TEXT= CHARS=200000 [RANDOM=1] python -u density_check.py
# Prints per-layer and per-head density (% of the N neurons in a head with a positive activation
# at a position) and appends one line per run to density_heldout.txt.
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bdh

HERE = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = os.environ["CKPT"]; TAG = os.environ["TAG"]
TEXT = os.environ.get("TEXT", os.path.join(HERE, "eval_text_100m.bin"))
CHARS = int(os.environ.get("CHARS", 200_000)); BLOCK = 512
RANDOM = bool(os.environ.get("RANDOM"))          # RANDOM=1: same config, freshly initialised weights

ck = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
cfg = bdh.BDHConfig(**ck["config"])
m = bdh.BDH(cfg)
if not RANDOM:
    m.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
m = m.to(device, torch.float32).eval()

if cfg.vocab_size == 256:
    enc = lambda s: list(s.encode("utf-8")); alpha = "bytes"
else:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(LAB, "bpe8k.json"))
    enc = lambda s: tok.encode(s).ids; alpha = "bpe8k"

raw = open(TEXT, "rb").read(CHARS * 4)
text = raw.decode("utf-8", errors="ignore")[:CHARS]
ids = enc(text)
nh, D, P = cfg.n_head, cfg.n_embd, cfg.n_layer
N = cfg.mlp_internal_dim_multiplier * D // nh
count = torch.zeros(P, nh, device=device); total = 0
t0 = time.time()
with torch.no_grad():
    for b in range((len(ids) + BLOCK - 1) // BLOCK):
        blk = ids[b * BLOCK:(b + 1) * BLOCK]
        if not blk: break
        T = len(blk)
        x = m.ln(m.embed(torch.tensor([blk], device=device)).unsqueeze(1))   # board
        for p in range(P):
            x_sparse = torch.relu(x @ m.encoder)         # chords (1, nh, T, N)
            c = x_sparse[0]
            count[p] += (c > 0).float().sum(2).sum(1)    # positives per head
            yKV = m.ln(m.attn(Q=x_sparse, K=x_sparse, V=x))
            xy = x_sparse * torch.relu(yKV @ m.encoder_v)                    # veto
            yMLP = xy.transpose(1, 2).reshape(1, 1, T, N * nh) @ m.decoder
            x = m.ln(x + m.ln(yMLP))                     # same forward pass as scope_export.run_text
        total += T
dens = (count / (total * N) * 100).tolist()
per_layer = [sum(r) / len(r) for r in dens]
per_head = [sum(dens[p][h] for p in range(P)) / P for h in range(nh)]
line = (f"{TAG}\t{'RANDOM' if RANDOM else 'trained'}\t{alpha}\t{os.path.basename(TEXT)}\t{total} atoms\t"
        f"mean {sum(per_layer)/len(per_layer):.2f}%\tlayers " + " ".join(f"{v:.1f}" for v in per_layer) +
        "\theads " + " ".join(f"{v:.1f}" for v in per_head) + f"\t{time.time()-t0:.0f}s")
print(line, flush=True)
open(os.path.join(HERE, "density_heldout.txt"), "a").write(line + "\n")
