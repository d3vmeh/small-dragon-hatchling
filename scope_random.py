# Top-8 firing contexts for a UNIFORMLY RANDOM sample of neurons, drawn from all 6 x 4 x N of them,
# not from the bands scope_export.py curates.
#   TAG= CKPT= CHARS=200000 NSAMPLE=200 SEED=7 python -u scope_random.py
import json, os, random, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bdh
HERE = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get("TAG", "bpe100m")
CKPT = os.environ.get("CKPT", os.path.join(HERE, "checkpoints", "bdh-100m-bpe8k.pt"))
TEXT = os.environ.get("TEXT", os.path.join(HERE, "corpus_mix.bin"))
OFFSET = int(os.environ.get("OFFSET", 1_000_000_000))
CHARS = int(os.environ.get("CHARS", 200_000)); NS = int(os.environ.get("NSAMPLE", 200))
SEED = int(os.environ.get("SEED", 7)); BLOCK = 512; TOP = 8
device = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
cfg = bdh.BDHConfig(**ck["config"]); m = bdh.BDH(cfg)
m.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
m = m.to(device, torch.float32).eval()
nh, D, P = cfg.n_head, cfg.n_embd, cfg.n_layer
N = cfg.mlp_internal_dim_multiplier * D // nh

if cfg.vocab_size == 256:
    enc = lambda s: list(s.encode()); span = lambda ids: bytes(ids).decode("utf-8", "ignore")
    atom = lambda i: chr(i) if i < 0x80 else f"\\x{i:02x}"; alphabet = "bytes"
else:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(LAB, "bpe8k.json"))
    enc = lambda s: tok.encode(s).ids; span = lambda ids: tok.decode(ids)
    atom = lambda i: tok.decode([i]); alphabet = "bpe8k"

raw = open(TEXT, "rb").read(OFFSET + CHARS * 4)[OFFSET:]
text = raw.decode("utf-8", "ignore")[:CHARS]
ids = enc(text)
rng = random.Random(SEED)
_pool = list({(rng.randrange(P), rng.randrange(nh), rng.randrange(N)) for _ in range(NS * 3)})
rng.shuffle(_pool)                      # shuffle, then truncate
sample = sorted(_pool[:NS])
by_pr = {}
for p, r, i in sample: by_pr.setdefault((p, r), []).append(i)
print(f"{TAG}: {len(ids)} atoms, sampling {len(sample)} neurons uniformly from {P}x{nh}x{N}", flush=True)

store = {k: {"count": 0, "sum": 0.0, "top": []} for k in sample}
with torch.no_grad():
    for b in range((len(ids) + BLOCK - 1) // BLOCK):
        blk = ids[b * BLOCK:(b + 1) * BLOCK]
        if not blk: break
        T = len(blk); base = b * BLOCK
        x = m.ln(m.embed(torch.tensor([blk], device=device)).unsqueeze(1))
        for p in range(P):
            xs = torch.relu(x @ m.encoder)
            c = xs[0]
            for (pp, rr), idlist in by_pr.items():
                if pp != p: continue
                vals = c[rr][:, idlist]                      # (T, k)
                for col, nid in enumerate(idlist):
                    v = vals[:, col]
                    nz = (v > 0)
                    st = store[(pp, rr, nid)]
                    st["count"] += int(nz.sum()); st["sum"] += float(v[nz].sum()) if int(nz.sum()) else 0.0
                    k = min(TOP, T)
                    tv, ti = v.topk(k)
                    for a in range(k):
                        if float(tv[a]) <= 0: break
                        st["top"].append((float(tv[a]), base + int(ti[a])))
                    st["top"] = sorted(st["top"], reverse=True)[:TOP]
            yKV = m.ln(m.attn(Q=xs, K=xs, V=x)); xy = xs * torch.relu(yKV @ m.encoder_v)
            x = m.ln(x + m.ln((xy.transpose(1, 2).reshape(1, 1, T, N * nh) @ m.decoder)))
        if b % 50 == 0: print(f"  block {b}", flush=True)

total = len(ids)
out = []
for (p, r, i), st in store.items():
    if not st["top"]: continue
    ctx = []
    for v, pos in st["top"]:
        bstart = pos - pos % BLOCK
        ctx.append({"pre": span(ids[max(bstart, pos - 16):pos])[-46:], "atom": atom(ids[pos]),
                    "post": span(ids[pos + 1:pos + 9])[:46], "act": round(v, 3)})
    out.append({"pulse": p, "room": r, "id": i, "rate": round(st["count"] / total, 4),
                "strength": round(st["sum"] / max(st["count"], 1), 4), "contexts": ctx})
json.dump({"meta": {"tag": TAG, "alphabet": alphabet, "N": N, "chars": CHARS, "atoms": total,
                    "sample": "uniform over all pulses x rooms x N", "seed": SEED},
           "neurons": out}, open(os.path.join(HERE, f"scope_random_{TAG}.json"), "w"))
print(f"wrote scope_random_{TAG}.json: {len(out)} neurons with contexts "
      f"({len(sample) - len(out)} never fired)")
