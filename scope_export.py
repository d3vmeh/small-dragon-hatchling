# Neuron Scope v2 exporter -- runs text through a BDH checkpoint, records which
# neurons fire (chords = relu(board @ encoder), per pulse and room), what text
# makes each one fire hardest, and the learned circuit (decoder @ encoder) for
# one room.  Writes scope_<TAG>.json for scope.html.
#
#   CKPT=checkpoints/bdh-100m-bpe8k.pt TAG=bpe100m python -u scope_export.py
# env: CKPT TAG TEXT CHARS(200000) TOPN(300) TOPCTX(8) ROOM(0) OFFSET(1e9) BLOCK(512)
import base64
import json
import os
import struct
import sys
import time
import zlib

import numpy as np
import torch

import bdh

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("CKPT", os.path.join(HERE, "checkpoints", "bdh-100m-bpe8k.pt"))
TAG = os.environ.get("TAG", "bpe100m")
CHARS = int(os.environ.get("CHARS", 200_000))
TOPN = int(os.environ.get("TOPN", 300))
TOPCTX = int(os.environ.get("TOPCTX", 8))
ROOM = int(os.environ.get("ROOM", 0))
BLOCK = int(os.environ.get("BLOCK", 512))
WARM = 16                                # atoms at a block start excluded from context picks
CAND = 4 * TOPCTX                        # running candidates per neuron (deduped by atom later)
_corpus = os.path.join(HERE, "corpus_mix.bin")
TEXT = os.environ.get("TEXT", _corpus if os.path.exists(_corpus) else os.path.join(HERE, "eval_text_100m.bin"))
# deep offset only makes sense inside the 6 GB corpus; any other file starts at 0
OFFSET = int(os.environ.get("OFFSET", 1_000_000_000 if TEXT == _corpus else 0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- model + text
def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    cfg = bdh.BDHConfig(**ck["config"])
    m = bdh.BDH(cfg)
    m.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
    return m.to(device, torch.float32).eval(), cfg


def load_text(path, chars, offset):
    offset = max(0, min(offset, os.path.getsize(path) - chars * 4))   # clamp to the file
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read(chars * 4)          # over-read; utf-8 chars can be multi-byte
    return raw.decode("utf-8", errors="ignore")[:chars]


def make_alphabet(cfg):
    """encode(text)->ids, span(ids)->str, atom(id)->str; PRE/POST are atom counts."""
    if cfg.vocab_size == 256:
        enc = lambda s: list(s.encode("utf-8"))
        span = lambda ids: bytes(ids).decode("utf-8", errors="ignore")
        atom = lambda i: chr(i) if i < 0x80 else f"\\x{i:02x}"   # lone utf-8 bytes shown as hex
        return "bytes", enc, span, atom, 40, 20
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(LAB, "bpe8k.json"))
    return ("bpe8k", lambda s: tok.encode(s).ids, lambda ids: tok.decode(ids),
            lambda i: tok.decode([i]), 16, 8)


# ------------------------------------------------------------ firing statistics
@torch.no_grad()
def run_text(m, cfg, ids):
    """Blocks of BLOCK atoms, each a fresh context (recompute mode).  Returns
    per-pulse count/sum of positive chords (nh, N) and, per neuron, the TOPCTX
    (activation, global position) pairs where it fired hardest."""
    nh, D, P = cfg.n_head, cfg.n_embd, cfg.n_layer
    N = cfg.mlp_internal_dim_multiplier * D // nh
    count = torch.zeros(P, nh, N, device=device)
    ssum = torch.zeros(P, nh, N, device=device)
    top_v = torch.full((P, nh, N, CAND), -1.0, device=device)
    top_p = torch.zeros(P, nh, N, CAND, dtype=torch.long, device=device)
    nblk = (len(ids) + BLOCK - 1) // BLOCK
    for b in range(nblk):
        s = b * BLOCK
        blk = ids[s:s + BLOCK]
        T = len(blk)
        pos = torch.arange(s, s + T, device=device)
        warm = (pos - s < WARM).float().view(1, T, 1)      # first atoms of a block have no context
        x = m.ln(m.embed(torch.tensor([blk], device=device)).unsqueeze(1))  # board
        for p in range(P):
            x_sparse = torch.relu(x @ m.encoder)           # chords (1, nh, T, N)
            c = x_sparse[0]
            count[p] += (c > 0).sum(1)
            ssum[p] += c.sum(1)
            cw = c - warm * 1e9                             # warm-up atoms never win a context slot
            for r in range(nh):                             # running top-K per neuron
                cand_v = torch.cat([top_v[p, r], cw[r].T], 1)             # (N, K+T)
                cand_p = torch.cat([top_p[p, r], pos.expand(N, T)], 1)
                v, i = cand_v.topk(CAND, dim=1)
                top_v[p, r], top_p[p, r] = v, cand_p.gather(1, i)
            yKV = m.ln(m.attn(Q=x_sparse, K=x_sparse, V=x))
            xy = x_sparse * torch.relu(yKV @ m.encoder_v)   # veto
            yMLP = xy.transpose(1, 2).reshape(1, 1, T, N * nh) @ m.decoder
            x = m.ln(x + m.ln(yMLP))
        if b % 50 == 0 or b == nblk - 1:
            log(f"block {b + 1}/{nblk}")
    return count.cpu(), ssum.cpu(), top_v.cpu(), top_p.cpu()


def snippet(ids, pos, span, atom, pre_n, post_n):
    bstart = pos - pos % BLOCK                              # model saw only its block
    pre = span(ids[max(bstart, pos - pre_n):pos])
    post = span(ids[pos + 1:pos + 1 + post_n])
    return {"pre": pre[-40:], "atom": atom(ids[pos]), "post": post[:20]}


# ------------------------------------------------------------------- circuit
def png_gray(img):
    """Minimal 8-bit grayscale PNG encoder."""
    h, w = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


@torch.no_grad()
def circuit(m, cfg, room, rows=1024, nblocks=256):
    """C = decoder[room] @ encoder[room]: C[i, j] = how much neuron i's output,
    written to the board, excites neuron j at the next pulse.  Never holds the
    full N x N; chunks over rows."""
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    enc = m.encoder[room]                                   # (D, N)
    dec = m.decoder[room * N:(room + 1) * N]                # (N, D)
    print(f"circuit room {room}: encoder {tuple(m.encoder.shape)} -> {tuple(enc.shape)}, "
          f"decoder {tuple(m.decoder.shape)} -> {tuple(dec.shape)}, C = {N}x{N}", flush=True)
    g = torch.Generator(device="cpu").manual_seed(0)        # pass 0: 99th pct of |C| (sampled rows)
    samp = torch.randperm(N, generator=g)[:128].to(device)
    thr = torch.quantile((dec[samp] @ enc).abs().flatten(), 0.99).item()
    degree = torch.zeros(N, dtype=torch.long, device=device)
    nb_idx = torch.zeros(N, 10, dtype=torch.long)
    nb_w = torch.zeros(N, 10)
    for i0 in range(0, N, rows):                            # pass 1: degree + top-10 out
        C = dec[i0:i0 + rows] @ enc
        A = C.abs()
        degree[i0:i0 + rows] = (A > thr).sum(1)
        v, j = A.topk(10, dim=1)
        nb_idx[i0:i0 + rows] = j.cpu()
        nb_w[i0:i0 + rows] = C.gather(1, j).cpu()
    perm = degree.argsort(descending=True)                  # neurons sorted by out-degree
    blk = torch.zeros(N, dtype=torch.long, device=device)
    blk[perm] = torch.arange(N, device=device) * nblocks // N
    M = torch.zeros(N, nblocks, device=device)
    M[torch.arange(N, device=device), blk] = 1.0
    M = M / M.sum(0, keepdim=True)                          # block-mean weights
    H = torch.zeros(nblocks, nblocks, device=device)
    for i0 in range(0, N, rows):                            # pass 2: block-averaged |C|
        A = (dec[i0:i0 + rows] @ enc).abs()
        H += M[i0:i0 + rows].T @ (A @ M)
    H = H.cpu().numpy()
    img = np.clip(H / max(np.percentile(H, 99.5), 1e-12), 0, 1)
    png = png_gray((img * 255).astype(np.uint8))
    deg = degree.cpu().numpy()
    counts, edges = np.histogram(deg, bins=40)
    log(f"circuit done: |C| 99th pct {thr:.4f}, degree mean {deg.mean():.1f} max {deg.max()}")
    return {"room": room, "threshold": round(thr, 5),
            "neighbors": {str(i): [[int(j), round(float(w), 4)] for j, w in
                                   zip(nb_idx[i].tolist(), nb_w[i].tolist())] for i in range(N)},
            "heatmap_png": base64.b64encode(png).decode(),
            "degree_hist": {"edges": [int(e) for e in edges], "counts": [int(c) for c in counts]}}


# ---------------------------------------------------------------------- main
def main():
    m, cfg = load_model(CKPT)
    nh, P = cfg.n_head, cfg.n_layer
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // nh
    alphabet, enc, span, atom, pre_n, post_n = make_alphabet(cfg)
    log(f"{CKPT}: D={cfg.n_embd} N={N} {P} pulses x {nh} rooms, alphabet={alphabet}, {device}")
    text = load_text(TEXT, CHARS, OFFSET)
    ids = enc(text)
    assert ids, "no atoms"
    log(f"text {TEXT}@{OFFSET}: {len(text)} chars -> {len(ids)} atoms")

    count, ssum, top_v, top_p = run_text(m, cfg, ids)
    total = len(ids)
    rate = count / total
    strength = torch.where(count > 0, ssum / count.clamp(min=1), torch.zeros_like(ssum))
    sparsity = (count.sum(2) / (total * N)).tolist()        # [pulse][room]
    for p in range(P):
        log(f"pulse {p}: sparsity per room " + " ".join(f"{s * 100:.2f}%" for s in sparsity[p]))
    # two bands per (layer pass, head): the always-on neurons by rate, and the selective
    # ones (0.002 < rate < 0.25) by strength
    n_rate, n_sel = TOPN // 3, TOPN - TOPN // 3
    neurons = []
    for p in range(P):
        for r in range(nh):
            picks = [(n, "rate") for n in rate[p, r].topk(n_rate).indices.tolist()]
            selective = (rate[p, r] > 0.002) & (rate[p, r] < 0.25)
            sel_str = torch.where(selective, strength[p, r], torch.full_like(strength[p, r], -1.0))
            picks += [(n, "selective") for n in sel_str.topk(n_sel).indices.tolist() if selective[n]]
            for n, band in picks:
                order = top_v[p, r, n].argsort(descending=True).tolist()
                order = [k for k in order if top_v[p, r, n, k] > 0]
                seen, distinct = set(), []                  # prefer different firing atoms
                for k in order:
                    a = ids[int(top_p[p, r, n, k])]
                    if a not in seen:
                        seen.add(a); distinct.append(k)
                keep = (distinct + [k for k in order if k not in distinct])[:TOPCTX]
                ctx = [dict(snippet(ids, int(top_p[p, r, n, k]), span, atom, pre_n, post_n),
                            act=round(float(top_v[p, r, n, k]), 3)) for k in keep]
                neurons.append({"pulse": p, "room": r, "id": n, "band": band,
                                "rate": round(float(rate[p, r, n]), 4),
                                "strength": round(float(strength[p, r, n]), 4), "contexts": ctx})
    log(f"{len(neurons)} neurons with contexts collected")

    out = {"meta": {"tag": TAG, "ckpt": os.path.relpath(CKPT, HERE), "alphabet": alphabet,
                    "D": cfg.n_embd, "N": N, "n_pulses": P, "n_rooms": nh, "chars": len(text),
                    "atoms": total, "block": BLOCK, "text_source": f"{os.path.relpath(TEXT, HERE)}@{OFFSET}"},
           "sparsity": sparsity, "neurons": neurons, "circuit": circuit(m, cfg, ROOM)}
    path = os.path.join(HERE, f"scope_{TAG}.json")
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    log(f"wrote {path}: {os.path.getsize(path) / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
