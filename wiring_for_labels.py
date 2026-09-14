# Top circuit neighbours for the neurons that carry labels, in whatever head they live in.
# C = decoder[room] @ encoder[room]; C[i,j] = how much neuron i exciting the board pushes neuron j
# at the next pulse (same definition as scope_export.circuit). Runs on CPU.
import json, os, torch, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bdh
HERE = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get("TAG", "bpe100m")
CKPT = os.environ.get("CKPT", os.path.join(HERE, "checkpoints", "bdh-100m-bpe8k.pt"))
K = int(os.environ.get("K", "8"))

ck = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
cfg = bdh.BDHConfig(**ck["config"]); m = bdh.BDH(cfg)
m.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
m = m.to(torch.float32).eval()
N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head

want = {}
for f in (f"scope_labels_{TAG}.json", f"scope_labels_{TAG}_v2.json"):
    p = os.path.join(HERE, f)
    if os.path.exists(p):
        for key in json.load(open(p)):
            _, pulse, room, nid = key.split(":")
            want.setdefault(int(room), set()).add(int(nid))
print({r: len(v) for r, v in want.items()}, "neurons per head", flush=True)

out = {}
with torch.no_grad():
    for room, ids in want.items():
        enc = m.encoder[room]                       # (D, N)
        dec = m.decoder[room * N:(room + 1) * N]    # (N, D)
        g = torch.Generator().manual_seed(0)
        samp = torch.randperm(N, generator=g)[:128]
        thr = torch.quantile((dec[samp] @ enc).abs().flatten(), 0.99).item()
        ids = sorted(ids)
        C = dec[ids] @ enc                          # (len(ids), N)
        v, j = C.abs().topk(K + 1, dim=1)
        for r, nid in enumerate(ids):
            pairs = [(int(j[r, c]), round(float(C[r, j[r, c]]), 4)) for c in range(K + 1)
                     if int(j[r, c]) != nid][:K]
            out[f"{room}:{nid}"] = {"thr": round(thr, 4), "top": pairs}
        print(f"head {room}: {len(ids)} neurons, 99th pct |C| = {thr:.4f}", flush=True)
json.dump(out, open(os.path.join(HERE, f"wiring_{TAG}.json"), "w"))
print("wrote", f"wiring_{TAG}.json", len(out), "neurons")
