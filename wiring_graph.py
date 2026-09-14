# Pairwise circuit weights AMONG the labelled neurons, per head.
# C[i,j] = decoder[room][i] . encoder[room][:,j] : neuron i firing pushes neuron j at the next pulse.
# Heads are separate circuits in this definition, so the graph has one component per head.
import json, os, torch, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bdh
HERE = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get("TAG", "bpe100m")
CKPT = os.environ.get("CKPT", os.path.join(HERE, "checkpoints", "bdh-100m-bpe8k.pt"))
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

edges, thr_by_head = [], {}
with torch.no_grad():
    for room, idset in sorted(want.items()):
        ids = sorted(idset)
        enc = m.encoder[room]; dec = m.decoder[room * N:(room + 1) * N]
        g = torch.Generator().manual_seed(0)
        samp = torch.randperm(N, generator=g)[:128]
        thr = torch.quantile((dec[samp] @ enc).abs().flatten(), 0.99).item()
        thr_by_head[room] = round(thr, 4)
        C = dec[ids] @ enc[:, ids]                       # (n, n) among labelled neurons only
        n = len(ids)
        for a in range(n):
            for b in range(n):
                if a == b: continue
                w = float(C[a, b])
                if abs(w) >= thr * 0.5:                  # half the 99th-percentile threshold
                    edges.append({"h": room, "s": ids[a], "t": ids[b], "w": round(w, 4)})
        print(f"head {room}: {n} labelled neurons, 99th pct |C| = {thr:.4f}, "
              f"{sum(1 for e in edges if e['h']==room)} edges above half of it", flush=True)
json.dump({"thr": thr_by_head, "edges": edges}, open(os.path.join(HERE, f"wiring_graph_{TAG}.json"), "w"))
print("wrote", f"wiring_graph_{TAG}.json", len(edges), "edges")
