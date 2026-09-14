# Does the sliding window hold quality far past the block? Stream 4096 positions of the
# external held-out text with window=None and window=480 on the 100M byte and BPE bases;
# report bits per character per 512-position chunk. Reference per chunk: the plain model
# scoring that chunk as a fresh block (context restarts at the chunk start).
#   START=<byte offset> python -u test_window_long.py
import math, os, time
import bdh, torch
from streaming import StreamingBDH

HERE = os.path.dirname(os.path.abspath(__file__))
NPOS = int(os.environ.get("NPOS", 4096)); CH = 512; WIN = int(os.environ.get("WINDOW", 480))
CKPTS = [(os.path.join(HERE, "checkpoints", "bdh-100m-bytes.pt"), "bytes"),
         (os.path.join(HERE, "checkpoints", "bdh-100m-bpe8k.pt"), "bpe8k")]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
START = int(os.environ.get("START", 0))  # byte offset into the held-out file
with open(os.path.join(HERE, "eval_text_100m.bin"), "rb") as fh:
    fh.seek(START); text = fh.read(40000).decode("utf-8", errors="ignore")
lines = [f"# window quality over {NPOS} positions of external held-out text (byte offset START={START}); bpc per {CH}-position chunk",
         f"# rows: fresh-block reference (plain model, context restarts each chunk) / stream no-window / stream window={WIN}"]
print("\n".join(lines), flush=True)



for path, alpha in CKPTS:
    if alpha == "bytes":
        ids = list(text.encode("utf-8")); chars_per_id = [1.0] * len(ids)
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(HERE, "bpe8k.json"))
        enc = tok.encode(text); ids = enc.ids
        chars_per_id = [b - a for (a, b) in enc.offsets]
    ids = ids[:NPOS + 1]; chars_per_id = chars_per_id[:NPOS + 1]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ref = bdh.BDH(bdh.BDHConfig(**ck["config"]))
    ref.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
    ref = ref.to(device, torch.float32).eval()
    # fresh-block reference per chunk
    ref_bpc = []
    with torch.no_grad():
        for c in range(NPOS // CH):
            x = torch.tensor([ids[c * CH:(c + 1) * CH]], dtype=torch.long, device=device)
            lg, _ = ref(x)
            y = torch.tensor(ids[c * CH + 1:(c + 1) * CH + 1], device=device)
            nll = -torch.log_softmax(lg[0].float(), -1).gather(-1, y[:, None]).squeeze(-1)
            ref_bpc.append(nll.sum().item() / math.log(2) / sum(chars_per_id[c * CH + 1:(c + 1) * CH + 1]))
    # truncated recompute: each position scored by the plain model given ONLY the last WIN positions
    if os.environ.get("TRUNC"):
        tb = [0.0] * (NPOS // CH); tc = [0.0] * (NPOS // CH); t0 = time.time()
        with torch.no_grad():
            for p in range(NPOS):
                lo = max(0, p + 1 - WIN)
                x = torch.tensor([ids[lo:p + 1]], dtype=torch.long, device=device)
                lg, _ = ref(x)
                nll = -torch.log_softmax(lg[0, -1].float(), -1)[ids[p + 1]].item()
                tb[p // CH] += nll / math.log(2); tc[p // CH] += chars_per_id[p + 1]
        trunc_bpc = [b / c for b, c in zip(tb, tc)]
        print(f"  ({alpha} truncated recompute: {(time.time()-t0)/NPOS*1000:.0f} ms/pos)", flush=True)
    del ref
    rows = {"fresh-block": ref_bpc}
    if os.environ.get("TRUNC"): rows[f"recompute last {WIN}"] = trunc_bpc
    for win in (None, WIN):
        sm = StreamingBDH(ckpt_path=path, device=device.type, dtype=torch.float32, window=win)
        bits = [0.0] * (NPOS // CH); chars = [0.0] * (NPOS // CH)
        t0 = time.time()
        for p in range(NPOS):
            lg = sm.step(ids[p])
            nll = -torch.log_softmax(lg.float(), -1)[ids[p + 1]].item()
            bits[p // CH] += nll / math.log(2); chars[p // CH] += chars_per_id[p + 1]
        rows[f"stream w={win}"] = [b / c for b, c in zip(bits, chars)]
        print(f"  ({alpha} window={win}: {(time.time()-t0)/NPOS*1000:.0f} ms/pos)", flush=True)
        del sm
    hdr = f"{alpha:7s} {'chunk->':14s}" + "".join(f"{f'{c*CH}-{(c+1)*CH}':>11s}" for c in range(NPOS // CH))
    lines.append(hdr); print(hdr, flush=True)
    for k, v in rows.items():
        l = f"{'':7s} {k:14s}" + "".join(f"{x:11.3f}" for x in v)
        lines.append(l); print(l, flush=True)
open(os.path.join(HERE, os.environ.get("OUT", "test_window_long.txt")), "w").write("\n".join(lines) + "\n")
print("DONE", flush=True)
