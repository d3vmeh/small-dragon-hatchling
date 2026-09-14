# Streaming-vs-recompute equivalence over a FULL block (512 positions) on
# several checkpoints, both alphabets. Text: the external held-out eval text (never trained on).
#   python -u test_streaming_long.py
# Env: N=512 positions, GREEDY=40 continuation length, CKPTS=comma list of paths.
import os, sys, time
import bdh, torch
from streaming import StreamingBDH

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(os.environ.get("N", 512)); G = int(os.environ.get("GREEDY", 40))
CKPTS = os.environ.get("CKPTS", ",".join([
    os.path.join(HERE, "checkpoints", "bdh-100m-bytes.pt"),
    os.path.join(HERE, "checkpoints", "bdh-100m-bpe8k.pt")])).split(",")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
raw = open(os.path.join(HERE, "eval_text_100m.bin"), "rb").read(20000)
text = raw.decode("utf-8", errors="ignore")
lines = []
for path in CKPTS:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = bdh.BDHConfig(**ck["config"])
    if cfg.vocab_size == 256:
        ids = list(text.encode("utf-8"))[:N]; alpha = "bytes"
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(HERE, "bpe8k.json"))
        ids = tok.encode(text).ids[:N]; alpha = "bpe8k"
    ref = bdh.BDH(cfg)
    ref.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
    ref = ref.to(device, torch.float32).eval()
    with torch.no_grad():
        ref_logits, _ = ref(torch.tensor([ids], dtype=torch.long, device=device))
    ref_logits = ref_logits[0].float()
    sm = StreamingBDH(ckpt_path=path, device=device.type, dtype=torch.float32)
    t0 = time.time()
    stream_logits = torch.stack([sm.step(i) for i in ids])
    dt = (time.time() - t0) / len(ids)
    diff = (ref_logits - stream_logits).abs()
    # greedy continuation from the same state
    out_s, lg = [], stream_logits[-1]
    for _ in range(G):
        b = int(lg.argmax()); out_s.append(b); lg = sm.step(b)
    ctx, out_r = list(ids), []
    with torch.no_grad():
        for _ in range(G):
            l, _ = ref(torch.tensor([ctx], dtype=torch.long, device=device))
            b = int(l[0, -1].argmax()); out_r.append(b); ctx.append(b)
    line = (f"{os.path.basename(path):22s} {alpha:6s} D={cfg.n_embd:4d} positions={len(ids):4d} "
            f"max|dlogit|={diff.max().item():.2e} mean={diff.mean().item():.2e} "
            f"greedy{G}={'IDENTICAL' if out_s == out_r else 'DIFFER'} stream_ms/pos={dt*1000:.1f}")
    print(line, flush=True); lines.append(line)
    del ref, sm; torch.cuda.empty_cache() if device.type == "cuda" else None
open(os.path.join(HERE, "test_streaming_long.txt"), "w").write("\n".join(lines) + "\n")
