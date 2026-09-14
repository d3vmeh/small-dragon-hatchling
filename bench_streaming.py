# Scaling benchmark: per-byte cost of streaming (synapse-grid) inference vs
# full recompute, as a function of context length. Writes bench_streaming.txt.
#   CKPT=checkpoints/bdh-100m-bytes.pt python -u bench_streaming.py [context lengths]
# Streaming cost should not depend on context length (the state has a fixed size);
# recompute grows with context (the whole prefix is re-run every byte).
import os
import sys
import time

import bdh
import torch

from streaming import StreamingBDH

HERE = os.path.dirname(os.path.abspath(__file__))
CTXS = [int(c) for c in sys.argv[1:]] or [128, 256, 512, 1024, 2048, 4096]
REPS = 5            # timed bytes per context point (after 1 warm-up)
OUT = os.path.join(HERE, "bench_streaming.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = os.environ.get("CKPT", os.path.join(HERE, "checkpoints", "bdh-100m-bytes.pt"))
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
ref = bdh.BDH(bdh.BDHConfig(**ck["config"]))
ref.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
ref = ref.to(device, torch.float32).eval()
sm = StreamingBDH(CKPT, device=device.type, dtype=torch.float32)

# deterministic filler text, long enough for the largest context
text = (b"The little dragon flew over the hills and valleys, looking for a friend. ") * 200
assert len(text) >= max(CTXS) + REPS + 1

def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()

def time_recompute(ctx_len):
    """ms per NEW byte when the whole prefix (ctx_len bytes) is re-run."""
    times = []
    with torch.no_grad():
        for i in range(REPS + 1):
            idx = torch.tensor([list(text[: ctx_len + i])], dtype=torch.long, device=device)
            sync(); t = time.perf_counter()
            ref(idx)
            sync(); times.append((time.perf_counter() - t) * 1e3)
    return sum(times[1:]) / REPS

def time_stream(ctx_len):
    """ms per new byte once ctx_len bytes are already in the state."""
    sm.reset()
    for b in text[:ctx_len]:
        sm.step(b)
    times = []
    for i in range(REPS + 1):
        sync(); t = time.perf_counter()
        sm.step(text[ctx_len + i])
        sync(); times.append((time.perf_counter() - t) * 1e3)
    return sum(times[1:]) / REPS

lines = [f"# streaming vs recompute, {os.path.basename(CKPT)}, fp32, {device.type} "
         f"({torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}), "
         f"{REPS} timed bytes/point, {time.strftime('%Y-%m-%d %H:%M')}",
         f"{'ctx':>6} {'recompute ms/byte':>18} {'stream ms/byte':>15} {'speedup':>8}"]
print("\n".join(lines), flush=True)
for c in CTXS:
    r, s = time_recompute(c), time_stream(c)
    line = f"{c:>6} {r:>18.1f} {s:>15.1f} {r/s:>7.1f}x"
    print(line, flush=True); lines.append(line)
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nsaved -> {OUT}")
