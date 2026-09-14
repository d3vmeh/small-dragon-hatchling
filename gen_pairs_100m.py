# Blinded judge pairs for the 100M byte chat tunes: v1 vs v2.
# The 120 prompts in pairs2.json; replies via streaming (window 480).
# Output: pairs_100m.json (blinded, order-shuffled) + key_100m.json (the reveal).
#   ARMS=v2,v3 CK_A=... CK_B=... SUF=_v23 python -u gen_pairs_100m.py
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from streaming import StreamingBDH  # noqa: E402

# env ARMS=a,b  CK_A=path CK_B=path  SUF=_100m  (defaults: v1 vs v2)
ARMS = os.environ.get("ARMS", "v1,v2").split(",")
CK = {ARMS[0]: os.environ["CK_A"], ARMS[1]: os.environ["CK_B"]}
SUF = os.environ.get("SUF", "_100m")
END, YOUR_TURN, MAX_NEW = b"<|endoftext|>", b"\nUser:", 420
prompts = [(p["id"], p["prompt"]) for p in json.load(open(os.path.join(HERE, "pairs2.json")))]


def alphabet(sm):
    """(encode, decode, prompt_fmt, max_new) chosen from the ckpt's vocab, as in chat_stream.py"""
    if sm.cfg.vocab_size == 256:
        return (lambda s: list(s.encode("utf-8")), lambda ids: bytes(ids).decode("utf-8", errors="ignore"),
                "User: {}\nAssistant: ", MAX_NEW)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(HERE, "bpe8k.json"))
    return (lambda s: tok.encode(s).ids, lambda ids: tok.decode(ids), "User: {}\nAssistant:", 130)


@torch.no_grad()
def reply(sm, prompt, seed):
    encode, decode, fmt, max_new = alphabet(sm)
    sm.reset()
    g = torch.Generator().manual_seed(seed)
    lg = None
    for t in encode(fmt.format(prompt)):
        lg = sm.step(t)
    out = []
    for _ in range(max_new):
        l = lg / 0.8
        v, _ = torch.topk(l, 40)
        l[l < v[-1]] = float("-inf")
        t = int(torch.multinomial(F.softmax(l, -1).cpu(), 1, generator=g))
        out.append(t)
        text = decode(out)
        for stop in (END.decode(), YOUR_TURN.decode()):
            if stop in text:
                return text.split(stop)[0]
        lg = sm.step(t)
    return decode(out)


replies = {}
for arm, path in CK.items():
    sm = StreamingBDH(os.path.join(HERE, path), window=480)
    replies[arm] = {}
    for i, (pid, prompt) in enumerate(prompts):
        replies[arm][pid] = reply(sm, prompt, seed=int(pid))
        print(f"[{arm}] {i+1}/{len(prompts)} {prompt[:40]!r} -> {replies[arm][pid][:60]!r}", flush=True)
    del sm; torch.cuda.empty_cache()

rng = random.Random(100)
pairs, key = [], {}
for pid, prompt in prompts:
    a, b = (ARMS[0], ARMS[1]) if rng.random() < 0.5 else (ARMS[1], ARMS[0])
    pairs.append({"id": pid, "prompt": prompt, "A": replies[a][pid], "B": replies[b][pid]})
    key[pid] = {"A": a, "B": b}
json.dump(pairs, open(os.path.join(HERE, f"pairs{SUF}.json"), "w"), indent=1)
json.dump(key, open(os.path.join(HERE, f"key{SUF}.json"), "w"), indent=1)
print(f"wrote {len(pairs)} blinded pairs -> pairs{SUF}.json / key{SUF}.json", flush=True)
