# Planted-fact probe on the streamed state (no chat tuning needed).
#   CKPT=checkpoints/bdh-100m-bytes.pt TAG=bytes_100m NEEDLES=a,b,.. SEED=0 python -u needle.py
# Write "The secret word is <W>." then D characters of filler text, then ask
# "The secret word is" and score the model's -log p over the bytes of " <W>."
# (teacher-forced), plus greedy top-1 recall. Baseline = same filler, no needle.
# Conditions: state that never forgets vs approximate sliding window 480.
# Filler: data/input.txt (the tiny Shakespeare text), slice chosen by SEED.
# Writes needle_results_<TAG>_n<words>_s<SEED>[_key][_cue][_two][_first].txt.
import math
import os

import torch

from streaming import StreamingBDH

HERE = os.path.dirname(os.path.abspath(__file__))
NEEDLES = os.environ.get("NEEDLES", "banana,purple,rocket").split(",")   # env NEEDLES=a,b,c
SEED = int(os.environ.get("SEED", "0"))                                    # picks a different filler slice
DISTS = [0, 64, 128, 256, 384, 600, 1024]
WINDOWS = [None, 480]
CKPT = os.environ.get("CKPT", os.path.join(HERE, "checkpoints", "bdh-100m-bytes.pt"))
TAG = os.environ.get("TAG", "bytes_100m")
FIRST = bool(os.environ.get("FIRST"))                                      # FIRST=1: also report gain on the word's first atom vs the rest
PLANT_D, CUE_D = "The secret word is {w}.", "The secret word is"
DISTRACT = os.environ.get("DISTRACT")   # DISTRACT="The dog's name is {w}." -> plant a SECOND word under
                                        # this key; the cue still asks for the first. Separates retrieval
                                        # by key from naming whichever word looks out of place.
PLANT = os.environ.get("PLANT", PLANT_D)     # PLANT="The dog's name is {w}." -> mismatched key (plant and cue share no prefix)
CUE = os.environ.get("CUE", CUE_D)           # CUE="The word I told you was"  -> paraphrased request
_sfx = ("_key" if PLANT != PLANT_D else "") + ("_cue" if CUE != CUE_D else "") + ("_two" if DISTRACT else "") + ("_first" if FIRST else "")
OUT = os.path.join(HERE, (f"needle_results_{TAG}.txt" if SEED == 0 and len(NEEDLES) == 3 else f"needle_results_{TAG}_n{len(NEEDLES)}_s{SEED}.txt").replace(".txt", _sfx + ".txt"))

_off = 5000 + SEED * 40000
raw = open(os.path.join(HERE, "data", "input.txt"), "rb").read()[_off:_off + 15000].decode(errors="ignore")
filler = " ".join(raw.split())
assert all(n not in filler for n in NEEDLES), "filler contains a needle"

device = "cuda" if torch.cuda.is_available() else "cpu"
models = {w: StreamingBDH(CKPT, device=device, dtype=torch.float32, window=w) for w in WINDOWS}
# alphabet from the checkpoint (as in chat_stream.py): distances stay in CHARACTERS of filler
if next(iter(models.values())).cfg.vocab_size == 256:
    encode = lambda s: list(s.encode("utf-8")); ALPHABET = "bytes"
else:
    from tokenizers import Tokenizer
    _tok = Tokenizer.from_file(os.path.join(HERE, "bpe8k.json"))
    encode = lambda s: _tok.encode(s).ids; ALPHABET = "bpe8k"


@torch.no_grad()
def score(sm, prefix_text: str, target_text: str):
    """-log p (nats) of target given prefix, and whether greedy decode hits it.
    BPE: the prefix+target are tokenized jointly so the seam matches training; the
    target atoms are those whose span starts at or after the prefix boundary."""
    sm.reset()
    if ALPHABET == "bytes":
        prefix, target = encode(prefix_text), encode(target_text)
    else:
        enc = _tok.encode(prefix_text + target_text)
        cut = next(i for i, (o0, o1) in enumerate(enc.offsets) if o0 >= len(prefix_text))
        prefix, target = enc.ids[:cut], enc.ids[cut:]
    lg = None
    for t in prefix:
        lg = sm.step(t)
    nats, greedy, per_atom = 0.0, [], []
    for t in target:
        logp = torch.log_softmax(lg, -1)
        nats -= logp[t].item(); per_atom.append(-logp[t].item())
        greedy.append(int(lg.argmax()))
        lg = sm.step(t)
    score.last_per_atom = per_atom          # FIRST=1: per-atom nats of the last call
    return nats, greedy == target


lines = [f"# needle probe, {TAG} ({os.path.basename(CKPT)}, {ALPHABET}, distances in characters), fp32 {device}; nats = -log p of ' <word>.' "
         f"after the cue; baseline = same filler, no plant; NEEDLES={','.join(NEEDLES)} SEED={SEED} PLANT={PLANT!r} CUE={CUE!r}"
         + (f" DISTRACT={DISTRACT!r} (second word planted under that key, alternating order)" if DISTRACT else ""),
         f"{'window':>8} {'dist':>5} {'needle nats':>12} {'baseline':>9} {'gain':>6} {'top1 recall':>12}" + (f" {'said distractor':>15}" if DISTRACT else "") + (f" {'gain@1st':>10} {'gain@rest':>9}" if FIRST else "")]
print("\n".join(lines), flush=True)
for w in WINDOWS:
    sm = models[w]
    for d in DISTS:
        n_sum = b_sum = hits = 0; f_sum = r_sum = 0.0; dhits = 0
        for wi, word in enumerate(NEEDLES):
            fill = filler[:d].rstrip() + (" " if d else "")
            query = CUE
            target = f" {word}."
            other = NEEDLES[(wi + 1) % len(NEEDLES)]          # the distractor, planted under DISTRACT
            if DISTRACT:
                a, b = PLANT.format(w=word), DISTRACT.format(w=other)
                head = f"{a} {b}" if wi % 2 == 0 else f"{b} {a}"   # alternate which plant comes first
            else:
                head = PLANT.format(w=word)
            nats, hit = score(sm, f"{head} {fill}{query}", target)
            if DISTRACT:
                _, dhit = score(sm, f"{head} {fill}{query}", f" {other}.")
                dhits += dhit
            pa_n = score.last_per_atom
            base, _ = score(sm, f"{fill}{query}", target)
            pa_b = score.last_per_atom
            n_sum += nats; b_sum += base; hits += hit
            if FIRST:   # split the word: its first atom (after the leading space for bytes) vs the rest (before the final '.')
                i0 = 1 if ALPHABET == "bytes" else 0
                f_sum += pa_b[i0] - pa_n[i0]; r_sum += sum(pa_b[i0+1:-1]) - sum(pa_n[i0+1:-1])
        k = len(NEEDLES)
        line = (f"{str(w or 'inf'):>8} {d:>5} {n_sum/k:>12.2f} {b_sum/k:>9.2f} "
                f"{(b_sum-n_sum)/k:>6.2f} {hits:>7}/{k}" + (f" {str(dhits)+'/'+str(k):>15}" if DISTRACT else "") + (f" {f_sum/k:>10.2f} {r_sum/k:>9.2f}" if FIRST else ""))
        print(line, flush=True); lines.append(line)
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"saved -> {OUT}")
