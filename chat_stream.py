# Streaming chat: the whole conversation lives in the synapse grids.
#   python chat_stream.py [ckpt] [tokenizer.json]
# Works for byte-level and BPE checkpoints: the grids don't care what the
# atom is. Alphabet is picked from the checkpoint's vocab_size (256 = bytes);
# a BPE ckpt needs its tokenizer json (default bpe8k.json next to this file).
# Nothing is ever re-read: each position typed or generated is written into the
# state once, and every new position costs the same.
# RAW=1: completion mode for base (untuned) checkpoints: the text is fed in
#        verbatim and continued; no User:/Assistant: wrapper, no stop markers.
# Commands: /reset (zero the state), quit.
import os
import sys

import torch
import torch.nn.functional as F

from streaming import StreamingBDH

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "checkpoints", "bdh-100m-bytes-chat-v3.pt")
if not os.path.exists(CKPT):
    CKPT = os.path.join(HERE, "checkpoints", "bdh-100m-bytes.pt")
TOKENIZER = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "bpe8k.json")

END = "<|endoftext|>"
YOUR_TURN = "\nUser:"
TEMPERATURE = float(os.environ.get("TEMP", "0.8"))   # env TEMP=0.3 for repeatable answers
TOP_K = int(os.environ.get("TOPK", "40"))
torch.manual_seed(int(os.environ.get("SEED", "0")))
TRAINED_BLOCK = 512      # clocks beyond this position were never seen in training

device = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW = int(os.environ.get("WINDOW", "0")) or None   # e.g. WINDOW=480: approximate sliding window
RAW = os.environ.get("RAW", "0") == "1"                # completion mode for base checkpoints
RAW_NEW = int(os.environ.get("RAW_NEW", "300"))         # positions to generate per turn in RAW mode
sm = StreamingBDH(CKPT, device=device, dtype=torch.float32, window=WINDOW)
if WINDOW and WINDOW > TRAINED_BLOCK:
    print(f"WARNING: WINDOW={WINDOW} exceeds the trained block ({TRAINED_BLOCK}); reads past {TRAINED_BLOCK} hit "
          f"untrained clock offsets and output collapses. Use WINDOW<={TRAINED_BLOCK - 32}.")

if sm.cfg.vocab_size == 256:
    ALPHABET = "bytes"
    encode = lambda s: list(s.encode("utf-8"))
    decode = lambda ids: bytes(ids).decode("utf-8", errors="ignore")
    MAX_NEW = 420
    PROMPT = "User: {}\nAssistant: "          # the byte model was tuned with the trailing space
else:
    from tokenizers import Tokenizer
    ALPHABET = f"bpe-{sm.cfg.vocab_size}"
    tok = Tokenizer.from_file(TOKENIZER)
    encode = lambda s: tok.encode(s).ids
    decode = lambda ids: tok.decode(ids)
    MAX_NEW = 130
    PROMPT = "User: {}\nAssistant:"           # hard turn boundary

print(f"loaded {os.path.basename(CKPT)} [{ALPHABET}] on {device}  (streaming: grids persist across turns; "
      f"window={WINDOW or 'infinite'}; {'RAW completion mode' if RAW else 'chat mode'})")
print("type a message; /reset clears the grids; 'quit' to exit.\n")


def snapshot():
    return sm.grids.clone(), sm.pos


def restore(snap):
    sm.grids, sm.pos = snap


def etch(ids):
    """Feed atoms into the grids; return logits for the next atom."""
    lg = None
    for t in ids:
        lg = sm.step(t)
    return lg


@torch.no_grad()
def reply(logits):
    out, shown = [], 0
    snap = None                          # grids as they were before the last atom holding '\n'
    for _ in range(MAX_NEW):
        lg = logits / TEMPERATURE
        v, _ = torch.topk(lg, TOP_K)
        lg[lg < v[-1]] = float("-inf")
        t = int(torch.multinomial(F.softmax(lg, -1).cpu(), 1))
        if "\n" in decode([t]):
            snap = snapshot()
        out.append(t)
        logits = sm.step(t)
        text = decode(out)
        if END in text:
            text = text.split(END)[0]
            print(text[shown:], end="", flush=True)
            return text, True            # END already written
        if YOUR_TURN in text:            # it started writing the next user turn: roll back
            if snap is not None:
                restore(snap)
            text = text.split(YOUR_TURN)[0]
            print(text[shown:], end="", flush=True)
            return text, False
        safe = len(text) - len(END)      # hold back chars that may be a marker mid-arrival
        if safe > shown:
            print(text[shown:safe], end="", flush=True)
            shown = safe
    print(text[shown:], end="", flush=True)
    return text, False


while True:
    try:
        msg = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if msg.lower() in ("quit", "exit"):
        break
    if msg == "/reset":
        sm.reset()
        print("(grids cleared)\n")
        continue
    if RAW:                              # base model: just continue the document
        logits = etch(encode(msg))
        print("...> ", end="", flush=True)
        with torch.no_grad():
            out = []
            for _ in range(RAW_NEW):
                lg = logits / TEMPERATURE
                v, _ = torch.topk(lg, TOP_K)
                lg[lg < v[-1]] = float("-inf")
                t = int(torch.multinomial(F.softmax(lg, -1).cpu(), 1))
                out.append(t); logits = sm.step(t)
                if len(out) % 8 == 0:
                    print(decode(out[-8:]), end="", flush=True)
            print(decode(out[len(out) - len(out) % 8:]), end="", flush=True)
        print(f"\n   (pos {sm.pos} {ALPHABET[:5]} atoms; type more to continue the same document)\n")
        continue
    logits = etch(encode(PROMPT.format(msg)))
    print("bdh> ", end="", flush=True)
    text, ended = reply(logits)
    if not ended:
        etch(encode(END))                # close the turn
    warn = "  [past trained context; clocks untrained here]" if sm.pos > TRAINED_BLOCK else ""
    print(f"\n   (pos {sm.pos} {ALPHABET[:5]} atoms{warn})\n")
