# Builds corpus_mix.bin: ~6GB with the same mix as the byte-100M training corpus (17% TinyStories,
# 83% FineWeb-Edu) for the matched-text BPE-100M run. CPU-only.
# Then tokenizes it with the existing bpe8k tokenizer -> corpus_mix_bpe.bin.
import os

import numpy as np
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TS = os.environ.get("TINYSTORIES", os.path.join(HERE, "tinystories.bin"))   # TinyStories train split as raw UTF-8 bytes
OUT = os.path.join(HERE, "corpus_mix.bin")
OUT_BPE = os.path.join(HERE, "corpus_mix_bpe.bin")
TS_BYTES = 1_020_000_000          # 17% of 6GB
FW_BYTES = 4_980_000_000          # 83%

if not os.path.exists(OUT):
    with open(OUT + ".tmp", "wb") as g:
        with open(TS, "rb") as f:
            g.write(f.read(TS_BYTES))
        print(f"tinystories slice: {g.tell()/1e9:.2f}GB", flush=True)
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        # skip the region used by the byte-100M corpus by jumping ahead deterministically
        it = iter(ds)
        for _ in range(120_000):
            next(it)
        target = TS_BYTES + FW_BYTES
        for doc in it:
            g.write(doc["text"].encode("utf-8"))
            g.write(b"\n<|endoftext|>\n")
            if g.tell() >= target:
                break
            if g.tell() % (1 << 30) < 4096:
                print(f"  {g.tell()/1e9:.1f}GB", flush=True)
    os.replace(OUT + ".tmp", OUT)
print(f"corpus_mix: {os.path.getsize(OUT)/1e9:.2f}GB", flush=True)

if not os.path.exists(OUT_BPE):
    tok = Tokenizer.from_file(os.path.join(HERE, "bpe8k.json"))
    CHUNK = 50_000_000
    with open(OUT, "rb") as f, open(OUT_BPE + ".tmp", "wb") as g:
        leftover = b""
        done = 0
        while True:
            raw = f.read(CHUNK)
            if not raw:
                break
            buf = leftover + raw
            cut = buf.rfind(b"\n")
            if cut <= 0:
                cut = len(buf)
            text, leftover = buf[:cut], buf[cut:]
            ids = tok.encode(text.decode("utf-8", errors="ignore")).ids
            g.write(np.asarray(ids, dtype=np.uint16).tobytes())
            done += len(text)
            if done % (1 << 30) < CHUNK:
                print(f"  tokenized {done/1e9:.1f}GB", flush=True)
        if leftover:
            ids = tok.encode(leftover.decode("utf-8", errors="ignore")).ids
            g.write(np.asarray(ids, dtype=np.uint16).tobytes())
    os.replace(OUT_BPE + ".tmp", OUT_BPE)
n = os.path.getsize(OUT_BPE) // 2
print(f"corpus_mix_bpe: {n/1e6:.0f}M tokens "
      f"({os.path.getsize(OUT)/max(n,1):.2f} bytes/token)", flush=True)
os._exit(0)   # the datasets library's threads can hang at shutdown
