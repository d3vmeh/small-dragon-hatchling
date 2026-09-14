# Trains an 8k BPE tokenizer on corpus.bin, then pre-tokenizes the corpus
# into corpus_bpe.bin (uint16 ids).
import os

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "corpus.bin")
TOK_PATH = os.path.join(HERE, "bpe8k.json")
OUT = os.path.join(HERE, "corpus_bpe.bin")
VOCAB = 8192

if not os.path.exists(TOK_PATH):
    print("training BPE-8k tokenizer...", flush=True)
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=[],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    # train on the first ~200MB
    sample_path = os.path.join(HERE, "_bpe_sample.txt")
    with open(DATA, "rb") as f, open(sample_path, "wb") as g:
        g.write(f.read(200_000_000))
    tok.train([sample_path], trainer)
    os.remove(sample_path)
    tok.save(TOK_PATH)
    print(f"tokenizer saved: {tok.get_vocab_size()} tokens", flush=True)

tok = Tokenizer.from_file(TOK_PATH)
if os.path.exists(OUT):
    print(f"tokenized corpus exists: {os.path.getsize(OUT)/1e9:.2f}GB", flush=True)
    raise SystemExit

print("tokenizing corpus...", flush=True)
CHUNK = 50_000_000
with open(DATA, "rb") as f, open(OUT + ".tmp", "wb") as g:
    leftover = b""
    while True:
        raw = f.read(CHUNK)
        if not raw:
            break
        buf = leftover + raw
        # cut at the last newline so no multi-byte character or word is split
        cut = buf.rfind(b"\n")
        if cut <= 0 or not raw:
            cut = len(buf)
        text, leftover = buf[:cut], buf[cut:]
        ids = tok.encode(text.decode("utf-8", errors="ignore")).ids
        g.write(np.asarray(ids, dtype=np.uint16).tobytes())
    if leftover:
        ids = tok.encode(leftover.decode("utf-8", errors="ignore")).ids
        g.write(np.asarray(ids, dtype=np.uint16).tobytes())
os.replace(OUT + ".tmp", OUT)
n_tok = os.path.getsize(OUT) // 2
n_byte = os.path.getsize(DATA)
print(f"tokenized: {n_tok/1e6:.0f}M tokens from {n_byte/1e6:.0f}M bytes "
      f"({n_byte/max(n_tok,1):.2f} bytes/token)", flush=True)
