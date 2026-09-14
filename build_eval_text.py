# Builds eval_text_100m.bin, the held-out text no model saw in training:
#   83% FineWeb-Edu from shard 013 (the training corpora used shards 000-005 only)
#   and 17% TinyStories validation split (training used the train split). CPU only.
#   python -u build_eval_text.py
import os
from datasets import load_dataset
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval_text_100m.bin")
FW, TS = 2_500_000, 500_000     # bytes
fw = bytearray()
ds = load_dataset("HuggingFaceFW/fineweb-edu", data_files="sample/10BT/013_00000.parquet", split="train", streaming=True)
for doc in ds:
    fw += doc["text"].encode("utf-8") + b"\n<|endoftext|>\n"
    if len(fw) >= FW: break
ts = bytearray()
for doc in load_dataset("roneneldan/TinyStories", split="validation", streaming=True):
    ts += doc["text"].encode("utf-8") + b"\n<|endoftext|>\n"
    if len(ts) >= TS: break
with open(OUT, "wb") as f:
    f.write(bytes(fw[:FW])); f.write(bytes(ts[:TS]))
print(f"wrote {OUT}: {FW/1e6:.1f}MB fineweb-edu shard 013 + {TS/1e6:.1f}MB tinystories validation", flush=True)
os._exit(0)
