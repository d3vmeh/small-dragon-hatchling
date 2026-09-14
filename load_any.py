# Load a checkpoint as either a BDH or a GPT (transformer baseline), by the
# "arch" key train_scale.py writes ("gpt"); anything else is a BDH.
#   from load_any import load_model
#   m, ck = load_model(path)          # m on CPU, eval() NOT applied; cfg = m.config
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def build_model(ck):
    if ck.get("arch") == "gpt":
        import gpt
        return gpt.GPT(gpt.GPTConfig(**ck["config"]))
    import bdh
    return bdh.BDH(bdh.BDHConfig(**ck["config"]))


def load_model(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    m = build_model(ck)
    m.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
    return m, ck
