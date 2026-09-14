# Compare exam runs: reads every exam_out/scores_<TAG>_s<SEED>.json and prints
# TAG x metric, averaged over seeds, with min/max in brackets.
#   python compare.py [exam_out]
import glob
import json
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "exam_out")


def flat(d, prefix=""):
    """scores json -> {metric: number}; nested per-qtype dict becomes G_sup.<qtype>."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, "G_sup." if k == "G_supported_by_qtype" else k + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("seed",):
            out[prefix + k] = float(v)
    return out


runs = {}                                    # tag -> list of flat dicts (one per seed)
for path in sorted(glob.glob(os.path.join(OUT, "scores_*_s*.json"))):
    if "_limit" in path:                     # partial (LIMIT) runs are not comparable
        continue
    s = json.load(open(path))
    runs.setdefault(s["tag"], []).append(flat(s))
if not runs:
    sys.exit(f"no scores files in {OUT}")

metrics = sorted({m for rs in runs.values() for r in rs for m in r})
tags = sorted(runs)
w = max(len(m) for m in metrics)
print(f"{'metric':{w}s} " + " ".join(f"{t:>26s}" for t in tags))
print(f"{'seeds':{w}s} " + " ".join(f"{len(runs[t]):>26d}" for t in tags))
for m in metrics:
    cells = []
    for t in tags:
        vals = [r[m] for r in runs[t] if m in r]
        if not vals:
            cells.append(f"{'-':>26s}")
        else:
            mean = sum(vals) / len(vals)
            cells.append(f"{mean:8.3f} [{min(vals):.2f},{max(vals):.2f}]".rjust(26))
    print(f"{m:{w}s} " + " ".join(cells))
