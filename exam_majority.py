"""Recompute the exam's story-item rates under the 2-of-3 majority of (opus, gpt, deepseek) from
exam_out/agreement_<tag>_s<seed>.json, and average v3 over seeds. Prints one block per model."""
import json, os, glob, collections, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
def rates(rows):
    labs = []
    for r in rows:
        votes = [r[j] for j in ("opus", "gpt", "deepseek") if r.get(j) in ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "EVASIVE")]
        c = collections.Counter(votes).most_common()
        labs.append((r["qtype"], c[0][0] if c and c[0][1] >= 2 else "NOMAJ"))
    n = len(labs); cnt = collections.Counter(l for _, l in labs)
    S, C = cnt["SUPPORTED"], cnt["CONTRADICTED"]
    hm = [l for q, l in labs if q == "howmany"]
    return {"supported": S / n, "contradicted": C / n, "unsupported": cnt["UNSUPPORTED"] / n, "evasive": cnt["EVASIVE"] / n,
            "nomaj": cnt["NOMAJ"] / n, "decided_correct": S / (S + C) if S + C else float("nan"),
            "howmany_supported": (hm.count("SUPPORTED") / len(hm)) if hm else float("nan"), "n": n}
runs = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(HERE, "exam_out", "agreement_*_s[0-9].json"))):
    d = json.load(open(f)); runs[d["tag"]].append((d["seed"], rates(d["rows"])))
keys = ["supported", "contradicted", "unsupported", "evasive", "nomaj", "decided_correct", "howmany_supported"]
print(f"{'metric (2-of-3 majority)':28s}" + "".join(f"{t:>22s}" for t in ("bytes_v2", "bytes_v3", "bpe_v2", "bpe_v3")))
for k in keys:
    line = f"{k:28s}"
    for t in ("bytes_v2", "bytes_v3", "bpe_v2", "bpe_v3"):
        vals = [r[k] for _, r in sorted(runs.get(t, []))]
        if not vals: line += f"{'-':>22s}"; continue
        m = statistics.mean(vals); rng = f" [{min(vals):.2f},{max(vals):.2f}]" if len(vals) > 1 else ""
        line += f"{m:>10.3f}{rng:>12s}"
    print(line)
print("seeds:", {t: sorted(s for s, _ in v) for t, v in runs.items()})
