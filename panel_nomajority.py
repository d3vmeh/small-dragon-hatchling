"""Break down the no-majority prompts of the five 100M judge panels (Table 6).

A prompt has no majority when neither model was chosen by two of the three judges. That can happen
three ways: every judge called it a tie, one judge chose a model and the other two called a tie, or
two judges chose opposite models. Reads the per-judge rulings files written by judge_pairs.py.
"""
import collections
import json

PANELS = [("", "v1 vs v2"), ("_alpha", "bytes v2 vs BPE v2"), ("_v23", "bytes v2 vs v3"),
          ("_bpe23", "BPE v2 vs v3"), ("_alpha3", "bytes v3 vs BPE v3")]

lines = ["no-majority prompts in the five 100M panels, from judge_rulings_100m*.json (overall verdict per judge)",
         f"{'panel':22s} {'prompts':>7s} {'no majority':>11s} {'all tie':>8s} {'one chose':>10s} {'split':>6s}"]
for tag, label in PANELS:
    rulings = json.load(open(f"judge_rulings_100m{tag}.json"))
    ids = set().union(*[set(j.keys()) for j in rulings.values()])
    kinds = collections.Counter()
    for i in ids:
        votes = [j[i]["overall"] for j in rulings.values() if i in j]
        c = collections.Counter(votes)
        if max(c.get("A", 0), c.get("B", 0)) >= 2:
            continue
        chosen = c.get("A", 0) + c.get("B", 0)
        if c.get("A", 0) and c.get("B", 0):
            kinds["split"] += 1
        elif chosen == 1:
            kinds["one chose"] += 1
        else:
            kinds["all tie"] += 1
    nm = sum(kinds.values())
    lines.append(f"{label:22s} {len(ids):7d} {nm:11d} {kinds['all tie']:8d} {kinds['one chose']:10d} {kinds['split']:6d}")
out = "\n".join(lines)
print(out)
open("results_panel_nomajority.txt", "w").write(out + "\n")
