"""End-to-end list accuracy in the multi-turn exam: a reply that formed no list counts as wrong, so the
denominator is every position question asked (9 first, 12 second, 6 last, 13 how-many per run) instead of the
replies that parsed as a list (Table 7, Appendix A11). Reads exam_out/replies_*.json; python exam_list_e2e.py"""
import json, glob, os, collections, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
items = {i["id"]: i for i in json.load(open(f"{HERE}/multiturn_items.json"))}
Q = ("first", "second", "last", "howmany")
per_run = {}
for f in sorted(glob.glob(f"{HERE}/exam_out/replies_*.json")):
    run = os.path.basename(f)[len("replies_"):-5]
    c = {q: [0, 0, 0] for q in Q}            # correct, scored (valid list), asked
    for r in json.load(open(f)):
        if r["type"] != "L": continue
        for q, v in zip(items[r["id"]]["qtypes"], r["verdicts"]):
            c[q][2] += 1; c[q][1] += v != "no-list"; c[q][0] += v == "correct"
    per_run[run] = c
    print(f"{run:12s} " + " | ".join(f"{q}: {c[q][0]}/{c[q][1]} of valid lists, {c[q][0]}/{c[q][2]} of all asked" for q in Q))
print("\nmean over seeds (valid-list rate as in Table 7 / end-to-end rate)")
for model in ("bytes_v2", "bytes_v3", "bpe_v2", "bpe_v3"):
    runs = [c for run, c in per_run.items() if run.rsplit("_", 1)[0] == model]
    cells = []
    for q in Q:
        valid = statistics.mean(c[q][0] / c[q][1] if c[q][1] else 0 for c in runs)
        e2e = statistics.mean(c[q][0] / c[q][2] for c in runs)
        cells.append(f"{q} {valid:.0%} / {e2e:.0%}")
    print(f"{model:9s} ({len(runs)} seed{'s' if len(runs) > 1 else ''})  " + "  ".join(cells))
