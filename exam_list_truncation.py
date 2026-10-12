"""How often a scored list reply had MORE items than requested, and whether scoring against the full parsed
list instead of the first n items changes any Table 7 list rate (Appendix A11 note).
Run from lab4090 with the CUDA venv: python exam_list_truncation.py > results_exam_list_truncation.txt
Scoring rules copied from exam.py (norm, parse_list, score_L). "full" keeps every parsed item, so "last" is the
model's final item and "how many" expects the parsed count."""
import json, os, re, glob, collections
HERE = os.path.dirname(os.path.abspath(__file__))
NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
ARTICLES = {"a", "an", "the", "some", "and"}

def norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return " ".join(w for w in s.split() if w not in ARTICLES)

def parse_list(text):
    if len(re.findall(r"[.!?]\s", text)) > 0:
        return []
    body = text.split(":")[-1] if ":" in text else text
    body = re.sub(r"(?m)^\s*(\d+[.)]|[-*])\s*", ",", body)
    parts = re.split(r",|;|\n|\band\b|\bor\b", body)
    parts = [re.sub(r"[.!?]+$", "", p.strip()).lower() for p in parts]
    parts = [norm(p) for p in parts]
    return [p for p in parts if p and len(p.split()) <= 3]

def score(items, recs, truncate):
    pos = {"first": [0, 0], "second": [0, 0], "last": [0, 0]}; hm = [0, 0]; longer = 0; scored = 0
    for it, r in zip(items, recs):
        if it["type"] != "L": continue
        lst = parse_list(r["replies"][0])
        if len(lst) < it["n"]: continue
        scored += 1; longer += len(lst) > it["n"]
        if truncate: lst = lst[:it["n"]]
        for q, ans in zip(it["qtypes"], r["replies"][1:]):
            a = norm(ans)
            if q == "howmany":
                n = it["n"] if truncate else len(lst)
                ok = bool(re.search(rf"\b({NUM.get(n, str(n))}|{n})\b", a)); hm[0] += ok; hm[1] += 1
            else:
                target = lst[{"first": 0, "second": 1, "last": -1}[q]]
                others = [x for x in lst if x != target]
                has = lambda x: re.search(rf"\b{re.escape(x)}\b", a) is not None
                ok = has(target) and not any(has(o) for o in others); pos[q][0] += ok; pos[q][1] += 1
    return scored, longer, pos, hm

items = json.load(open(f"{HERE}/multiturn_items.json"))
print("run            scored  longer-than-asked | first  second  last  howmany (scored against first n) | same, against the full list")
for f in sorted(glob.glob(f"{HERE}/exam_out/replies_*.json")):
    run = os.path.basename(f)[len("replies_"):-5]
    recs = json.load(open(f))
    recs = [r for r in recs if r["type"] == "L"]
    its = [it for it in items if it["type"] == "L"]
    assert [r["id"] for r in recs] == [it["id"] for it in its], run
    s, longer, p, h = score(its, recs, True); s2, _, p2, h2 = score(its, recs, False)
    fmt = lambda p, h: "  ".join(f"{p[k][0]}/{p[k][1]}" for k in ("first", "second", "last")) + f"  {h[0]}/{h[1]}"
    print(f"{run:14s} {s:6d}  {longer:5d}             | {fmt(p, h)} | {fmt(p2, h2)}")
