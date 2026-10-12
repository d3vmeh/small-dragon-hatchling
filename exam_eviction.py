"""Story eviction per answer in the multi-turn exam, from exam_out/replies_*.json (Appendix A21).
Run from lab4090 with the CUDA venv (needs tokenizers): python exam_eviction.py > results_exam_eviction.txt

Layout per story item (fresh shelf): [prompt0][story reply + END][q1 prompt][a1 + END][q2 prompt][a2 + END]...
positions[k] = sm.pos after reply k (END included, etched separately when the reply did not end itself).
Window 480: when generating at position p, atoms with index < p - 480 are gone.
Story region = [P0, P0 + n_story) with n_story = len(encode(story text)).
Answer k starts at a_k = positions[k-1] + len(encode(prompt.format(q_k))) and ends at positions[k] (incl. END).
"""
import json, os, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__))
items = {it["id"]: it for it in json.load(open(f"{HERE}/multiturn_items.json"))}
END = "<|endoftext|>"
WINDOW = 480
from tokenizers import Tokenizer
tok = Tokenizer.from_file(f"{HERE}/bpe8k.json")

def coder(run):
    if run.startswith("bytes"):
        return (lambda s: list(s.encode("utf-8"))), "User: {}\nAssistant: "
    return (lambda s: tok.encode(s).ids), "User: {}\nAssistant:"

for f in sorted(glob.glob(f"{HERE}/exam_out/replies_*.json")):
    run = os.path.basename(f)[len("replies_"):-5]
    enc, prompt = coder(run)
    recs = [r for r in json.load(open(f)) if r["type"] == "G"]
    rows = []
    for r in recs:
        it = items[r["id"]]
        P0 = len(enc(prompt.format(it["turns"][0])))
        n_story = len(enc(r["replies"][0]))
        s0, s1 = P0, P0 + n_story
        pos = r["positions"]
        per_ans = []
        for k in (1, 2, 3):
            a_start = pos[k - 1] + len(enc(prompt.format(it["turns"][k])))
            a_end = pos[k]
            ev_start = min(max(a_start - WINDOW - s0, 0), n_story)
            ev_end = min(max(a_end - WINDOW - s0, 0), n_story)
            per_ans.append((ev_start / n_story, ev_end / n_story, a_start, a_end))
        rows.append((r["id"], n_story, P0, per_ans))
    print(f"\n== {run}: {len(rows)} story items, story length median {sorted(x[1] for x in rows)[len(rows)//2]} atoms, "
          f"max {max(x[1] for x in rows)}")
    for k in range(3):
        st = [row[3][k][0] for row in rows]; en = [row[3][k][1] for row in rows]
        n_st = sum(x > 0 for x in st); n_en = sum(x > 0 for x in en)
        n_full_en = sum(x >= 1.0 for x in en)
        a_starts = sorted(row[3][k][2] for row in rows)
        print(f"  answer {k+1}: story partly evicted at answer START in {n_st:2d}/{len(rows)} items "
              f"(mean evicted fraction {sum(st)/len(st):.3f}, max {max(st):.2f}); "
              f"by answer END in {n_en:2d}/{len(rows)} (mean {sum(en)/len(en):.3f}, max {max(en):.2f}, fully gone {n_full_en}); "
              f"start position median {a_starts[len(a_starts)//2]}, over 480 in {sum(p > 480 for p in a_starts)}")
    worst = sorted(rows, key=lambda row: -row[3][2][1])[:3]
    for w in worst:
        print(f"    {w[0]} story {w[1]} atoms, prompt0 {w[2]}: evicted at (start,end) per answer "
              + ", ".join(f"({a:.2f},{b:.2f})" for a, b, _, _ in w[3]))

# ---- join with the 2-of-3 majority labels: supported rate for answers with story intact vs partly evicted at answer START
import collections
print("\n== supported rate (2-of-3 majority, NOMAJ kept in the denominator) by whether any story atom had left the window when the answer began")
pool = collections.defaultdict(lambda: [0, 0])   # (model, q, evicted?) -> [supported, n]
for f in sorted(glob.glob(f"{HERE}/exam_out/replies_bytes_*.json")):
    run = os.path.basename(f)[len("replies_"):-5]
    enc, prompt = coder(run)
    ag = json.load(open(f"{HERE}/exam_out/agreement_{run}.json"))["rows"]
    lab = {}
    for r in ag:
        votes = [r[j] for j in ("opus", "gpt", "deepseek") if r.get(j) in ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "EVASIVE")]
        c = collections.Counter(votes).most_common()
        lab[(r["id"], r["q"])] = c[0][0] if c and c[0][1] >= 2 else "NOMAJ"
    model = run.rsplit("_", 1)[0]
    for r in json.load(open(f)):
        if r["type"] != "G": continue
        it = items[r["id"]]
        P0 = len(enc(prompt.format(it["turns"][0]))); n_story = len(enc(r["replies"][0])); pos = r["positions"]
        for k in (1, 2, 3):
            a_start = pos[k - 1] + len(enc(prompt.format(it["turns"][k])))
            ev = min(max(a_start - WINDOW - P0, 0), n_story) / n_story
            key = (model, k, ev > 0)
            pool[key][1] += 1; pool[key][0] += lab[(r["id"], k - 1)] == "SUPPORTED"
            key2 = (model, "all", ev > 0)
            pool[key2][1] += 1; pool[key2][0] += lab[(r["id"], k - 1)] == "SUPPORTED"
            a_end = pos[k]
            ev_end = min(max(a_end - WINDOW - P0, 0), n_story) / n_story
            key3 = (model, f"Q{k} by answer END", ev_end > 0)
            pool[key3][1] += 1; pool[key3][0] += lab[(r["id"], k - 1)] == "SUPPORTED"
for model in ("bytes_v2", "bytes_v3"):
    for k in (1, 2, 3, "all", "Q1 by answer END", "Q2 by answer END", "Q3 by answer END"):
        a, b = pool[(model, k, False)], pool[(model, k, True)]
        print(f"  {model} {k if isinstance(k, str) and k.startswith('Q') else 'Q' + str(k)}: story intact {a[0]}/{a[1]}" + (f" = {a[0]/a[1]:.0%}" if a[1] else "")
              + f" | partly evicted {b[0]}/{b[1]}" + (f" = {b[0]/b[1]:.0%}" if b[1] else ""))
