# Judge-agreement check for the multi-turn exam's type-G (story) labels.
# exam.py labels every story answer with ONE judge (claude-opus-5). This re-labels one
# existing run with gpt-5.6-sol and deepseek-v4-pro (same JUDGE_SYSTEM prompt and user
# message as exam.judge_one, verbatim) and reports pairwise agreement, Cohen's kappa,
# label distributions and the SUPPORTED / decided-correct rates under each judge.
# API only (no GPU). Does not modify the exam result files.
#   python -u exam_judge_agreement.py           # full 120 answers
#   LIMIT=6 python -u exam_judge_agreement.py   # first 6 answers
# Env: TAG (bpe_v3), SEED (0), LIMIT (first n answers), OUT (exam_out), WORKERS (4 per judge)
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exam import JUDGE_SYSTEM, load_items                       # noqa: E402  (no torch at import)
from judge_pairs import DEEPSEEK_MODEL, OPENAI_MODEL, load_env  # noqa: E402

TAG = os.environ.get("TAG", "bpe_v3")
SEED = int(os.environ.get("SEED", "0"))
OUT = os.environ.get("OUT", os.path.join(HERE, "exam_out"))
LIMIT = int(os.environ.get("LIMIT", "0")) or None
WORKERS = int(os.environ.get("WORKERS", "4"))
LABELS = ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "EVASIVE")
JUDGES = {"opus": "claude-opus-5 (existing exam verdicts)", "gpt": OPENAI_MODEL, "deepseek": DEEPSEEK_MODEL}


def user_msg(story, q, a):                    # identical to exam.judge_one
    return f"STORY:\n{story}\n\nQUESTION: {q}\nANSWER: {a}"


def parse(text):
    m = re.search(r"\b(SUPPORTED|CONTRADICTED|UNSUPPORTED|EVASIVE)\b", (text or "").upper())
    return m.group(1) if m else "UNPARSED"


def make_judge(base_url, api_key_env, model, extra):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ[api_key_env], base_url=base_url)

    def judge(story, q, a):
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": JUDGE_SYSTEM},
                              {"role": "user", "content": user_msg(story, q, a)}],
                    **extra)
                return parse(resp.choices[0].message.content)
            except Exception as e:           # rate limit, timeout, 5xx: back off and retry
                print(f"[{model}] attempt {attempt + 1}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                time.sleep(10 * (attempt + 1))
        return "ERROR"
    return judge


def cohen_kappa(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a in LABELS and b in LABELS]
    n = len(pairs)
    if n == 0:
        return float("nan"), 0
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == l for a, _ in pairs) / n) * (sum(b == l for _, b in pairs) / n) for l in LABELS)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan"), n


def majority(votes):
    v = [x for x in votes if x in LABELS]
    for l in LABELS:
        if v.count(l) >= 2:
            return l
    return "NONE"


def rates(labels):
    n = sum(l in LABELS for l in labels)
    c = {l: labels.count(l) for l in LABELS}
    dec = c["SUPPORTED"] + c["CONTRADICTED"]
    return {"n": n, "share": {l: c[l] / n if n else float("nan") for l in LABELS},
            "supported_rate": c["SUPPORTED"] / n if n else float("nan"),
            "decided_correct": c["SUPPORTED"] / dec if dec else float("nan"), "decided_n": dec}


def report(rows):
    J = ["opus", "gpt", "deepseek"]
    col = {j: [r[j] for r in rows] for j in J}
    col["majority"] = [majority([r[j] for j in J]) for r in rows]
    L = [f"== judge agreement on exam story labels: {TAG} seed {SEED}{f' LIMIT={LIMIT}' if LIMIT else ''}"
         f"  ({time.strftime('%Y-%m-%d %H:%M')})", f"  answers: {len(rows)}   judges: " +
         "  ".join(f"{k}={v}" for k, v in JUDGES.items())]
    errs = {j: sum(v not in LABELS for v in col[j]) for j in J}
    L.append("  non-label outputs (ERROR/UNPARSED): " + "  ".join(f"{j}={e}" for j, e in errs.items()))
    L.append("-- pairwise agreement (exact label) and Cohen's kappa, over answers where both gave a label")
    for a, b in (("opus", "gpt"), ("opus", "deepseek"), ("gpt", "deepseek")):
        k, n = cohen_kappa(col[a], col[b])
        agree = sum(x == y for x, y in zip(col[a], col[b]) if x in LABELS and y in LABELS)
        L.append(f"  {a:8s} vs {b:8s}  agree {agree:3d}/{n:3d} = {agree / n if n else float('nan'):.3f}   kappa {k:.3f}")
    full = [r for r in rows if all(r[j] in LABELS for j in J)]
    una = sum(len({r[j] for j in J}) == 1 for r in full)
    L.append(f"  3-way unanimous      {una:3d}/{len(full):3d} = {una / len(full) if full else float('nan'):.3f}")
    L.append(f"  majority undecided (all three differ): {sum(m == 'NONE' for m in col['majority'])}")
    L.append("-- label distribution, SUPPORTED rate, decided-correct rate S/(S+C)")
    L.append(f"  {'judge':10s} {'n':>4s} {'SUPP':>6s} {'CONTR':>6s} {'UNSUP':>6s} {'EVAS':>6s} | {'S rate':>7s} {'S/(S+C)':>8s} {'(S+C)':>5s}")
    for j in J + ["majority"]:
        r = rates(col[j]); s = r["share"]
        L.append(f"  {j:10s} {r['n']:4d} {s['SUPPORTED']:6.3f} {s['CONTRADICTED']:6.3f} {s['UNSUPPORTED']:6.3f} "
                 f"{s['EVASIVE']:6.3f} | {r['supported_rate']:7.3f} {r['decided_correct']:8.3f} {r['decided_n']:5d}")
    L.append("-- confusion: rows = opus, cols = 2-of-3 majority")
    cols = list(LABELS) + ["NONE"]
    L.append(f"  {'opus \\ maj':13s}" + "".join(f"{c[:6]:>8s}" for c in cols))
    for a in LABELS:
        L.append(f"  {a:13s}" + "".join(f"{sum(o == a and m == c for o, m in zip(col['opus'], col['majority'])):8d}" for c in cols))
    L.append("-- SUPPORTED rate by question type under each judge")
    qts = sorted({r["qtype"] for r in rows})
    L.append(f"  {'qtype':10s}" + "".join(f"{j:>10s}" for j in J + ["majority"]) + "     n")
    for qt in qts:
        sub = [r for r in rows if r["qtype"] == qt]
        L.append(f"  {qt:10s}" + "".join(f"{rates([r[j] if j != 'majority' else majority([r[x] for x in J]) for r in sub])['supported_rate']:10.3f}"
                                         for j in J + ["majority"]) + f"  {len(sub):4d}")
    return "\n".join(L)


def main():
    load_env()
    items = {it["id"]: it for it in load_items() if it["type"] == "G"}
    recs = [r for r in json.load(open(os.path.join(OUT, f"replies_{TAG}_s{SEED}.json"))) if r["type"] == "G"]
    rows = []
    for r in recs:
        it = items[r["id"]]; story = r["replies"][0]
        for j, (q, qt, ans) in enumerate(zip(it["turns"][1:], it["qtypes"], r["replies"][1:])):
            rows.append({"id": r["id"], "q": j, "qtype": qt, "question": q, "story": story, "answer": ans,
                         "opus": r["verdicts"][j]})
    rows = rows[:LIMIT] if LIMIT else rows
    print(f"{len(rows)} answers to re-judge with {OPENAI_MODEL} and {DEEPSEEK_MODEL}", flush=True)
    judges = {"gpt": make_judge(None, "OPENAI_API_KEY", OPENAI_MODEL, {"reasoning_effort": "medium"}),
              "deepseek": make_judge("https://api.deepseek.com", "DEEPSEEK_API_KEY", DEEPSEEK_MODEL, {})}

    def run(name):
        f = judges[name]
        with ThreadPoolExecutor(WORKERS) as ex:
            out = list(ex.map(lambda r: f(r["story"], r["question"], r["answer"]), rows))
        for r, v in zip(rows, out):
            r[name] = v
        print(f"  {name} done: " + "  ".join(f"{l}={out.count(l)}" for l in LABELS + ("ERROR", "UNPARSED")), flush=True)
        return name
    t0 = time.time()
    with ThreadPoolExecutor(2) as ex:                # one worker per provider
        list(ex.map(run, judges))
    print(f"judging done in {(time.time() - t0) / 60:.1f} min", flush=True)
    for r in rows:
        print(f"  {r['id']} q{r['q'] + 1} [{r['qtype']:7s}] opus={r['opus']:12s} gpt={r['gpt']:12s} "
              f"deepseek={r['deepseek']:12s} | {r['answer'][:50]!r}", flush=True)
    suffix = f"_limit{LIMIT}" if LIMIT else ""
    slim = [{k: r[k] for k in ("id", "q", "qtype", "question", "answer", "opus", "gpt", "deepseek")} for r in rows]
    json.dump({"tag": TAG, "seed": SEED, "judges": JUDGES, "system_prompt": JUDGE_SYSTEM, "rows": slim},
              open(os.path.join(OUT, f"agreement_{TAG}_s{SEED}{suffix}.json"), "w"), indent=1)
    t = report(rows)
    print(t)
    with open(os.path.join(OUT, f"agreement_{TAG}_s{SEED}{suffix}.txt"), "w") as f:
        f.write(t + "\n")


if __name__ == "__main__":
    main()
