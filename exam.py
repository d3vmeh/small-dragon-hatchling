# Multi-turn exam: does a chat model answer questions about
# the story it JUST wrote (type G, judged) and list-position questions (type L, code)?
# Each item starts from a zeroed state (sm.reset()); turns are streamed in once.
#   CKPT=... TAG=... SEED=0 JUDGE=0 \
#       python -u exam.py
# Env: CKPT, TAG, SEED (0), JUDGE (0/1), OUT (exam_out), LIMIT (first n items OF EACH TYPE), TOKENIZER,
#      JUDGE_ONLY=1 -> skip generation, judge+score the existing replies file (no torch).
import json
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TAG = os.environ.get("TAG", "model")
SEED = int(os.environ.get("SEED", "0"))
OUT = os.environ.get("OUT", os.path.join(HERE, "exam_out"))
LIMIT = int(os.environ.get("LIMIT", "0")) or None
ITEMS_FILE = os.path.join(HERE, "multiturn_items.json")
END, YOUR_TURN = "<|endoftext|>", "\nUser:"
TEMPERATURE, TOP_K, WINDOW = 0.8, 40, 480
JUDGE_MODEL = "claude-opus-5"

# ---------------- items (fixed file, generated once with RNG 7) ----------------
STORY_TPL = ["tell me a story about {}", "write a story about {}", "write a short story about {}",
             "can you tell me a story about {}", "please tell me a short story about {}"]
# topic, character noun, pronoun, countable things
TOPICS = [("a girl who finds something in the garden", "girl", "she", "flowers"),
          ("a boy and his pet", "boy", "he", "pets"), ("a trip to the sea", "child", "they", "boats"),
          ("a dog who goes on an adventure", "dog", "it", "dogs"), ("two friends at the park", "friend", "they", "friends"),
          ("a cat who gets lost", "cat", "it", "cats"), ("a girl and her grandmother", "girl", "she", "cookies"),
          ("a boy who builds something", "boy", "he", "blocks"), ("a family picnic", "family", "they", "sandwiches"),
          ("a bird who wants to fly", "bird", "it", "birds"), ("a rainy day at home", "child", "they", "toys"),
          ("a kid's first day at school", "kid", "they", "kids"), ("a trip to the farm", "child", "they", "animals"),
          ("a boy who finds a box", "boy", "he", "boxes"), ("a girl who plants a seed", "girl", "she", "seeds")]
Q_TPL = {"name": "What was the {character}'s name?", "find": "What did {pronoun} find?", "color": "What color was it?",
         "where": "Where did the story happen?", "howmany": "How many {things} were there?",
         "end": "What happened at the end?", "who": "Who else was in the story?"}
CATEGORIES = ["animals", "colors", "fruits", "toys", "things in a kitchen", "vegetables", "things you wear",
              "things in a garden", "things at the beach", "things in a school"]
POS_TPL = {"first": "what was the first one you named?", "second": "what was the second one?",
           "last": "what was the last one you named?", "howmany": "how many did you name?"}
NUM = {3: "three", 4: "four"}


def make_items():
    rng = random.Random(7)
    items = []
    for i in range(40):
        topic, ch, pr, th = TOPICS[i % len(TOPICS)]
        qs = rng.sample(list(Q_TPL), 3)
        turns = [rng.choice(STORY_TPL).format(topic)] + \
                [Q_TPL[q].format(character=ch, pronoun=pr, things=th) for q in qs]
        items.append({"id": f"G{i:02d}", "type": "G", "turns": turns, "qtypes": qs})
    for i in range(20):
        n, cat = rng.choice([3, 4]), CATEGORIES[i % len(CATEGORIES)]
        qs = rng.sample(list(POS_TPL), 2)
        items.append({"id": f"L{i:02d}", "type": "L", "n": n, "turns": [f"name {NUM[n]} {cat}"] + [POS_TPL[q] for q in qs],
                      "qtypes": qs})
    return items


def load_items():
    if not os.path.exists(ITEMS_FILE):
        json.dump(make_items(), open(ITEMS_FILE, "w"), indent=1)
    return json.load(open(ITEMS_FILE))


# ---------------- generation (chat_stream.py logic, no rollback: just stop) ----------------
def run_model(items):
    import torch
    import torch.nn.functional as F
    from streaming import StreamingBDH
    sm = StreamingBDH(os.environ["CKPT"], device="cuda", dtype=torch.float32, window=WINDOW)
    if sm.cfg.vocab_size == 256:
        encode = lambda s: list(s.encode("utf-8"))
        decode = lambda ids: bytes(ids).decode("utf-8", errors="ignore")
        max_new, prompt = 420, "User: {}\nAssistant: "     # the byte model was tuned with the trailing space
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.environ.get("TOKENIZER", os.path.join(HERE, "bpe8k.json")))
        encode, decode = lambda s: tok.encode(s).ids, lambda ids: tok.decode(ids)
        max_new, prompt = 130, "User: {}\nAssistant:"

    def etch(ids):
        lg = None
        for t in ids:
            lg = sm.step(t)
        return lg

    @torch.no_grad()
    def reply(logits, gen):
        out = []
        for _ in range(max_new):
            lg = logits / TEMPERATURE
            v, _ = torch.topk(lg, TOP_K)
            lg[lg < v[-1]] = float("-inf")
            t = int(torch.multinomial(F.softmax(lg, -1).cpu(), 1, generator=gen))
            out.append(t)
            logits = sm.step(t)
            text = decode(out)
            if END in text:
                return text.split(END)[0], out, True
            if YOUR_TURN in text:            # it started writing the next user turn: stop here
                return text.split(YOUR_TURN)[0], out, False
        return text, out, False

    records = []
    for k, it in enumerate(items):
        t0 = time.time()
        sm.reset()                           # zeroed state per item
        gen = torch.Generator(device="cpu").manual_seed(SEED * 1000 + k)
        rec = {"id": it["id"], "type": it["type"], "replies": [], "positions": [], "atoms": []}
        for msg in it["turns"]:
            logits = etch(encode(prompt.format(msg)))
            text, atoms, ended = reply(logits, gen)
            if not ended:
                etch(encode(END))
            rec["replies"].append(text.strip()); rec["atoms"].append(atoms); rec["positions"].append(sm.pos)
        records.append(rec)
        print(f"[{k + 1}/{len(items)}] {it['id']} pos={sm.pos} {time.time() - t0:.0f}s | "
              f"{it['turns'][-1]} -> {rec['replies'][-1][:70]!r}", flush=True)
    return records


# ---------------- type L scoring (code only) ----------------
ARTICLES = {"a", "an", "the", "some", "and"}


def norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return " ".join(w for w in s.split() if w not in ARTICLES)


def parse_list(text):
    """List reply -> items. Take text after the last ':' (drops 'Here are three animals:'),
    split on commas/'and'/newlines/bullets, strip articles + punctuation, lowercase."""
    if len(re.findall(r"[.!?]\s", text)) > 0:                    # multi-sentence reply = a story, not a list
        return []
    body = text.split(":")[-1] if ":" in text else text
    body = re.sub(r"(?m)^\s*(\d+[.)]|[-*])\s*", ",", body)      # numbered / bulleted lists
    parts = re.split(r",|;|\n|\band\b|\bor\b", body)
    parts = [re.sub(r"[.!?]+$", "", p.strip()).lower() for p in parts]
    parts = [norm(p) for p in parts]
    return [p for p in parts if p and len(p.split()) <= 3]


def score_L(items, recs):
    c = {"n": 0, "list_ok": 0, "pos": {"first": [0, 0], "second": [0, 0], "last": [0, 0]}, "howmany": [0, 0],
         "wrong": 0, "wrong_named_last": 0}
    for it, r in zip(items, recs):
        if it["type"] != "L":
            continue
        c["n"] += 1
        lst = parse_list(r["replies"][0])
        r["parsed_list"] = lst
        if len(lst) < it["n"]:
            r["verdicts"] = ["no-list"] * len(it["qtypes"]); continue
        c["list_ok"] += 1
        lst = lst[:it["n"]]                  # the requested count, in order
        r["verdicts"] = []
        for q, ans in zip(it["qtypes"], r["replies"][1:]):
            a = norm(ans)
            if q == "howmany":
                ok = bool(re.search(rf"\b({NUM[it['n']]}|{it['n']})\b", a))
                c["howmany"][0] += ok; c["howmany"][1] += 1
            else:
                idx = {"first": 0, "second": 1, "last": -1}[q]
                target = lst[idx]
                others = [x for x in lst if x != target]
                has = lambda x: re.search(rf"\b{re.escape(x)}\b", a) is not None   # whole-word phrase match
                ok = has(target) and not any(has(o) for o in others)
                c["pos"][q][0] += ok; c["pos"][q][1] += 1
                if not ok and q != "last":   # recency: a wrong first/second answer that named the LAST item
                    c["wrong"] += 1; c["wrong_named_last"] += has(lst[-1])
            r["verdicts"].append("correct" if ok else "wrong")
    rate = lambda p: (p[0] / p[1]) if p[1] else None
    return {"L_items": c["n"], "L_list_format_rate": c["list_ok"] / max(c["n"], 1),
            "L_acc_first": rate(c["pos"]["first"]), "L_acc_second": rate(c["pos"]["second"]),
            "L_acc_last": rate(c["pos"]["last"]), "L_acc_howmany": rate(c["howmany"]),
            "L_recency_rate": (c["wrong_named_last"] / c["wrong"]) if c["wrong"] else None,
            "L_position_questions": sum(v[1] for v in c["pos"].values())}


# ---------------- type G scoring: code proxy + judge ----------------
STOP = set("the a an and or but of to in on at for with was were is are it its he she they his her their them "
           "him this that there then so very not did do does had has have be been what who where how when which "
           "one some all any into out up down about just from by as if".split())
words = lambda s: {w for w in re.findall(r"[a-z]+", s.lower()) if len(w) >= 3 and w not in STOP}

JUDGE_SYSTEM = """You judge whether an ANSWER to a QUESTION is supported by a STORY. Use ONLY the story text; no outside knowledge.
Output exactly one word:
SUPPORTED (the answer states something the story says), CONTRADICTED (the story says otherwise),
UNSUPPORTED (nothing in the story decides it), EVASIVE (not an answer to the question)."""


def judge_one(client, story, q, a):
    import anthropic
    for attempt in range(3):
        try:
            resp = client.messages.create(model=JUDGE_MODEL, max_tokens=4096, system=JUDGE_SYSTEM,
                                          messages=[{"role": "user", "content": f"STORY:\n{story}\n\nQUESTION: {q}\nANSWER: {a}"}])
            text = " ".join(b.text for b in resp.content if b.type == "text")
            m = re.search(r"\b(SUPPORTED|CONTRADICTED|UNSUPPORTED|EVASIVE)\b", text.upper())
            return m.group(1) if m else "UNPARSED"
        except anthropic.RateLimitError:
            time.sleep(15 * (attempt + 1))
    return "ERROR"


def score_G(items, recs, judge):
    n = 0; gw = []; lens = []; verd = {}; per_q = {}
    client = None
    if judge:
        import anthropic
        client = anthropic.Anthropic()
    for it, r in zip(items, recs):
        if it["type"] != "G":
            continue
        story = r["replies"][0]
        sw = words(story)
        r.setdefault("verdicts", [None] * len(it["qtypes"]))
        for j, (q, qt, ans) in enumerate(zip(it["turns"][1:], it["qtypes"], r["replies"][1:])):
            n += 1; lens.append(len(ans))
            cw = words(ans) - words(q)
            if cw:
                gw.append(len(cw & sw) / len(cw))
            if judge and r["verdicts"][j] in (None, "ERROR", "UNPARSED"):
                r["verdicts"][j] = judge_one(client, story, q, ans)
                print(f"  judge {it['id']} q{j + 1} [{qt}] {r['verdicts'][j]} | {ans[:60]!r}", flush=True)
            v = r["verdicts"][j]
            if v:
                verd[v] = verd.get(v, 0) + 1
                per_q.setdefault(qt, [0, 0]); per_q[qt][0] += v == "SUPPORTED"; per_q[qt][1] += 1
    tot = sum(verd.values())
    out = {"G_answers": n, "G_grounding_word_rate": sum(gw) / len(gw) if gw else None,
           "G_mean_reply_len": sum(lens) / len(lens) if lens else None, "G_judged": tot}
    for k in ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "EVASIVE"):
        out[f"G_{k.lower()}_rate"] = verd.get(k, 0) / tot if tot else None
    out["G_supported_by_qtype"] = {k: v[0] / v[1] for k, v in per_q.items()} if tot else {}
    return out


# ---------------- main ----------------
def table(scores):
    lines = [f"== {TAG} seed {SEED}{f' LIMIT={LIMIT}' if LIMIT else ''}  ({time.strftime('%Y-%m-%d %H:%M')})"]
    for k, v in scores.items():
        if isinstance(v, dict):
            v = "  ".join(f"{a}={b:.2f}" for a, b in v.items())
        elif isinstance(v, float):
            v = f"{v:.3f}"
        lines.append(f"  {k:26s} {v}")
    return "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    items = load_items()
    if LIMIT:                                 # first LIMIT of each type (smoke / judge trials)
        items = [it for it in items if it["type"] == "G"][:LIMIT] + [it for it in items if it["type"] == "L"][:LIMIT]
    replies_path = os.path.join(OUT, f"replies_{TAG}_s{SEED}.json")
    if os.environ.get("JUDGE_ONLY") == "1":   # re-score an existing replies file; never regenerate
        all_recs = json.load(open(replies_path))
        have = {r["id"]: r for r in all_recs}
        items = [it for it in items if it["id"] in have]
        recs = [have[it["id"]] for it in items]   # same dicts: verdicts land in all_recs too
    else:
        t0 = time.time()
        recs = all_recs = run_model(items)
        print(f"generation done in {(time.time() - t0) / 60:.1f} min", flush=True)
    judge = os.environ.get("JUDGE", "0") == "1"
    if judge:
        sys.path.insert(0, HERE); from judge_pairs import load_env; load_env()
    scores = {"tag": TAG, "seed": SEED, "items": len(items)}
    scores.update(score_L(items, recs))
    scores.update(score_G(items, recs, judge))
    json.dump(all_recs, open(replies_path, "w"), indent=1)
    suffix = f"_limit{LIMIT}" if LIMIT else ""     # partial runs never overwrite the full scores file
    json.dump(scores, open(os.path.join(OUT, f"scores_{TAG}_s{SEED}{suffix}.json"), "w"), indent=1)
    t = table(scores)
    print(t)
    with open(os.path.join(OUT, "results.txt"), "a") as f:
        f.write(t + "\n\n")


if __name__ == "__main__":
    main()
