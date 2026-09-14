# Three-judge blinded evaluation of pairs2.json. Judges run IN PARALLEL
# (one worker per provider); pairs go sequentially within each judge.
# API keys from the environment or a .env file next to this script.
#   SUF= ARMS=a,b JUDGES=anthropic,openai,deepseek python -u judge_pairs.py
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- config ----
ANTHROPIC_MODEL = "claude-opus-5"          
OPENAI_MODEL = "gpt-5.6-sol"               # reasoning effort: medium
DEEPSEEK_MODEL = "deepseek-v4-pro"         # provider default

RUBRIC = """You are judging two AI assistants' responses to the same user prompt.
Both assistants are very small experimental models, so judge them RELATIVE to
each other, not against normal chatbot standards.

Compare on these axes:
- coherence: which response makes more sense as connected text?
- instruction: which better does what the prompt asked?
- fluency: which has better spelling, grammar, and natural wording?
- overall: considering everything, which is the better response?

For each axis answer "A", "B", or "tie".
Reply with ONLY a JSON object, no other text:
{"coherence": "...", "instruction": "...", "fluency": "...", "overall": "...", "reason": "<15 words max>"}"""


def load_env():
    if not os.path.exists(os.path.join(HERE, ".env")):
        return
    with open(os.path.join(HERE, ".env")) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def item_prompt(p):
    return (f"USER PROMPT:\n{p['prompt']}\n\n"
            f"RESPONSE A:\n{p['A']}\n\nRESPONSE B:\n{p['B']}")


def parse_ruling(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        r = json.loads(m.group(0))
        return {k: str(r.get(k, "tie")).strip().upper()[:3].rstrip(".").replace("TIE", "tie")
                for k in ("coherence", "instruction", "fluency", "overall")} | \
               {"reason": str(r.get("reason", ""))[:120]}
    except json.JSONDecodeError:
        return None


def judge_anthropic(pairs):
    import anthropic
    client = anthropic.Anthropic()
    out = {}
    for p in pairs:
        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model=ANTHROPIC_MODEL, max_tokens=16000,
                    system=RUBRIC,
                    messages=[{"role": "user", "content": item_prompt(p)}],
                )
                text = next(b.text for b in resp.content if b.type == "text")
                r = parse_ruling(text)
                if r:
                    out[p["id"]] = r
                break
            except anthropic.RateLimitError:
                time.sleep(15 * (attempt + 1))
            except Exception as e:
                print(f"[anthropic] {p['id']}: {e}", flush=True)
                break
    return "claude-opus-5", out


def _openai_style(pairs, base_url, api_key_env, model, tag, extra):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ[api_key_env], base_url=base_url)
    out = {}
    for p in pairs:
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": RUBRIC},
                              {"role": "user", "content": item_prompt(p)}],
                    **extra)
                r = parse_ruling(resp.choices[0].message.content or "")
                if r:
                    out[p["id"]] = r
                break
            except Exception as e:
                if "rate" in str(e).lower() and attempt < 2:
                    time.sleep(15 * (attempt + 1))
                    continue
                print(f"[{tag}] {p['id']}: {e}", flush=True)
                break
    return tag, out


def judge_openai(pairs):
    return _openai_style(pairs, None, "OPENAI_API_KEY", OPENAI_MODEL,
                         "gpt-5.6-sol", {"reasoning_effort": "medium"})


def judge_deepseek(pairs):
    return _openai_style(pairs, "https://api.deepseek.com", "DEEPSEEK_API_KEY",
                         DEEPSEEK_MODEL, "deepseek-v4-pro", {})


def main():
    load_env()
    # SUF selects the pair set (default "2" = 57M bytes-vs-bpe exam); ARMS names the two sides
    # as they appear in the key file, e.g. SUF=_100m ARMS=v1,v2
    SUF = os.environ.get("SUF", "2")
    ARM_A, ARM_B = os.environ.get("ARMS", "bytes,bpe").split(",")
    with open(os.path.join(HERE, f"pairs{SUF}.json")) as f:
        pairs = json.load(f)
    with open(os.path.join(HERE, f"key{SUF}.json")) as f:
        key = json.load(f)
    # JUDGES selects the panel (default all three); OUT_SUF renames the outputs so a
    # two-judge rerun does not clobber the three-judge files.
    _FNS = {"anthropic": judge_anthropic, "openai": judge_openai, "deepseek": judge_deepseek}
    JUDGES = [j.strip() for j in os.environ.get("JUDGES", "anthropic,openai,deepseek").split(",") if j.strip()]
    sel = [_FNS[j] for j in JUDGES]
    OUT = os.environ.get("OUT_SUF", SUF)
    print(f"judging {len(pairs)} pairs with {len(sel)} judges ({', '.join(JUDGES)}) in parallel -> {OUT}", flush=True)

    with ThreadPoolExecutor(max_workers=len(sel)) as ex:
        results = dict(ex.map(lambda fn: fn(pairs), sel))
    with open(os.path.join(HERE, f"judge_rulings{OUT}.json"), "w") as f:
        json.dump(results, f, indent=1)

    # ---- tally ----
    lines = [f"=== LLM-judge panel: {ARM_A} vs {ARM_B} (blinded, order-randomized); "
             f"judges: {', '.join(JUDGES)} ==="]
    per_judge_overall = {}
    for judge, rulings in results.items():
        wins = {ARM_A: 0, ARM_B: 0, "tie": 0}
        for pid, r in rulings.items():
            v = r["overall"]
            wins[key[pid][v] if v in ("A", "B") else "tie"] += 1
        n = max(sum(wins.values()), 1)
        per_judge_overall[judge] = {
            pid: (key[pid][r["overall"]] if r["overall"] in ("A", "B") else "tie")
            for pid, r in rulings.items()}
        lines.append(f"{judge:16s}: {ARM_B} {wins[ARM_B]}/{n}  {ARM_A} {wins[ARM_A]}/{n}"
                     f"  tie {wins['tie']}/{n}  ({len(rulings)} ruled)")

    # majority vote per item
    maj = {ARM_A: 0, ARM_B: 0, "none": 0}
    ids = [p["id"] for p in pairs]
    for pid in ids:
        votes = [per_judge_overall[j].get(pid) for j in results if pid in per_judge_overall[j]]
        for side in (ARM_A, ARM_B):
            if votes.count(side) >= 2:
                maj[side] += 1
                break
        else:
            maj["none"] += 1
    lines.append(f"majority (2 of {len(results)}): {ARM_B} {maj[ARM_B]}  {ARM_A} {maj[ARM_A]}"
                 f"  no majority {maj['none']}")

    # pairwise agreement on overall
    judges = list(results)
    for i in range(len(judges)):
        for j in range(i + 1, len(judges)):
            a, b = per_judge_overall[judges[i]], per_judge_overall[judges[j]]
            common = [pid for pid in ids if pid in a and pid in b]
            agree = sum(a[pid] == b[pid] for pid in common)
            if common:
                lines.append(f"agreement {judges[i]} vs {judges[j]}: "
                             f"{agree}/{len(common)} ({100*agree/len(common):.0f}%)")

    report = "\n".join(lines)
    print(report, flush=True)
    with open(os.path.join(HERE, f"judge_results{OUT}.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
