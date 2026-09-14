# Curriculum v3:
#   grounded  - user request is GENERIC; the story INVENTS specifics (name, object,
#               color, number, place) that appear nowhere in the request; follow-up
#               questions ask for exactly those specifics; answers come only from the story.
#   ordinal   - "name three X" -> a list; then first/second/last/how-many questions
#               answered correctly by position (the model has to use position).
# Reuses the micro/constraint/stop documents of the v2 curriculum (curriculum.json, made the same way). Output: curriculum_v3.json
#   python -u make_curriculum_v3.py   (API key from the environment or a .env file next to this script)
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import anthropic

HERE = os.path.dirname(os.path.abspath(__file__))


def load_env():
    if not os.path.exists(os.path.join(HERE, ".env")):
        return
    with open(os.path.join(HERE, ".env")) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
client = anthropic.Anthropic()

SYSTEM = """You write training dialogues for a VERY small language model (100M
params) that only knows simple English: the vocabulary and style of TinyStories
(children's stories: short words, short sentences, names like Tim, Lily, Sue).

Output ONLY a JSON array of dialogues. Each dialogue is
{"turns": [{"user": "...", "assistant": "..."}, ...]}. No other text.
Hard rules:
- Simple words only. No facts beyond a young child's world.
- The WHOLE dialogue (all turns) must be under 430 characters total.
- Assistant replies never mention being an AI, never apologize, never ramble."""

SPECS = {
    "grounded": (40, """Generate 25 dialogues of exactly 3 turns about {topic}.
turn 1: the user asks for a story in a GENERIC way that names NO specifics
("tell me a story about a girl in a garden", "write a story about a boy and
his pet", "a story about finding something"). The assistant writes a 3-5
sentence story that INVENTS specifics: a character NAME, a concrete OBJECT,
a COLOR, a NUMBER or a PLACE, none of which appear in the user's request.
turns 2-3: the user asks about exactly those invented specifics ("what was
her name?", "what did he find?", "what color was the box?", "how many
ducks were there?", "where did they go?"). The assistant answers in ONE
short sentence using ONLY what the story said. CRITICAL: the answer must
NOT be guessable from the question or the request, only from the story.
Keep every story under 260 characters so the dialogue stays under 430."""),
    "ordinal": (16, """Generate 30 dialogues of exactly 2 or 3 turns about {topic}.
turn 1: the user asks for a short list ("name three animals", "name four
colors", "tell me three things in a kitchen", "list two fruits"); the
assistant answers with a comma-separated list in a specific order.
turns 2-3: the user asks a POSITION question about that list ("what was the
first one?", "which did you say second?", "what was the last one you named?",
"how many did you name?", "which came before X?") and the assistant answers
correctly BY POSITION in one short sentence. Vary the lists so the same
items are not always used; vary which position is asked. Never answer a
"first" question with the last item."""),
}
TOPICS = ["animals", "toys", "the park", "food and cooking", "family and friends",
          "the sea and boats", "space and stars", "school", "weather and seasons",
          "vehicles", "birthdays and parties", "gardens and plants", "pets",
          "the forest", "colors and shapes", "night and dreams", "helping others",
          "bugs and butterflies", "a farm", "a castle", "a beach day", "snow",
          "magic", "a lost thing", "sharing", "being brave", "a new friend",
          "music", "a dragon", "a robot", "a picnic", "a race", "a treasure",
          "a rainy day", "grandparents", "a kitten", "a river", "a market",
          "a tree house", "a rainy walk"]


def one_call(cat, topic):
    _, prompt = SPECS[cat]
    try:
        resp = client.messages.create(
            model="claude-opus-5", max_tokens=16000, system=SYSTEM,
            messages=[{"role": "user", "content": prompt.format(topic=topic)}])
        text = next(b.text for b in resp.content if b.type == "text")
        m = re.search(r"\[.*\]", text, re.DOTALL)
        batch = json.loads(m.group(0)) if m else []
    except Exception as e:  # noqa: BLE001
        print(f"[{cat}/{topic}] failed: {e}", flush=True)
        return [], (0, 0)
    good = []
    for d in batch:
        turns = d.get("turns") if isinstance(d, dict) else None
        if not turns or not all(isinstance(t, dict) and t.get("user") and t.get("assistant") for t in turns):
            continue
        if sum(len(t["user"]) + len(t["assistant"]) for t in turns) > 440:
            continue
        good.append({"category": cat, "turns": [{"user": t["user"].strip(), "assistant": t["assistant"].strip()} for t in turns]})
    print(f"[{cat}/{topic}] +{len(good)}", flush=True)
    return good, (resp.usage.input_tokens, resp.usage.output_tokens)


jobs = [(cat, TOPICS[i % len(TOPICS)]) for cat, (n, _) in SPECS.items() for i in range(n)]
docs, tin, tout = [], 0, 0
with ThreadPoolExecutor(max_workers=6) as ex:
    for good, (i, o) in ex.map(lambda j: one_call(*j), jobs):
        docs.extend(good); tin += i; tout += o

# carry over v2's non-story categories unchanged
v2 = json.load(open(os.path.join(HERE, "curriculum.json")))
carry = [d for d in v2 if d["category"] in ("micro", "constraint", "stop")]
docs.extend(carry)
json.dump(docs, open(os.path.join(HERE, "curriculum_v3.json"), "w"), indent=1)
print("counts:", dict(Counter(d["category"] for d in docs)), flush=True)
print(f"wrote {len(docs)} dialogues -> curriculum_v3.json", flush=True)
print(f"usage: {tin} in / {tout} out tokens (~${tin*5/1e6 + tout*25/1e6:.2f} at Opus 5 rates)", flush=True)
