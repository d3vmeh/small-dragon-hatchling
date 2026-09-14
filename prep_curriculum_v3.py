# Builds the upgraded tuning set from curriculum_v3.json + a story subset.
# Output: curriculum_v3_examples.pkl  [(utf8_bytes, [(start, end), ...]), ...]
# Masks are SPANS (one per assistant reply, marker included) so multi-turn
# documents train only on assistant bytes, never on the user's next question.
import json
import os
import pickle
import random

HERE = os.path.dirname(os.path.abspath(__file__))
END = "<|endoftext|>"
rng = random.Random(42)

with open(os.path.join(HERE, "curriculum_v3.json")) as f:
    docs = json.load(f)
with open(os.environ.get("STORIES", os.path.join(HERE, "chat_examples.pkl")), "rb") as f:
    stories = pickle.load(f)                      # single-turn story documents as (bytes, mask_from)


def build(turns):
    text, spans = "", []
    for t in turns:
        text += f"User: {t['user']}\nAssistant: "
        s = len(text.encode("utf-8"))
        text += t["assistant"] + END
        spans.append((s, len(text.encode("utf-8"))))
    return text.encode("utf-8"), spans


examples = []
for d in docs:
    b, spans = build(d["turns"])
    if len(b) <= 512:
        examples.append((b, spans))
# stories keep a single span; subsample so the dialogues are not drowned out
story_sub = [(b, [(m, len(b))]) for b, m in rng.sample(stories, 3000)]
mixed = examples * 2 + story_sub          # ~4k curriculum + 3k stories
rng.shuffle(mixed)
with open(os.path.join(HERE, "curriculum_v3_examples.pkl"), "wb") as f:
    pickle.dump(mixed, f)
print(f"{len(examples)} curriculum docs (x3) + {len(story_sub)} stories -> {len(mixed)} examples")
print("sample:", examples[0][0].decode()[:200], examples[0][1])
