"""Per-intent template grammars for dataset generation.

This is the collective dataset's text generator. Each template yields
``(text, intent, slots)``. The TTS stage (WP3) reads these, synthesizes
audio, and writes manifests. The same grammar is used to build the
benchmark's synthetic stand-in test set (see ``benchmark/synthetic_set.py``),
so the parser and the data always agree.

Usage:
    python -m data.templates.grammar --out manifest.jsonl --n 10000 --seed 0
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

# Allow running both as a module (python -m data.templates.grammar) and as a
# plain script (python data/templates/grammar.py) from the repo root.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vcm import slot_space as ss  # noqa: E402
from vcm.numbers import int_to_words  # noqa: E402


def _num(n):
    """Render a number the way a speaker would: digits most of the time,
    words sometimes (the model must handle both)."""
    return str(n) if random.random() < 0.55 else int_to_words(n)


def _pick(rng, seq):
    return rng.choice(list(seq))


# --------------------------------------------------------------------------- #
# Each generator: (rng) -> (text, intent, slots). `rng` is a random.Random.
# --------------------------------------------------------------------------- #
def gen_play_music(rng):
    loc = rng.random()
    if loc < 0.45:
        return "play music", "play_music", {}
    if loc < 0.75:
        g = _pick(rng, ss.MUSIC_GENRES)
        return f"play {g} music", "play_music", {"query": f"{g} music"}
    a = _pick(rng, ss.MUSIC_ARTISTS)
    # gold query = what the parser emits: lowercased, leading article stripped
    q = a.lower()
    q = re.sub(r"^(the|a|an)\s+", "", q)
    return f"play {a}", "play_music", {"query": q}


def gen_ask_question(rng):
    # (text, gold query type) — closed set, matches vcm.parser._QUESTION_PATTERNS.
    opts = [
        ("what's the weather", "weather"),
        ("what's the weather today", "weather"),
        ("what time is it", "time"),
        # "temperature outside" is a WEATHER question (no `temperature` query
        # in the 31-command taxonomy); the parser maps it to weather.
        ("what's the temperature outside", "weather"),
        ("who won the game yesterday", "sports"),
        ("how far is the nearest gas station", "distance"),
        ("what day is it today", "date"),
    ]
    t, q = _pick(rng, opts)
    return t, "ask_question", {"query": q}


def gen_lights_switch(rng):
    state = _pick(rng, ["on", "off"])
    loc = _pick(rng, ss.LOCATIONS) if rng.random() < 0.5 else None
    if loc:
        text = f"turn {state} the {loc} lights"
    else:
        text = f"turn {state} the lights"
    slots = {"state": state}
    if loc:
        slots["location"] = loc
    return text, "lights_switch", slots


def gen_lights_adjust(rng):
    pct = _pick(rng, ss.PERCENTS)
    loc = _pick(rng, ss.LOCATIONS) if rng.random() < 0.3 else None
    where = f" {loc} " if loc else " "
    # NOTE: color and percent are mutually exclusive in the grammar — the
    # parser extracts color only from "set ... to <color>" forms, so a dim
    # sentence must not carry a color slot (gold must match parser output).
    if rng.random() < 0.35:
        color = _pick(rng, ss.COLORS)
        text = f"set the{where}lights to {color}"
        slots = {"color": color}
    else:
        text = f"dim the{where}lights to {_num(pct)} percent"
        slots = {"percent": pct}
    if loc:
        slots["location"] = loc
    return text, "lights_adjust", slots


def gen_set_timer(rng):
    # gold duration format = the parser's canonical output ("1m", "10s", "1h").
    if rng.random() < 0.8:
        m = _pick(rng, ss.DURATION_MINUTES)
        text = f"set a timer for {_num(m)} minutes"
        slots = {"duration": f"{m}m"}
    else:
        h = _pick(rng, ss.DURATION_HOURS)
        text = f"set a timer for {_num(h)} hours"
        slots = {"duration": f"{h}h"}
    return text, "set_timer", slots


def gen_set_alarm(rng):
    h = _pick(rng, ss.CLOCK_HOURS)
    mer = _pick(rng, ss.MERIDIEM)
    text = f"set an alarm for {_num(h)} {mer}"
    # gold time format = the parser's canonical output ("6:00 AM").
    return text, "set_alarm", {"time": f"{h}:00 {mer.upper()}"}


def gen_set_temperature(rng):
    fah = rng.random() < 0.5
    v = _pick(rng, ss.TEMP_F if fah else ss.TEMP_C)
    unit = "F" if fah else "C"
    word = "fahrenheit" if fah else "celsius"
    text = f"set the temperature to {_num(v)} degrees {word}"
    return text, "set_temperature", {"value": v, "unit": unit}


def gen_media_control(rng):
    opts = [
        ("pause", {"action": "pause"}),
        ("stop the music", {"action": "stop"}),
        ("next song", {"action": "next"}),
        ("skip to the next track", {"action": "next"}),
        ("previous song", {"action": "previous"}),
        ("volume up", {"action": "volume_up"}),
        ("turn the volume down", {"action": "volume_down"}),
        ("make it louder", {"action": "volume_up"}),
        ("make it quieter", {"action": "volume_down"}),
        ("mute", {"action": "mute"}),
    ]
    t, s = _pick(rng, opts)
    return t, "media_control", s


def gen_reminders_lists(rng):
    if rng.random() < 0.7:
        a = _pick(rng, ss.REMINDER_ACTIONS)
        return f"remind me to {a}", "reminders_lists", {"action": a}
    opts = [
        ("what are my reminders", {"action": "list"}),
        ("show my reminders", {"action": "list"}),
        ("what's on my to do list", {"action": "list"}),
    ]
    t, s = _pick(rng, opts)
    return t, "reminders_lists", s


def gen_call(rng):
    c = _pick(rng, ss.CONTACTS)
    text = f"call {c}" if rng.random() < 0.8 else f"phone {c}"
    return text, "call", {"contact": c}


GENERATORS = {
    "play_music": gen_play_music,
    "ask_question": gen_ask_question,
    "lights_switch": gen_lights_switch,
    "lights_adjust": gen_lights_adjust,
    "set_timer": gen_set_timer,
    "set_alarm": gen_set_alarm,
    "set_temperature": gen_set_temperature,
    "media_control": gen_media_control,
    "reminders_lists": gen_reminders_lists,
    "call": gen_call,
}

# Out-of-vocabulary / rejection utterances (the model must NOT fire on these).
# NOTE: keep this list in sync with the parser — every entry here must parse
# to intent == unknown (tools/check_grammar.py enforces it).
OOV_UTTERANCES = [
    "hello", "good morning", "thank you", "i love you", "who are you",
    "tell me a joke", "what is the meaning of life", "open the browser",
    "send an email to john", "how are you feeling",
    "i want to order pizza", "book a flight to manila", "read me a book",
    "what is two plus two", "turn on the fan",
    "set a calendar event", "lock the front door", "who is the president",
    "tell me about the stock market", "play a game", "take a photo",
    "what is the capital of france", "turn on the television",
    "wake me up gently", "i am hungry", "where is my car",
]


def generate(n, seed=0, oov_ratio=0.05):
    """Yield n rows of {id, text, intent, slots} (balanced across intents)."""
    rng = random.Random(seed)
    intents = list(GENERATORS)
    rows = []
    n_oov = int(n * oov_ratio)
    n_cmd = n - n_oov
    per = n_cmd // len(intents)
    for i, intent in enumerate(intents):
        for _ in range(per + (1 if i < n_cmd % len(intents) else 0)):
            text, intent, slots = GENERATORS[intent](rng)
            rows.append({"text": text, "intent": intent, "slots": slots})
    for _ in range(n_oov):
        text = _pick(rng, OOV_UTTERANCES)
        rows.append({"text": text, "intent": "unknown", "slots": {}})
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r = {"id": f"gen_{seed}_{i:06d}", **r}
        yield r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oov-ratio", type=float, default=0.05)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in generate(args.n, args.seed, args.oov_ratio):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {args.n} rows -> {out}")


if __name__ == "__main__":
    main()
