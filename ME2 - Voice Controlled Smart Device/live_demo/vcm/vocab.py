"""Vocabulary for the VCM (Plan A: data-driven, transcript-based).

The vocab is loaded from ``vocab_words.json`` — built by
``scripts/build_vocab.py`` from the actual ME2 transcripts:

    base 185 words (intent words + slot values + number words 0-120)
  + every transcript word (after the training tokenizer's normalization,
    digits excluded) that appears >= 5 times in the positive rows.

This gives ~1033 words covering 99.3% of transcript *words* (97.7% of rows
fully in-vocab after digit->number-word normalization), versus 34.4% of rows
for the old 185-word set. The model is trained on the transcript (what the
audio says), so the vocab must cover the spoken words — including the
paraphrase triggers the old set dropped (switch, brightness, wake, countdown,
loud, lower, color, create, start, increase, ...).

Digits are NOT in the vocab: training targets are normalized to number-words
(the audio says "twenty-two", not "22"), and the parser accepts both forms.

One source of truth: the model head, the constrained decoder, and the parser
all index into this list.
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORDS_PATH = os.path.join(_HERE, "vocab_words.json")


def _load_words():
    if os.path.exists(_WORDS_PATH):
        with open(_WORDS_PATH) as f:
            return json.load(f)
    # Fallback: the original closed-set vocab (pre-Plan-A).
    from . import slot_space as ss
    from .numbers import number_words
    intent_words = {
        "play", "music", "turn", "on", "off", "dim", "set", "timer", "alarm",
        "temperature", "thermostat", "degrees", "degree", "pause", "stop",
        "next", "skip", "volume", "up", "down", "louder", "quieter", "mute",
        "remind", "me", "to", "call", "phone", "ring", "message", "send",
        "reminder", "reminders", "what", "whats", "time", "weather", "is",
        "it", "the", "a", "an", "my", "are", "lights", "light", "for", "in",
        "and", "please",
    }
    slot_value_words = (
        ss.phrases_to_words(ss.LOCATIONS)
        | ss.phrases_to_words(ss.COLORS)
        | ss.phrases_to_words(ss.CONTACTS)
        | ss.phrases_to_words(ss.MUSIC_GENRES)
        | ss.phrases_to_words(ss.MUSIC_ARTISTS)
        | ss.phrases_to_words(ss.REMINDER_ACTIONS)
        | {"percent", "minutes", "minute", "seconds", "second", "hours",
           "hour", "am", "pm", "degrees", "degree", "celsius", "fahrenheit"}
    )
    return sorted(intent_words | slot_value_words | number_words(120))


VOCAB = _load_words()
WORD2ID = {w: i for i, w in enumerate(VOCAB)}
ID2WORD = {i: w for i, w in enumerate(VOCAB)}


def to_ids(words):
    """[str] -> [int] using the vocab (unknown words -> -1)."""
    return [WORD2ID.get(w, -1) for w in words]


def from_ids(ids):
    """[int] -> [str] (drops -1)."""
    return [ID2WORD[i] for i in ids if i >= 0]


if __name__ == "__main__":
    print(f"vocab size: {len(VOCAB)}  (from {os.path.basename(_WORDS_PATH)})")
    print("sample:", VOCAB[:20])
