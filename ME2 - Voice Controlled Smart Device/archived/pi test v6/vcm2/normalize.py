"""Shared text normalization for VCM v2 (SELF-CONTAINED).

The single most important consistency rule in the whole pipeline:

  * the ME2 manifest `transcript` column uses ARABIC DIGITS  ("Alarm 6 AM")
  * the ASR outputs NUMBER-WORDS                             ("alarm six am")

Slot commands are distinguished ONLY by that number (ALARM_6_00AM vs
ALARM_8_00AM, TEMPERATURE_18 vs _22, TIMER_10s vs _30s ...), so the two sides
MUST be brought to one format before the classifier sees them. We normalize
everything to NUMBER-WORDS (the ASR's native output), so:

    normalize("Alarm 6 AM")   -> "alarm six am"
    normalize("alarm six am") -> "alarm six am"     (ASR output, unchanged)

The two helpers below (`_base_norm`, `int_to_words`) are inlined from the
Voice-Controlled-Smart-Device `vcm` package so this folder runs standalone on
the Pi (where only Machine-Learning-Operations is checked out).
"""
import re

# --- inlined from vcm/parser.py::_normalize --------------------------------
def _base_norm(text: str) -> str:
    t = text.lower().strip()
    t = t.replace("\u2019", "'")
    t = t.replace("what's", "whats")          # contractions before stripping
    t = re.sub(r"[^\w\s%]", " ", t)          # keep alnum, space, %
    t = re.sub(r"\s+", " ", t).strip()
    return t


# --- inlined from vcm/numbers.py (0-120 -> words) --------------------------
_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_ONES_REV = {v: k for k, v in _ONES.items()}
_TENS_REV = {v: k for k, v in _TENS.items()}


def int_to_words(n: int) -> str:
    """0-120 -> English words (used for the vocab + TTS templates)."""
    if not 0 <= n <= 120:
        raise ValueError(f"out of supported range 0-120: {n}")
    if n < 20:
        return _ONES_REV[n]
    if n < 100:
        t, r = divmod(n, 10)
        return _TENS_REV[t * 10] + (f" {_ONES_REV[r]}" if r else "")
    h, r = divmod(n, 100)
    s = f"{_ONES_REV[h]} hundred"
    if r:
        s += f" {int_to_words(r)}"
    return s


def _one(text: str) -> str:
    t = _base_norm(text or "")
    toks = []
    for tok in t.split():
        if tok.isdigit() and 0 <= int(tok) <= 120:
            toks.extend(int_to_words(int(tok)).split())
        else:
            toks.append(tok)
    return " ".join(toks)


def normalize(text) -> str:
    """Lowercase, strip punctuation, expand digits (0-120) to number-words.

    Accepts a single string or an array-like of strings (the latter is what
    sklearn's FunctionTransformer passes).
    """
    if isinstance(text, str):
        return _one(text)
    return [_one(x) for x in text]


if __name__ == "__main__":
    for s in ["Alarm 6 AM", "alarm six am", "set the temperature to 22 degrees",
              "brightness 100 percent", "timer 10 seconds", "what's the weather",
              "change color to blue", "create a reminder to drink water"]:
        print(f"  {s!r:45s} -> {normalize(s)!r}")
