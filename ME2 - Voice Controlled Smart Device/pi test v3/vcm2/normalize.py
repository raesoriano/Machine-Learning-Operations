"""Shared text normalization for VCM v2.

The single most important consistency rule in the whole pipeline:

  * the ME2 manifest `transcript` column uses ARABIC DIGITS  ("Alarm 6 AM")
  * the ASR (me2_v6) outputs NUMBER-WORDS                     ("alarm six am")

Slot commands are distinguished ONLY by that number (ALARM_6_00AM vs
ALARM_8_00AM, TEMPERATURE_18 vs _22, TIMER_10s vs _30s ...), so the two sides
MUST be brought to one format before the classifier sees them. We normalize
everything to NUMBER-WORDS (the ASR's native output), so:

    normalize("Alarm 6 AM")   -> "alarm six am"
    normalize("alarm six am") -> "alarm six am"     (ASR output, unchanged)

Reuses vcm.parser._normalize (lowercase, contractions, strip punctuation) and
vcm.numbers.int_to_words (digit -> words), exactly mirroring the training
tokenizer in model/train.py so train/serve text can never drift.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../VCM-v2/vcm2
_SANDBOX = os.path.dirname(os.path.dirname(_HERE))          # .../sandbox
_ME2 = os.path.join(_SANDBOX, "Machine-Learning-Operations",
                    "ME2 - Voice Controlled Smart Device")
if _ME2 not in sys.path:
    sys.path.insert(0, _ME2)

from vcm.parser import _normalize as _base_norm      # noqa: E402
from vcm.numbers import int_to_words                 # noqa: E402


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
