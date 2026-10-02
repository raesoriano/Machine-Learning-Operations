"""Spoken-form tokenizer: written transcript -> word sequence the ASR hears.

The dataset transcripts are in *written* form ("Alarm 6:00 AM", "Brightness
100 percent", "Set the timer for one minute"). The acoustic model hears
*spoken* form, so every digit / time / number must be spelled out as words
before it is turned into a phone sequence.

This module is deterministic and dependency-free so the exact same mapping
runs at training time and (for the FSA) at build time.
"""
from __future__ import annotations
import re

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]
_SCALES = {100: "hundred", 1000: "thousand", 1000000: "million",
           1000000000: "billion"}


def _three_digits(n: int) -> list[str]:
    """0..999 -> words (no leading/trailing empty)."""
    out = []
    if n >= 100:
        out += [_ONES[n // 100], "hundred"]
        n %= 100
    if n >= 20:
        out.append(_TENS[n // 10])
        n %= 10
    if n > 0:
        out.append(_ONES[n])
    return out


def number_to_words(n: int) -> list[str]:
    """Non-negative integer -> spoken words. 0 -> [zero]; 100 -> [one, hundred]."""
    if n < 0:
        return ["minus"] + number_to_words(-n)
    if n == 0:
        return ["zero"]
    words = []
    # peel off scale units (billion, million, thousand) then the rest
    for scale in (1000000000, 1000000, 1000):
        if n >= scale:
            words += number_to_words(n // scale) + [_SCALES[scale]]
            n %= scale
    if n > 0:
        words += _three_digits(n)
    return words


_TIME_RE = re.compile(r"^(\d{1,2})[:.](\d{2})$")


def _token_time(tok: str) -> list[str]:
    m = _TIME_RE.match(tok)
    if not m:
        return [tok]
    h, mm = int(m.group(1)), int(m.group(2))
    if mm == 0:
        return number_to_words(h)
    return number_to_words(h) + number_to_words(mm)


_PUNCT = re.compile(r"[^0-9a-z\s:]+")


def tokenize_spoken(transcript: str) -> list[str]:
    """Written transcript -> lowercased spoken word list.

    - punctuation stripped
    - integers spelled out (10 -> ten, 100 -> one hundred)
    - clock times spelled out (6:00 -> six, 6:30 -> six thirty)
    - existing words (e.g. "one minute") pass through unchanged
    """
    if transcript is None:
        return []
    t = transcript.lower().strip()
    t = _PUNCT.sub(" ", t)
    words = []
    for tok in t.split():
        if tok.isdigit():
            words += number_to_words(int(tok))
        elif _TIME_RE.match(tok):
            words += _token_time(tok)
        else:
            words.append(tok)
    return words


def transcript_to_words(transcript: str) -> list[str]:
    return tokenize_spoken(transcript)
