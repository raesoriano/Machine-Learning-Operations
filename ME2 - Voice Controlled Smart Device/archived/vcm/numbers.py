"""English number-word parsing for the closed slot spaces.

Covers exactly the ranges the VCM needs:
  percent 0-100, duration 1-120 min / 1-24 h, clock hours 1-12,
  temperature 16-30 C / 60-85 F.
Pure Python, no dependencies.
"""

ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
TEENS = dict(ONES)  # 10-19 handled in ONES
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_ONES_REV = {v: k for k, v in ONES.items()}
_TENS_REV = {v: k for k, v in TENS.items()}


def words_to_int(text: str):
    """'forty' -> 40, 'twenty five' -> 25, 'one hundred twenty' -> 120,
    'a hundred' -> 100. Returns None if not a parseable number."""
    tokens = text.lower().replace("-", " ").split()
    if not tokens:
        return None
    current = 0
    for t in tokens:
        if t in ("a", "an"):
            current += 1
        elif t == "and":
            continue
        elif t in ONES:
            current += ONES[t]
        elif t in TENS:
            current += TENS[t]
        elif t == "hundred":
            current = (current or 1) * 100
        else:
            return None
    return current


def int_to_words(n: int) -> str:
    """0-120 -> English words (used for TTS templates and the vocab)."""
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


def number_words(max_n: int = 120):
    """Set of all individual words appearing in 0..max_n (for the vocab)."""
    words = set()
    for n in range(0, max_n + 1):
        words.update(int_to_words(n).split())
    return words
