"""Phonetic dictionary: word -> phoneme (ARPAbet) sequence.

Built from the CMU pronunciation dictionary (used only as a *resource* at build
time -- no PocketSphinx code runs at runtime). Every word in the command
vocabulary is mapped to its ARPAbet phone sequence. Silence is NOT baked into
the words; it is a dedicated SIL phone in the acoustic model (trained on real
silence) and is handled at word boundaries by the decoder.

Format on disk (dictionary.txt): one line per word
    word  PH1 PH2 ...
"""
from __future__ import annotations
import numpy as np

CMU_PATH = ("/home/ron.andrei.soriano/.local/lib/python3.13/site-packages/"
            "pocketsphinx/model/en-us/cmudict-en-us.dict")

SIL = "SIL"


def _norm_word(w: str) -> str:
    w = w.lower().strip()
    if w == "what's":
        w = "whats"
    return w


def load_cmudict(path: str = CMU_PATH) -> dict[str, list[str]]:
    cmu: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = line.split()
            w = parts[0].lower()
            if w not in cmu:
                cmu[w] = parts[1:]
    return cmu


def build_dictionary(words: list[str], path: str = CMU_PATH) -> dict[str, list[str]]:
    """word -> [ph1, ..., phN]. Raises if any word is missing."""
    cmu = load_cmudict(path)
    d: dict[str, list[str]] = {}
    missing = []
    for w in words:
        k = _norm_word(w)
        if k not in cmu:
            missing.append(w)
            continue
        d[w] = cmu[k]
    if missing:
        raise KeyError(f"words missing from cmudict: {missing}")
    return d


def phones_in_dict(d: dict[str, list[str]]) -> list[str]:
    """Sorted unique phone inventory (SIL first)."""
    ph = set()
    for seq in d.values():
        ph.update(seq)
    return [SIL] + sorted(ph - {SIL})


def save(d: dict[str, list[str]], path: str):
    with open(path, "w") as f:
        for w in sorted(d):
            f.write(w + "  " + " ".join(d[w]) + "\n")


def load(path: str) -> dict[str, list[str]]:
    d: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                d[parts[0]] = parts[1:]
    return d
