"""JSGF grammar parsing: extract the flat set of command phrases + the word
inventory. Robust to the `phrase |` line layout and empty tokens."""
from __future__ import annotations
import re


def parse_jsgf(path: str) -> tuple[list[str], list[str]]:
    """Returns (phrases, words). phrases are the complete command strings."""
    js = open(path).read()
    # take the body of the <command> rule (inside the outer parens)
    m = re.search(r"=\s*\((.*)\)\s*;", js, re.DOTALL)
    body = m.group(1) if m else js
    phrases = []
    for tok in body.split("|"):
        p = " ".join(tok.split())          # collapse whitespace
        if p:
            phrases.append(p)
    # de-duplicate, preserve order
    seen = set()
    uniq = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    words = sorted({w for p in uniq for w in p.split()})
    return uniq, words
