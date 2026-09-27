#!/usr/bin/env python3
"""Build the expanded VCM vocab from the ME2 transcripts (Plan A).

Rule (principled, avoids one-off overfit words):
  * start from the current base vocab (intent words + slot values + number
    words 0-120),
  * add every OOV word that appears in the val OR test split (we must be able
    to decode it at inference),
  * add every OOV word with total frequency >= --min-count (common paraphrase
    words seen enough to learn),
  * digits are NOT added: training targets are normalized to number-words
    (the audio says "twenty-two", not "22"), and the parser already accepts
    both forms.

Saves the sorted word list to vcm/vocab_words.json and reports size + coverage.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from vcm.vocab import VOCAB as BASE  # noqa: E402
from vcm.parser import _normalize  # noqa: E402


def tokenize(text):
    """Same normalization the training tokenizer uses (vcm.parser._normalize)."""
    return _normalize(text).split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(_REPO, "vcm",
                                                  "vocab_words.json"))
    args = ap.parse_args()

    with open(args.manifest) as f:
        rows = [r for r in csv.DictReader(f) if r["polarity"] == "positive"]

    freq = Counter()
    in_valtest = set()
    row_toks = []
    for r in rows:
        toks = tokenize(r["transcript"])
        row_toks.append(toks)
        for t in toks:
            freq[t] += 1
            if r["split"] in ("val", "test"):
                in_valtest.add(t)

    base = set(BASE)
    added = set()
    for w, c in freq.items():
        if w in base or w.isdigit():
            continue
        if c >= args.min_count:
            added.add(w)

    vocab = sorted(base | added)
    vset = set(vocab)

    total_words = sum(freq.values())
    nw = sum(c for w, c in freq.items() if w in vset)
    nrow = sum(1 for toks in row_toks
               if toks and all(t in vset for t in toks))
    print(f"base vocab:            {len(base)}")
    print(f"+ OOV freq>={args.min_count} (all splits): {len(added)}")
    print(f"= TOTAL vocab:         {len(vocab)}")
    print(f"word coverage:         {100*nw/total_words:.1f}%")
    print(f"row  coverage:         {100*nrow/len(row_toks):.1f}%")
    # what's still OOV
    still = Counter({w: c for w, c in freq.items() if w not in vset})
    print(f"still-OOV types:       {len(still)}  "
          f"({100*sum(still.values())/total_words:.2f}% of words)")

    with open(args.out, "w") as f:
        json.dump(vocab, f, indent=0)
    print(f"\nsaved {len(vocab)} words -> {args.out}")


if __name__ == "__main__":
    main()
