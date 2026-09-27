#!/usr/bin/env python3
"""Analyze transcript word coverage for the VCM vocab (Plan A).

Reads the positive rows of the ME2 manifest, tokenizes the *transcript*
column (what the audio actually says), and reports:
  * word frequencies (train/val/test)
  * OOV against the current 185-word vocab
  * proposed expanded vocab (top-N OOV words + numbers + function words)

Usage:
  python3 scripts/analyze_vocab.py \
      --manifest data/manifests/positive_negative_manifest.csv
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

from vcm.vocab import VOCAB  # noqa: E402


def tokenize(text):
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return [w for w in text.split() if w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=400,
                    help="report N most frequent OOV words")
    args = ap.parse_args()

    with open(args.manifest) as f:
        rows = [r for r in csv.DictReader(f) if r["polarity"] == "positive"]

    cur = set(VOCAB)
    freq = Counter()
    freq_split = {"train": Counter(), "val": Counter(), "test": Counter()}
    n_in, n_out = 0, 0
    fully_in = 0
    total_words = 0
    for r in rows:
        toks = tokenize(r["transcript"])
        if not toks:
            continue
        total_words += len(toks)
        if all(t in cur for t in toks):
            fully_in += 1
        for t in toks:
            freq[t] += 1
            freq_split[r["split"]][t] += 1
            if t in cur:
                n_in += 1
            else:
                n_out += 1

    oov = Counter({w: c for w, c in freq.items() if w not in cur})
    print(f"positive rows: {len(rows)}")
    print(f"total words: {total_words}")
    print(f"in-vocab words: {n_in} ({100*n_in/total_words:.1f}%)")
    print(f"OOV words: {n_out} ({100*n_out/total_words:.1f}%)")
    print(f"distinct OOV types: {len(oov)}")
    print(f"rows fully in-vocab: {fully_in} "
          f"({100*fully_in/len(rows):.1f}%)")
    print(f"\n=== {args.limit} MOST FREQUENT OOV WORDS (count) ===")
    for w, c in oov.most_common(args.limit):
        print(f"  {c:6d}  {w}")

    # words that appear in train but not val/test (risk of overfit vocab)
    only_train = [w for w in oov if freq_split["val"][w] == 0
                  and freq_split["test"][w] == 0]
    print(f"\nOOV types only in train split: {len(only_train)}")

    # save full OOV table for the vocab builder
    out = os.path.join(_REPO, "data", "manifests", "oov_words.json")
    with open(out, "w") as f:
        json.dump({w: c for w, c in oov.most_common()}, f, indent=1)
    print(f"\nsaved OOV table -> {out}")


if __name__ == "__main__":
    main()
