#!/usr/bin/env python3
"""Vocab coverage curve + digit-token distribution for the VCM (Plan A).

Decides how many OOV words to add: reports the fraction of *words* (and of
*rows*) that become fully in-vocab as we add the top-K OOV word types.
Also dumps the digit-token distribution (transcripts use digits for slots).
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
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return [w for w in text.split() if w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    with open(args.manifest) as f:
        rows = [r for r in csv.DictReader(f) if r["polarity"] == "positive"]

    cur = set(VOCAB)
    freq = Counter()
    row_toks = []
    for r in rows:
        toks = tokenize(r["transcript"])
        row_toks.append(toks)
        for t in toks:
            freq[t] += 1

    oov = Counter({w: c for w, c in freq.items() if w not in cur})
    total_words = sum(freq.values())

    def coverage(extra):
        vocab = cur | set(extra)
        nw = sum(c for w, c in freq.items() if w in vocab)
        nrow = sum(1 for toks in row_toks
                   if toks and all(t in vocab for t in toks))
        return 100 * nw / total_words, 100 * nrow / len(row_toks), len(vocab)

    print("K_added  vocab_sz  word_cov%  row_cov%")
    base_w, base_r, base_sz = coverage([])
    print(f"   {0:5d}  {base_sz:8d}  {base_w:8.1f}  {base_r:8.1f}   (current)")
    oov_sorted = [w for w, _ in oov.most_common()]
    for K in [50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1447]:
        K = min(K, len(oov_sorted))
        w, r, sz = coverage(oov_sorted[:K])
        print(f"   {K:5d}  {sz:8d}  {w:8.1f}  {r:8.1f}")

    # digit tokens
    digits = Counter({w: c for w, c in freq.items() if w.isdigit()})
    print(f"\ndigit tokens: {len(digits)} types, "
          f"{sum(digits.values())} words")
    print("digit token freq (all):")
    print("  " + "  ".join(f"{w}:{c}" for w, c in digits.most_common()))

    # alpha-only OOV (non-digit) count
    alpha_oov = {w: c for w, c in oov.items() if not w.isdigit()}
    print(f"\nalpha OOV types: {len(alpha_oov)}, "
          f"words: {sum(alpha_oov.values())}")

    out = os.path.join(_REPO, "data", "manifests", "vocab_curve.json")
    with open(out, "w") as f:
        json.dump({"oov_sorted": oov_sorted,
                   "digit": dict(digits.most_common())}, f)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
