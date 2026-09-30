#!/usr/bin/env python3
"""OOV frequency-threshold distribution (to pick the vocab size)."""
import csv, os, re, sys
from collections import Counter
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from vcm.vocab import VOCAB as BASE

def tok(t):
    t = t.lower(); t = re.sub(r"[^a-z0-9'\s]", " ", t)
    return [w for w in t.split() if w]

with open(sys.argv[1]) as f:
    rows = [r for r in csv.DictReader(f) if r["polarity"] == "positive"]
freq = Counter()
row_toks = []
for r in rows:
    ts = tok(r["transcript"]); row_toks.append(ts)
    for t in ts: freq[t] += 1
oov = {w: c for w, c in freq.items() if w not in BASE and not w.isdigit()}
total = sum(freq.values())
print("freq_thr  n_oov_words  vocab_sz  word_cov%  row_cov%")
for thr in [1, 2, 3, 5, 8, 10, 15, 20]:
    add = {w for w, c in oov.items() if c >= thr}
    vset = set(BASE) | add
    nw = sum(c for w, c in freq.items() if w in vset)
    nr = sum(1 for ts in row_toks if ts and all(t in vset for t in ts))
    print(f"   {thr:4d}  {len(add):11d}  {len(vset):8d}  "
          f"{100*nw/total:8.1f}  {100*nr/len(row_toks):8.1f}")
