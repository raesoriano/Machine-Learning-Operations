#!/usr/bin/env python3
"""Coverage AFTER digit->number-word normalization (the real training target).

The model is trained on normalize(transcript): standalone digit tokens become
number-words ("22" -> "twenty two"), matching what TTS/humans actually say.
This is the coverage that matters for CTC learnability.
"""
import csv, os, re, sys
from collections import Counter
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from vcm.vocab import VOCAB as BASE
from vcm.numbers import int_to_words

def tok(t):
    t = t.lower(); t = re.sub(r"[^a-z0-9'\s]", " ", t)
    return [w for w in t.split() if w]

def normalize(words):
    out = []
    for w in words:
        if w.isdigit() and 0 <= int(w) <= 120:
            out.extend(int_to_words(int(w)).split())
        else:
            out.append(w)
    return out

with open(sys.argv[1]) as f:
    rows = [r for r in csv.DictReader(f) if r["polarity"] == "positive"]
freq = Counter(); row_toks = []
for r in rows:
    ts = normalize(tok(r["transcript"])); row_toks.append(ts)
    for t in ts: freq[t] += 1
oov = {w: c for w, c in freq.items() if w not in BASE}
total = sum(freq.values())
print("freq_thr  n_added  vocab_sz  word_cov%  row_cov%")
for thr in [1, 2, 3, 5, 8, 10, 15, 20]:
    add = {w for w, c in oov.items() if c >= thr}
    vset = set(BASE) | add
    nw = sum(c for w, c in freq.items() if w in vset)
    nr = sum(1 for ts in row_toks if ts and all(t in vset for t in ts))
    print(f"   {thr:4d}  {len(add):7d}  {len(vset):8d}  "
          f"{100*nw/total:8.1f}  {100*nr/len(row_toks):8.1f}")
# also: what fraction of rows have ALL slot-relevant words covered is what
# matters for parsing; report rows with <=1 OOV word as "near-full"
vset = set(BASE) | {w for w, c in oov.items() if c >= 5}
near = sum(1 for ts in row_toks
           if ts and sum(1 for t in ts if t not in vset) <= 1)
print(f"\nwith thr=5 vocab: rows with <=1 OOV word: {100*near/len(row_toks):.1f}%")
