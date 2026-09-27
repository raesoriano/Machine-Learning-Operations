#!/usr/bin/env python3
"""Verify sample rates across every source in the ME2 manifest (16k check)."""
import csv, os, sys
from collections import Counter, defaultdict
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = os.path.join(REPO, "data", "manifests",
                   "positive_negative_manifest.csv")
DATA = os.path.join(REPO, "data")

with open(MAN) as f:
    rows = list(csv.DictReader(f))

by_src = defaultdict(list)
for r in rows:
    by_src[r["source"]].append(r)

print(f"{'source':32s} {'n':>6s}  sample_rates (checked up to 8 each)")
for src, rs in sorted(by_src.items()):
    rates = Counter()
    for r in rs[:8]:
        p = r["audio"]
        if not os.path.isabs(p):
            p = os.path.join(DATA, p)
        if not os.path.exists(p):
            rates["<missing>"] += 1
            continue
        try:
            info = sf.info(p)
            rates[f"{info.samplerate}Hz/{info.channels}ch/{info.subtype}"] += 1
        except Exception as e:
            rates[f"<err {type(e).__name__}>"] += 1
    print(f"{src:32s} {len(rs):6d}  {dict(rates)}")
