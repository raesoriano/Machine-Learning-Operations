#!/usr/bin/env python3
"""Command-level ceiling: parse(transcript) -> canonicalize -> gold command?

This is the upper bound any model can reach on the frozen benchmark: if the
model emits the transcript perfectly, pred_command == gold_command. Measures
how well the (hardened) parser + canonicalizer recover the 31 gold commands.
"""
import csv, json, os, sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from vcm.parser import parse  # noqa: E402
from vcm.canonical import canonicalize  # noqa: E402

MAN = os.path.join(REPO, "data", "manifests",
                   "positive_negative_manifest.csv")
with open(MAN) as f:
    rows = [r for r in csv.DictReader(f) if r["polarity"] == "positive"]

ok = 0
by_intent = defaultdict(lambda: [0, 0])
by_cmd = defaultdict(lambda: [0, 0])
fails = Counter()
examples = defaultdict(list)
for r in rows:
    pred = canonicalize(parse(r["transcript"]).intent,
                        parse(r["transcript"]).slots)
    gold = r["command"]
    hit = pred == gold
    ok += hit
    by_intent[r["intent"]][1] += 1
    by_intent[r["intent"]][0] += hit
    by_cmd[gold][1] += 1
    by_cmd[gold][0] += hit
    if not hit:
        key = f'{gold} -> {pred}'
        fails[key] += 1
        if len(examples[key]) < 2:
            examples[key].append(r["transcript"])

n = len(rows)
print(f"rows: {n}")
print(f"COMMAND CEILING (parse+canonicalize vs gold command): {100*ok/n:.1f}%")
print("\nper-intent command accuracy:")
for intent, (e, c) in sorted(by_intent.items(), key=lambda kv: -kv[1][1]):
    print(f"  {intent:18s} {100*e/c:5.1f}%  ({e}/{c})")
print("\ntop failure transitions (gold -> pred):")
for k, c in fails.most_common(15):
    print(f"  {c:5d}  {k}")
    for ex in examples[k]:
        print(f"        {ex!r}")
