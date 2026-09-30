#!/usr/bin/env python3
"""Parser ceiling: for each positive row, does parse(transcript) recover the
gold (intent, slots)? This is the upper bound any model can reach on the
frozen benchmark (the model must emit the transcript the parser needs).
Also reports per-intent and the failure modes.
"""
import csv, json, os, sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from vcm.parser import parse  # noqa: E402
from benchmark.metrics import slots_equal  # noqa: E402

MAN = os.path.join(REPO, "data", "manifests",
                   "positive_negative_manifest.csv")

with open(MAN) as f:
    rows = [r for r in csv.DictReader(f) if r["polarity"] == "positive"]

ok = 0
intent_ok = 0
by_intent = defaultdict(lambda: [0, 0])   # intent -> [exact, n]
fails = Counter()
examples = defaultdict(list)
for r in rows:
    gold = {"intent": r["intent"], "slots": json.loads(r["slots"])}
    pred = parse(r["transcript"]).to_dict()
    exact = (pred["intent"] == gold["intent"]
             and slots_equal(gold["slots"], pred["slots"]))
    iok = pred["intent"] == gold["intent"]
    ok += exact
    intent_ok += iok
    by_intent[r["intent"]][1] += 1
    by_intent[r["intent"]][0] += exact
    if not exact:
        key = f'{gold["intent"]} -> {pred["intent"]}'
        fails[key] += 1
        if len(examples[key]) < 3:
            examples[key].append(
                f'  {r["transcript"]!r} | gold={gold["slots"]} '
                f'pred={pred["slots"]}')

n = len(rows)
print(f"rows: {n}")
print(f"parser ceiling (exact intent+slots): {100*ok/n:.1f}%")
print(f"parser intent-only:                  {100*intent_ok/n:.1f}%")
print("\nper-intent exact:")
for intent, (e, c) in sorted(by_intent.items(),
                             key=lambda kv: -kv[1][1]):
    print(f"  {intent:18s} {100*e/c:5.1f}%  ({e}/{c})")
print("\ntop failure transitions (gold -> pred):")
for k, c in fails.most_common(15):
    print(f"  {c:5d}  {k}")
    for ex in examples[k]:
        print(ex)
