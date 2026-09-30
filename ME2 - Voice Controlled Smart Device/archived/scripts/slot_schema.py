#!/usr/bin/env python3
"""Per-intent: manifest slots schema vs parser output schema.

For each intent, show distinct manifest slot dicts and distinct
parse(transcript) slot dicts, plus how often they agree. Decides whether to
(a) fix the parser to reproduce manifest slots, or (b) use parse(gold) as gold.
"""
import csv, json, os, sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from vcm.parser import parse  # noqa: E402

MAN = os.path.join(REPO, "data", "manifests",
                   "positive_negative_manifest.csv")
with open(MAN) as f:
    rows = [r for r in csv.DictReader(f) if r["polarity"] == "positive"]

by_intent = defaultdict(list)
for r in rows:
    by_intent[r["intent"]].append(r)

for intent in sorted(by_intent):
    rs = by_intent[intent]
    man_slots = Counter(json.dumps(json.loads(r["slots"]), sort_keys=True)
                        for r in rs)
    par_slots = Counter(json.dumps(parse(r["transcript"]).slots,
                                   sort_keys=True) for r in rs)
    par_intent = Counter(parse(r["transcript"]).intent for r in rs)
    agree = sum(1 for r in rs
                if parse(r["transcript"]).intent == intent
                and parse(r["transcript"]).slots == json.loads(r["slots"]))
    print(f"\n===== {intent}  (n={len(rs)})  parser-intent dist: "
          f"{dict(par_intent.most_common(4))}")
    print(f"  manifest slots (top5): {man_slots.most_common(5)}")
    print(f"  parser   slots (top5): {par_slots.most_common(5)}")
    print(f"  exact (intent+slots) agree: {100*agree/len(rs):.1f}%")
    # sample transcripts
    print(f"  sample transcripts: {[r['transcript'] for r in rs[:4]]}")
