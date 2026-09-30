"""PR check: every generated row must round-trip through the parser.

Catches grammar/parser drift (the #1 way the dataset and benchmark silently
disagree). Run it in CI and before any PR that touches vcm/ or
data/templates/:

    python -m tools.check_grammar            # quick (2000 rows, 3 seeds)
    python -m tools.check_grammar --n 20000  # full

Exit code 0 = all rows parse to their gold (intent, slots).
"""
import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data.templates.grammar import generate  # noqa: E402
from vcm.parser import parse  # noqa: E402


def check(n, seed):
    bad = 0
    total = 0
    examples = []
    for row in generate(n, seed=seed, oov_ratio=0.10):
        total += 1
        pred = parse(row["text"])
        gold_ok = row["intent"] == pred.intent
        slots_ok = (row["intent"] == "unknown") or row["slots"] == pred.slots
        if not (gold_ok and slots_ok):
            bad += 1
            if len(examples) < 10:
                examples.append((row["text"], row, pred.to_dict()))
    return total, bad, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    total = bad = 0
    for seed in seeds:
        t, b, examples = check(args.n, seed)
        total += t
        bad += b
        status = "OK " if b == 0 else "FAIL"
        print(f"[{status}] seed {seed}: {t - b}/{t} rows consistent")
        for text, gold, pred in examples:
            print(f"    {text!r}\n      gold {gold}\n      pred {pred}")
    print(f"\ntotal: {total - bad}/{total} consistent "
          f"({(total - bad) / max(total, 1) * 100:.2f}%)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
