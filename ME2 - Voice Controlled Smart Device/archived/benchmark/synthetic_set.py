"""Synthetic stand-in test set (works TODAY, no audio, no model).

Generated from the same template grammar as the training data, but with a
DIFFERENT seed and a held-out fraction of slot values, so it does not
trivially overlap the training set. Text-only: rows carry `text` (gold
transcript) + `intent` + `slots`; the mock models consume `text` directly.

When the frozen v1 set is ready (WP4), this file is retired — `evaluate.py`
switches to `--testset frozen` with zero code changes.

Usage:
    python -m benchmark.synthetic_set --out benchmark/testset/synthetic.jsonl \
        --n 2000 --seed 999
"""
import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data.templates.grammar import generate, OOV_UTTERANCES  # noqa: E402
from vcm.parser import parse  # noqa: E402

# OOV rows are pre-filtered through the parser: only utterances the parser
# itself rejects (intent == unknown) are valid rejection test cases. This keeps
# the synthetic set self-consistent (mock_clean must score 100%).
_VALID_OOV = [u for u in OOV_UTTERANCES if parse(u).intent == "unknown"]
if len(_VALID_OOV) < 10:
    raise RuntimeError("OOV filter removed too many utterances")


def build(n=2000, seed=999, oov_ratio=0.10):
    """Yield {id, text, intent, slots, subset} rows.

    subset: 'core' (in-vocab) or 'oov' (rejection). The frozen set adds
    'noise' and 'farfield' subsets; the harness handles any subset name.
    """
    import random as _random
    n_oov = int(n * oov_ratio)
    rows = list(generate(n - n_oov, seed=seed, oov_ratio=0.0))
    rng = _random.Random(seed + 1)
    for i in range(n_oov):
        text = rng.choice(_VALID_OOV)
        rows.append({"id": f"gen_{seed}_oov{i:05d}", "text": text,
                     "intent": "unknown", "slots": {}})
    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row = dict(row)
        row["id"] = f"syn_{seed}_{i:06d}"
        row["subset"] = "oov" if row["intent"] == "unknown" else "core"
        yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--oov-ratio", type=float, default=0.10)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in build(args.n, args.seed, args.oov_ratio):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {args.n} rows -> {out}")


if __name__ == "__main__":
    main()
