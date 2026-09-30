"""Mock models — stand-ins so the benchmark works before any model exists.

``mock_clean``   : returns the gold text verbatim (harness sanity check;
                   should score ~100% on the synthetic set).
``mock_corrupt`` : returns the gold text with a controllable word error rate
                   (simulates an imperfect ASR front-end; shows the harness
                   producing realistic, sub-100% numbers).

These are NOT baselines for the leaderboard — they validate the pipeline.
"""
import random

from .base import Model


class MockClean(Model):
    name = "mock_clean"

    def __init__(self, gold_lookup=None):
        self.gold = gold_lookup or {}

    def load(self, path):
        pass

    def transcribe(self, audio, sr=16000):
        # The synthetic test set passes the gold text through a side channel
        # (see synthetic_set.py) because there is no real audio yet.
        return self.gold.get("_last_", "")


class MockCorrupt(Model):
    name = "mock_corrupt"

    def __init__(self, wer_target=0.15, seed=0):
        self.wer_target = wer_target
        self.rng = random.Random(seed)
        self.gold = {}

    def load(self, path):
        pass

    def transcribe(self, audio, sr=16000):
        text = self.gold.get("_last_", "")
        words = text.split()
        out = []
        for w in words:
            r = self.rng.random()
            if r < self.wer_target / 3:          # delete
                continue
            elif r < 2 * self.wer_target / 3:    # substitute
                out.append("the" if w != "the" else "a")
            else:
                out.append(w)
        return " ".join(out)
