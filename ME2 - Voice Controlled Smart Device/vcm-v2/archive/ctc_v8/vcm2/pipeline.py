"""End-to-end VCM v2 pipeline:  audio -> ASR transcript -> 31 commands | REJECT.

    audio (16 kHz mono)
      -> ASR (me2_v6 CTC)            transcript
      -> normalize (digits->words)
      -> classifier (TF-IDF+LogReg)  command | REJECT

A single `Pipeline` object wraps both stages and reports per-clip latency.
"""
import time

import numpy as np

from .asr import ASR
from .classifier import load_classifier, predict
from .normalize import normalize


class Pipeline:
    def __init__(self, asr: ASR, classifier=None):
        self.asr = asr
        self.clf = classifier or load_classifier()

    def run(self, audio, sr=16000, warm=False):
        """audio -> dict with transcript, command, prob, latencies."""
        t0 = time.perf_counter()
        transcript = self.asr.transcribe(audio, sr=sr)
        t_asr = time.perf_counter() - t0

        t1 = time.perf_counter()
        command, prob = predict(self.clf, transcript)
        t_cls = time.perf_counter() - t1

        return {
            "transcript": transcript,
            "transcript_norm": normalize(transcript),
            "command": command,
            "prob": prob,
            "asr_ms": t_asr * 1e3,
            "cls_ms": t_cls * 1e3,
            "total_ms": (t_asr + t_cls) * 1e3,
        }


def wer(ref, hyp):
    """Word error rate between two strings (Levenshtein on word lists)."""
    ref = (ref or "").split()
    hyp = (hyp or "").split()
    if not ref:
        return 0.0 if not hyp else 1.0
    import difflib
    sm = difflib.SequenceMatcher(None, ref, hyp)
    ops = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        ops += max(i2 - i1, j2 - j1)
    return ops / len(ref)
