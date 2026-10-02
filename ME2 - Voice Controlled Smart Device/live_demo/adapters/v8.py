"""Adapter for the v8 Conformer+CTC recognizer (ONNX, no torch/torchaudio).

Imports the PRODUCTION runtime from `pi test v8-conformer-ctc/v8_onnx.py`
(the exact decode protocol the v8 eval uses: constrained CTC-FSA over the 93
command phrases + greedy free decode + reject rules) and wraps it in the
shared interface:

    classify_warm(pcm: bytes, reps: int = 1) -> (command, intent, transcript, prob)

`command` is the COARSE 19-command schema label (or "REJECT") -- the same
schema the shared live loop's response map uses. The log-mel front-end is
baked into the ONNX graph, so this adapter needs only `onnxruntime` +
`numpy` (no torch, no torchaudio, no import from `pi test v6`).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

from shared import _ME2, COARSE_INTENT

_V8_DIR = os.path.join(_ME2, "pi test v8-conformer-ctc")
_DEFAULT_ONNX = os.path.join(_V8_DIR, "models_neg", "best.onnx")


def _load_v8_module():
    """Import pi test v8-conformer-ctc/v8_onnx.py under a unique module name
    (its top level only defines constants + classes; main() is guarded)."""
    path = os.path.join(_V8_DIR, "v8_onnx.py")
    spec = importlib.util.spec_from_file_location("_v8_onnx", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_v8_onnx"] = mod
    spec.loader.exec_module(mod)
    return mod


class V8Adapter:
    def __init__(self, model_path: str | None = None,
                 reject_empty: bool = True):
        self._mod = _load_v8_module()
        path = model_path or _DEFAULT_ONNX
        print(f"loading v8 Conformer+CTC ONNX ({path}) ...", flush=True)
        self._v8 = self._mod.V8Onnx(path, reject_empty=reject_empty)
        # warm the ONNX session (graph init + first inference) so the first
        # real command after the wake word is not slow
        self._v8._logprobs(np.zeros(1600, dtype=np.float32))
        self.name = "v8"

    def classify_warm(self, pcm: bytes, reps: int = 1):
        """16 kHz mono 16-bit PCM -> (command, intent, transcript, prob).

        `reps` is ignored: the ONNX session is stateless (no decoder AGC to
        warm), so one decode is the converged result.
        """
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        lp = self._v8._logprobs(x)
        c_lp, c_words = self._v8.fsa.decode(lp)
        f_lp, f_words = self._mod.greedy_ctc(lp)
        if c_words is not None:
            phrase = " ".join(self._v8.words[w - 1] for w in c_words)
            fine = self._mod.phrase_to_class(phrase)
            pred = self._mod.coarse_class(fine)
        else:
            phrase, fine, pred = "", "REJECT", "REJECT"
        reject = ((f_lp - c_lp) / max(1, lp.shape[0]) > self._v8.reject_margin)
        if self._v8.reject_empty and not f_words:
            reject = True
        if reject:
            pred = "REJECT"
        transcript = " ".join(self._v8.words[w - 1] for w in f_words)
        # confidence: log-prob gap between the free and the constrained
        # decode, scaled to [0, 1] (0.5 == tied, like the v3/v7 scale)
        prob = float(1.0 / (1.0 + np.exp(-(f_lp - c_lp) / max(1, lp.shape[0]))))
        return pred, COARSE_INTENT.get(pred, "unknown"), transcript, prob


def build(model_path: str | None = None, reject_empty: bool = True) -> V8Adapter:
    return V8Adapter(model_path, reject_empty=reject_empty)
