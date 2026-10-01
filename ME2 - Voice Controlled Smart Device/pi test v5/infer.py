"""pi test v5 -- ONNX inference (no torch; onnxruntime + numpy only).

Loads the two exported models and runs the full cascade on a 16 kHz clip:

    pcm16 -> log-Mel (feats.py) -> ASRModel (CTC) -> word sequence
          -> PhraseModel -> command / REJECT

This is the EXACT pipeline used at training time, so the Pi behaves like the
HPC evaluation.  Only onnxruntime + numpy + scipy are required.
"""
from __future__ import annotations
import os, json
import numpy as np
import onnxruntime as ort

import feats as F

HERE = os.path.dirname(os.path.abspath(__file__))
ASR_ONNX = os.path.join(HERE, "models", "asr.onnx")
CLS_ONNX = os.path.join(HERE, "models", "classify.onnx")
VOCAB = os.path.join(HERE, "data", "vocab.json")

REJECT = "REJECT"


class V5Recognizer:
    def __init__(self, asr_path=ASR_ONNX, cls_path=CLS_ONNX, vocab_path=VOCAB):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.asr = ort.InferenceSession(asr_path, opts, providers=["CPUExecutionProvider"])
        self.cls = ort.InferenceSession(cls_path, opts, providers=["CPUExecutionProvider"])
        v = json.load(open(vocab_path))
        self.i2w = {int(k): w for k, w in v["idx2word"].items()}
        self.blank = v["blank"]
        self.unk = v["unk"]
        self.classes = v["classes"]
        # The classifier's word-sequence width comes from the ONNX graph so it
        # always matches the exported model (trained at maxlen 16, which is
        # wider than the ASR target maxlen stored in vocab.json).
        self.maxlen = int(self.cls.get_inputs()[0].shape[1])

    # ---- ASR ----
    def _asr_decode(self, mel):
        logits = self.asr.run(None, {"mel": mel.astype(np.float32)})[0]  # (1,T,V)
        idx = logits[0].argmax(axis=-1)
        seq = []
        prev = None
        for t in range(idx.shape[0]):
            tok = int(idx[t])
            if tok != prev:
                if tok != self.blank:
                    seq.append(tok)
            prev = tok
        return seq

    # ---- classify ----
    def _classify(self, seq):
        w = np.zeros((1, self.maxlen), dtype=np.int64)
        seq = seq[:self.maxlen]
        w[0, :len(seq)] = seq
        logits = self.cls.run(None, {"words": w})[0]  # (1, C)
        return int(logits[0].argmax())

    def recognize(self, pcm16: bytes, sr: int = F.SR) -> dict:
        """pcm16: int16 mono bytes -> {command, words, heard, is_reject, scores}."""
        mel = F.logmel_from_pcm16(pcm16, sr)          # (40, 301)
        mel = mel[None, ...]                           # (1, 40, 301)
        seq = self._asr_decode(mel)
        words = [self.i2w[t] for t in seq]
        heard = " ".join(words)
        ci = self._classify(seq)
        command = self.classes[ci]
        return {
            "command": command,
            "words": words,
            "heard": heard,
            "is_reject": command == REJECT,
            "class_index": ci,
        }

    def recognize_float(self, x: np.ndarray, sr: int = F.SR) -> dict:
        """x: float32 mono in [-1,1] -> same dict as recognize()."""
        mel = F.logmel(x, sr)[None, ...]
        seq = self._asr_decode(mel)
        words = [self.i2w[t] for t in seq]
        ci = self._classify(seq)
        return {"command": self.classes[ci], "words": words,
                "heard": " ".join(words), "is_reject": self.classes[ci] == REJECT,
                "class_index": ci}


if __name__ == "__main__":
    import sys, soundfile as sf
    rec = V5Recognizer()
    for p in sys.argv[1:]:
        x, sr = sf.read(p, dtype="float32")
        r = rec.recognize_float(x, sr)
        print(f"{os.path.basename(p):40s} -> {r['command']:28s} heard='{r['heard']}'")
