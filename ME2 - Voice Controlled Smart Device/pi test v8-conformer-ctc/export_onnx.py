#!/usr/bin/env python3
"""pi test v8-conformer-ctc -- export the Conformer+CTC model to ONNX.

Exports the FULL inference graph so the runtime needs no torch/torchaudio:

    16 kHz float32 waveform [B, T]
      -> log-mel front-end (torchaudio MelSpectrogram, identical to data.py)
      -> causal Conformer (6 layers, d=256) + CTC log-softmax
      -> log-probabilities [B, T', 110]   (T' = (T_mel - 2)//4 + 1)

Exported with the dynamo exporter at opset 18 (the legacy TorchScript
exporter cannot trace torch.stft).

The standalone runtime `v8_onnx.py` therefore needs only onnxruntime +
numpy (plus the stdlib) -- no torch, no torchaudio, no v6-folder import.

This script also bakes `meta.json` next to the ONNX file (the 109-word
vocab + the 93 command phrases) so v8_onnx.py is fully self-contained.

Usage:
    python export_onnx.py --model models/best.pt --out models/best.onnx
    python export_onnx.py --model models_neg/best.pt --out models_neg/best.onnx
    python export_onnx.py --model models/best.pt --out models/best.onnx --int8
        (also writes models/best_int8.onnx, dynamic int8 quantization)

Verification: after export, a handful of real holdout clips are run through
both the torch model (fp32, CPU) and the ONNX session; max log-prob
difference and argmax agreement are printed.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchaudio

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from model import V8Model                        # noqa: E402
from data import read_wav16, load_test           # noqa: E402

MEL_BINS = 80
FREQ_MIN, FREQ_MAX = 133.0, 6855.0
N_FFT, WIN, HOP = 512, 400, 160
OPSET = 18  # dynamo exporter needs >= 18 (its 17 down-conversion fails on Pad)


def make_mel_tf() -> nn.Module:
    """The exact front-end used in data.py log_mel()."""
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
        f_min=FREQ_MIN, f_max=FREQ_MAX, n_mels=MEL_BINS,
        window_fn=lambda n: torch.hann_window(n))


class V8Full(nn.Module):
    """wav [B, T] -> CTC log-probs [B, T', V+1]."""

    def __init__(self, model: V8Model, mel: nn.Module):
        super().__init__()
        self.model = model
        self.mel = mel

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        m = self.mel(wav)                                # [B, 80, Tm]
        m = torch.log(m.clamp_min(1e-5)).transpose(1, 2)  # [B, Tm, 80]
        return self.model(m)


def load_phrases() -> list[str]:
    import csv as _csv
    from hgm.spoken import tokenize_spoken
    out = []
    with open(os.path.join(_HERE, "variations.csv")) as f:
        for r in _csv.DictReader(f):
            out.append(" ".join(tokenize_spoken(r["phrase"])))
    return out


def export(ckpt_path: str, out_path: str, do_int8: bool,
           verify_clips: list[str]) -> None:
    ck = torch.load(ckpt_path, map_location="cpu")
    words = ck["words"]
    model = V8Model(ck["n_words"], d_model=ck["d_model"],
                    layers=ck["layers"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    full = V8Full(model, make_mel_tf()).eval()
    dummy = torch.zeros(1, 16000)
    dynamic_shapes = {"wav": {0: "B", 1: "T"}}
    with torch.no_grad():
        torch.onnx.export(
            full, (dummy,), out_path,
            input_names=["wav"], output_names=["logprobs"],
            dynamic_shapes=dynamic_shapes, opset_version=OPSET,
            dynamo=True)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"exported {out_path} ({size_mb:.1f} MB, opset {OPSET})",
          flush=True)

    # bake meta (vocab + phrases) next to the onnx
    meta = {"words": words, "phrases": load_phrases(),
            "n_words": ck["n_words"], "d_model": ck["d_model"],
            "layers": ck["layers"], "source_ckpt": os.path.basename(
                ckpt_path)}
    with open(os.path.join(os.path.dirname(out_path), "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"baked {os.path.join(os.path.dirname(out_path), 'meta.json')}",
          flush=True)

    verify(full, out_path, verify_clips)

    if do_int8:
        int8_path = os.path.splitext(out_path)[0] + "_int8.onnx"
        qm = torch.quantization.quantize_dynamic(
            full, {torch.nn.Linear}, dtype=torch.qint8)
        qm.eval()
        try:
            with torch.no_grad():
                torch.onnx.export(
                    qm, (dummy,), int8_path,
                    input_names=["wav"], output_names=["logprobs"],
                    dynamic_shapes=dynamic_shapes, opset_version=OPSET,
                    dynamo=True)
            print(f"exported {int8_path} "
                  f"({os.path.getsize(int8_path) / 1e6:.1f} MB)", flush=True)
            verify(qm, int8_path, verify_clips)
        except Exception as e:  # noqa: BLE001
            print(f"INT8 EXPORT FAILED ({type(e).__name__}: {str(e)[:200]}) "
                  f"-- keeping fp32 only", flush=True)
            if os.path.exists(int8_path):
                os.remove(int8_path)


def verify(torch_mod: nn.Module, onnx_path: str,
           clips: list[str]) -> None:
    """Run real clips through torch (fp32 CPU) and onnxruntime; compare."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    torch_mod.eval()
    max_diff, n_argmax_flip, n_clips = 0.0, 0, 0
    for p in clips:
        x = read_wav16(p)
        if x.shape[0] > 16000 * 8:
            x = x[:16000 * 8]
        t = torch.from_numpy(x).unsqueeze(0)
        with torch.no_grad():
            ref = torch_mod(t).numpy()
        got = sess.run(None, {"wav": t.numpy().astype(np.float32)})[0]
        T = min(ref.shape[1], got.shape[1])
        d = float(np.abs(ref[0, :T] - got[0, :T]).max())
        max_diff = max(max_diff, d)
        flip = int((ref[0, :T].argmax(-1) != got[0, :T].argmax(-1)).sum())
        n_argmax_flip += flip
        n_clips += 1
        print(f"  verify {os.path.basename(p)}: T={T} max|dlogp|={d:.3e} "
              f"argmax_flips={flip}", flush=True)
    tot = n_argmax_flip
    print(f"VERIFY {os.path.basename(onnx_path)}: {n_clips} clips, "
          f"max|dlogp|={max_diff:.3e}, total argmax flips={tot}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--data",
                    default=os.path.normpath(os.path.join(_HERE, "..", "data", "dataset")))
    ap.add_argument("--verify-n", type=int, default=8)
    args = ap.parse_args()

    rows = load_test(args.data, "holdout")
    clips = [r[0] for r in rows[:args.verify_n]]
    export(args.model, args.out, args.int8, clips)


if __name__ == "__main__":
    main()
