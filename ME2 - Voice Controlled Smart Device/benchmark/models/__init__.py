"""Swappable model backends for the benchmark.

    from benchmark.models import get_model
    model = get_model("mock_clean")            # harness sanity
    model = get_model("onnx", "model/checkpoints/vcm_v1.onnx")
    model = get_model("vosk", "model/vosk-model-small-en-us-0.15")
"""
from pathlib import Path

from .base import Model  # noqa: F401
from .mock import MockClean, MockCorrupt  # noqa: F401


def get_model(kind, path=None, **kw):
    kind = kind.lower()
    if kind == "mock_clean":
        return MockClean()
    if kind == "mock_corrupt":
        return MockCorrupt(wer_target=kw.get("wer", 0.15),
                           seed=kw.get("seed", 0))
    if kind == "onnx":
        return _OnnxModel(path)
    if kind == "vosk":
        return _VoskModel(path)
    if kind == "torch":
        return _TorchModel(path)
    raise ValueError(f"unknown model kind: {kind} "
                     f"(have: mock_clean, mock_corrupt, onnx, torch, vosk)")


class _OnnxModel(Model):
    """Runs an exported ONNX VCM (log-mel in -> CTC logits out).

    Expected ONNX signature (see deploy/export_onnx.py):
        input  "mels"   : [1, T, 40]  float32, 16 kHz, 25 ms / 10 ms
        output "logits" : [1, T, V]   float32 CTC logits (V = |VOCAB| + 1,
                                      blank = 0)

    Greedy CTC collapse + vocab mapping happen here in numpy - the same
    decoder the RPi service uses. Constrained decoding is inherent: the head
    only has |VOCAB| + 1 outputs, so the model can only ever emit command
    words.
    """
    name = "onnx"

    def __init__(self, path):
        self.path = path
        self.session = None

    def load(self, path=None):
        import onnxruntime as ort  # local import: optional dependency
        self.session = ort.InferenceSession(
            path or self.path,
            providers=["CPUExecutionProvider"])

    def transcribe(self, audio, sr=16000):
        import numpy as np
        from .feats import log_mel  # local feature extraction, no torch
        mel = log_mel(audio, sr=sr)                    # [T, 40]
        logits = self.session.run(None, {"mels": mel[None]})[0][0]  # [T, V]
        return greedy_ctc_to_text(logits)


def greedy_ctc_to_text(logits, blank=0):
    """CTC logits [T, V] -> transcript string (greedy collapse, numpy only).

    Mirror of model/decode.py so the ONNX runtime needs no torch.
    """
    import numpy as np
    from vcm.vocab import ID2WORD
    frame_ids = np.argmax(logits, axis=-1).tolist()
    out, prev = [], None
    for i in frame_ids:
        if i != blank and i != prev:
            out.append(ID2WORD[i - 1])
        prev = i
    return " ".join(out)


class _TorchModel(Model):
    """Runs a PyTorch checkpoint (best.pt) directly - for GPU dev/eval.

    Same pipeline as _OnnxModel (log-mel -> CTC logits -> greedy collapse),
    so the torch and onnx backends are interchangeable for benchmarking.
    """
    name = "torch"

    def __init__(self, path):
        self.path = path
        self.model = None
        self.device = None

    def load(self, path=None):
        import torch
        import yaml
        from model.model_def import VCMEncoder
        p = Path(path or self.path)
        cfg_path = p.parent / "config.yaml"
        cfg = yaml.safe_load(open(cfg_path)) if cfg_path.exists() else {}
        model = VCMEncoder(n_mels=cfg.get("n_mels", 40),
                           channels=cfg.get("channels", 128),
                           blocks=cfg.get("blocks", 4))
        model.load_state_dict(torch.load(p, map_location="cpu"))
        model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def transcribe(self, audio, sr=16000):
        import torch
        from .feats import log_mel
        mel = log_mel(audio, sr=sr)
        with torch.no_grad():
            logits = self.model(
                torch.from_numpy(mel)[None].to(self.device))[0].cpu().numpy()
        return greedy_ctc_to_text(logits)


class _VoskModel(Model):
    """Baseline: Vosk small model + the shared parser.

    Vosk is a tiny local ASR (no cloud). It is NOT a VCM — it is the
    "off-the-shelf ASR + rules" baseline the leaderboard starts from.
    """
    name = "vosk"

    def __init__(self, path):
        self.path = path
        self.rec = None

    def load(self, path=None):
        from vosk import Model, KaldiRecognizer  # local import: optional
        m = Model(path or self.path)
        self.rec = KaldiRecognizer(m, 16000)

    def transcribe(self, audio, sr=16000):
        import json
        import numpy as np
        from scipy.signal import resample_poly
        if sr != 16000:
            audio = resample_poly(audio, 16000, sr)
        self.rec.Reset()
        pcm = (audio * 32767).astype(np.int16).tobytes()
        ok, res = self.rec.AcceptWaveform(pcm)
        if not ok:
            res, _ = self.rec.FinalResult(), True
        return json.loads(res).get("text", "")
