"""Stage 1: ASR  audio -> transcript (words actually spoken).

Reuses the ME2 me2_v6 transcript-trained CTC model (the exact model v1 trained
on the spoken transcript, with speaker augmentation). This is a *general*
command ASR: it emits whatever words the audio says, constrained to the ME2
command vocabulary. The classifier (Stage 2) then maps those words to one of
the 31 commands or REJECT.

Runs the PyTorch checkpoint on CPU (the RPi-representative path). The same
model can be exported to ONNX int8 for the RPi service (see ME2
deploy/export_onnx.py) - the classifier stage is unchanged.
"""
import os
import sys

import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_SANDBOX = os.path.dirname(os.path.dirname(_HERE))
_ME2 = os.path.join(_SANDBOX, "Machine-Learning-Operations",
                    "ME2 - Voice Controlled Smart Device")
if _ME2 not in sys.path:
    sys.path.insert(0, _ME2)

from vcm.features import log_mel                 # noqa: E402  (16 kHz, [T,40])
from model.decode import decode_to_text          # noqa: E402
from model.model_def import VCMEncoder           # noqa: E402

_CKPT = os.path.join(_ME2, "model", "checkpoints", "me2_v6", "best.pt")


class ASR:
    """audio (float32 mono, any sr) -> transcript string."""

    def __init__(self, ckpt=None, device="cpu"):
        ckpt = ckpt or _CKPT
        cfg = yaml.safe_load(open(os.path.join(os.path.dirname(ckpt),
                                               "config.yaml")))
        self.model = VCMEncoder(n_mels=cfg.get("n_mels", 40),
                                channels=cfg.get("channels", 128),
                                blocks=cfg.get("blocks", 4))
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.ckpt = ckpt
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)

    def transcribe(self, audio, sr=16000):
        """audio: float32 mono ndarray -> transcript string."""
        audio = np.asarray(audio, dtype=np.float32)
        mel = log_mel(audio, sr=sr)
        with torch.no_grad():
            logits = self.model(
                torch.from_numpy(mel)[None].to(self.device))[0].cpu().numpy()
        return decode_to_text(logits)


if __name__ == "__main__":
    import soundfile as sf
    import glob
    adt = os.path.join(_SANDBOX, "additional_test_data")
    asr = ASR()
    files = sorted(glob.glob(os.path.join(adt, "*", "*.wav")))[:8]
    for fp in files:
        a, sr = sf.read(fp, dtype="float32")
        print(f"  {os.path.basename(fp):34s} -> {asr.transcribe(a, sr)!r}")
