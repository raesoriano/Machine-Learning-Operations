"""Stage 1 (backbone): audio -> transcript via a fine-tuned wav2vec2.

Replaces the from-scratch 422k CTC encoder (archive/ctc_v8/vcm2/asr.py) with
a pretrained facebook/wav2vec2-base-960h fine-tuned with CTC over the same
constrained ME2 vocabulary (185 words + blank = 186 classes). The stage-2
classifier is therefore unchanged, and the two ASR backends are directly
comparable on the same held-out 171-clip new-speaker test set.

The model is loaded from backbone/artifacts/w2v2_base (config.json +
model.safetensors + vocab.json, produced by scripts/finetune_w2v2.py).

IMPORTANT: the saved config.json still carries the backbone's ORIGINAL
vocab_size (32), because save_pretrained serialises the config object rather
than the (replaced) head. So from_pretrained builds a 32-way head that does
NOT match the 186-way weights in model.safetensors. We therefore load with
strict=False and rebuild lm_head to the size recorded in vocab.json -- the
same 186-way head the fine-tune trained.

Greedy CTC decoding + the ME2 confidence function are reused from the ME2
repo (model/decode.py) so the decode behaviour matches the CTC pipeline
exactly (same blank handling, same confidence metric for the reject gate).

Usage:
    python -m vcm2b.asr_w2v2            # smoke test on a few clips
"""
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../backbone/vcm2b
_BACK = os.path.dirname(_HERE)                              # .../backbone
_REPO = os.path.dirname(_BACK)                              # .../VCM-v2
_SANDBOX = os.path.dirname(_REPO)                           # .../sandbox
_ME2 = os.path.join(_SANDBOX, "Machine-Learning-Operations",
                    "ME2 - Voice Controlled Smart Device")
if _ME2 not in sys.path:
    sys.path.insert(0, _ME2)

from model.decode import decode_to_text, confidence  # noqa: E402

from transformers import AutoModelForCTC  # noqa: E402

_ART = os.path.join(_BACK, "artifacts", "w2v2_base")
SR = 16000


class ASRWav2Vec2:
    """audio (float32 mono, any sr) -> transcript string."""

    def __init__(self, art_dir=None, device="cpu"):
        art = art_dir or _ART
        # Rebuild the 186-way head (config.json still says 32 -- see module
        # docstring). strict=False ignores the mismatched original head.
        self.model = AutoModelForCTC.from_pretrained(art)
        with open(os.path.join(art, "vocab.json")) as f:
            vocab = json.load(f)
        n_classes = len(vocab["vocab"]) + 1  # words + blank
        self.model.lm_head = torch.nn.Linear(
            self.model.config.hidden_size, n_classes)
        from safetensors.torch import load_file
        sd = load_file(os.path.join(art, "model.safetensors"))
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        real_missing = [m for m in missing if not m.startswith("lm_head")]
        if real_missing:
            raise RuntimeError(f"missing non-head weights: {real_missing[:5]}")
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.art = art

    def _logits(self, audio, sr):
        audio = np.asarray(audio, dtype=np.float32)
        if sr != SR:
            import torchaudio
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio), sr, SR).numpy().astype(np.float32)
        x = torch.from_numpy(audio)[None].to(self.device)
        with torch.no_grad():
            return self.model(x).logits[0].float().cpu()

    def transcribe(self, audio, sr=SR):
        """audio: float32 mono ndarray -> transcript string."""
        return decode_to_text(self._logits(audio, sr))

    def transcribe_conf(self, audio, sr=SR, min_conf=None):
        """-> (transcript, mean non-blank log-prob). With min_conf set,
        low-confidence output is rejected to '' (the reject gate)."""
        logits = self._logits(audio, sr)
        conf = confidence(logits)
        text = decode_to_text(logits)
        if min_conf is not None and conf < min_conf:
            return "", conf
        return text, conf


if __name__ == "__main__":
    import glob
    import soundfile as sf
    adt = os.path.join(_SANDBOX, "additional_test_data")
    asr = ASRWav2Vec2()
    files = sorted(glob.glob(os.path.join(adt, "*", "*.wav")))[:8]
    for fp in files:
        a, sr = sf.read(fp, dtype="float32", always_2d=True)
        a = a.mean(axis=1)
        print(f"  {os.path.basename(fp):34s} -> {asr.transcribe(a, sr)!r}")
