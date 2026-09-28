"""Stage 1 (whisper variant): audio -> transcript via a pretrained general
ASR (faster-whisper), instead of the constrained-vocab wav2vec2 CTC.

Why this exists:
    The wav2vec2-base-960h fine-tuned with a CTC head over the 1033-word
    constrained ME2 vocab does NOT converge -- on the in-domain ME2 test
    split it emits incoherent word streams (250% WER vs the from-scratch
    CTC's 47.7%). The constrained head fights the pretrained encoder.

    A general-purpose, speaker-robust ASR (Whisper, trained on ~680k h of
    diverse speech) is the natural fix for the *new-speaker* test set: it
    was never seen this speaker or this domain, so it should transcribe the
    spoken words faithfully, and the stage-2 classifier (99.4% on gold
    transcripts) then maps words -> command.

    The transcript is post-processed through the same `normalize()` the
    pipeline uses (lowercase, digits<->number-words alignment) so the
    classifier sees exactly the same text form it was trained on.

Usage:
    python -m vcm2b.asr_whisper        # smoke test on a few clips
"""
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../backbone/vcm2b
_BACK = os.path.dirname(_HERE)                              # .../backbone
_REPO = os.path.dirname(_BACK)                              # .../VCM-v2
_SANDBOX = os.path.dirname(os.path.dirname(_ME2))  # .../sandbox
_ME2 = os.path.dirname(_REPO)  # vcm-v2 lives inside the ME2 folder
sys.path.insert(0, _BACK)
sys.path.insert(0, os.path.join(_REPO, "archive", "ctc_v8"))
sys.path.insert(0, _ME2)

from vcm2.normalize import normalize  # noqa: E402

SR = 16000


class ASRWhisper:
    """audio (float32 mono, any sr) -> transcript string (normalized)."""

    def __init__(self, model_size="base.en", device="cuda", compute="float16",
                 language="en"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, device=device,
                                  compute_type=compute)
        self.language = language

    def transcribe(self, audio, sr=SR):
        """audio: float32 mono ndarray -> normalized transcript string."""
        audio = np.asarray(audio, dtype=np.float32)
        if sr != SR:
            import torchaudio
            audio = torchaudio.functional.resample(
                torch_from(audio), sr, SR).numpy().astype(np.float32)
        segments, _info = self.model.transcribe(
            audio, language=self.language, beam_size=1,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            vad_filter=False)
        text = " ".join(s.text.strip() for s in segments)
        # Anti-hallucination guard: cap pathological repetition loops.
        words = text.split()
        if len(words) > 200:
            text = " ".join(words[:200])
        return normalize(text)


def torch_from(a):
    import torch
    return torch.from_numpy(a)


if __name__ == "__main__":
    import glob
    import soundfile as sf
    adt = os.path.join(_ME2, "data", "additional_test_data")
    dev = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "0") != "" else "cpu"
    asr = ASRWhisper(device=dev)
    files = sorted(glob.glob(os.path.join(adt, "*", "*.wav")))[:8]
    for fp in files:
        a, sr = sf.read(fp, dtype="float32", always_2d=True)
        a = a.mean(axis=1)
        print(f"  {os.path.basename(fp):34s} -> {asr.transcribe(a, sr)!r}")
