"""Local feature extraction — SINGLE SOURCE OF TRUTH for the VCM.

log_mel(audio, sr=16000) -> [T, 40] float32
  25 ms Hann window, 10 ms hop, 40 log-mel filters (50 Hz – 8 kHz),
  log1p scaling, at 16 kHz.

16 kHz is the one sample rate for the whole project: every dataset in the
manifest (optionb, Fluent, GSCv2, SLURP, LibriSpeech) is 16 kHz mono PCM_16,
the RPi mic captures at 16 kHz, and the VAD runs at 16 kHz — so no resampling
happens anywhere in the pipeline. (The mel is capped at 8 kHz, so 16 kHz
Nyquist is exactly sufficient.)

Used identically by training (model/train.py), the ONNX runtime
(benchmark/models/feats.py re-exports this), and the RPi service, so
train/serve features can never drift apart.
"""
import numpy as np
from scipy.signal import resample_poly

N_MELS = 40
SR = 16000
FRAME = 400          # 25 ms @ 16 kHz
HOP = 160            # 10 ms @ 16 kHz


def _mel_filters(n_fft, n_mels, sr, fmin=50.0, fmax=8000.0):
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min, mel_max = hz2mel(fmin), hz2mel(fmax)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    pts = mel2hz(mels).clip(0, sr / 2)
    bins = (n_fft / 2 + 1) * (pts / (sr / 2))
    f = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        if c > l:
            f[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            f[i, c:r] = (r - np.arange(c, r)) / (r - c)
    return f


_MEL = None


def _get_mel(n_fft, sr):
    global _MEL
    if _MEL is None:
        _MEL = _mel_filters(n_fft, N_MELS, sr)
    return _MEL


def log_mel(audio, sr=16000, n_fft=512):
    """audio: float32 mono ndarray (any sr) -> [T, 40] float32.

    If sr != 16000 the audio is resampled to 16 kHz first (defensive; the
    whole project is 16 kHz, so this is a no-op on the real pipeline).
    """
    if sr != SR:
        audio = resample_poly(audio, SR, sr)
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < FRAME:
        audio = np.pad(audio, (0, FRAME - len(audio)))
    n_frames = 1 + (len(audio) - FRAME) // HOP
    idx = (np.arange(n_frames)[:, None] * HOP +
           np.arange(FRAME)[None, :])
    frames = audio[idx]
    window = np.hanning(FRAME).astype(np.float32)
    frames = frames * window
    spec = np.abs(np.fft.rfft(frames, n_fft, axis=1)) ** 2
    mel = spec @ _get_mel(n_fft, SR).T
    return np.log1p(mel).astype(np.float32)
