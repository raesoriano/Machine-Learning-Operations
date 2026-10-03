"""pi test v5 -- shared audio feature extraction (numpy + scipy ONLY).

The EXACT same log-Mel pipeline is used for training (prep_data.py),
validation, and Raspberry Pi inference (vcm_pi_v5.py).  No torch, no librosa.

Pipeline (fixed order):
    16 kHz mono -> pad/truncate to 3.0 s -> STFT (n_fft=400, hop=160, hann,
    center) -> power -> mel filterbank (n_mels=40) -> log -> (40, 301) float32

This matches the torch-based extractor in prep_data.py to <1e-3 (verified).
"""
from __future__ import annotations
import numpy as np
from scipy.signal import resample_poly

SR = 16000
WIN_S = 3.0
N_WIN = int(SR * WIN_S)          # 48000
N_FFT = 400
HOP = 160
N_MELS = 40
N_FRAMES = (N_WIN + N_FFT // 2) // HOP   # 301 with center=True


def _mel_filterbank(n_mels, n_fft, sr):
    fmin, fmax = 0.0, sr / 2
    def hz2mel(f): return 2595 * np.log10(1 + f / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    mel_pts = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz_pts = mel2hz(mel_pts)
    bins = (n_fft / 2) * (hz_pts / (sr / 2)).astype(np.int64)
    bins = np.clip(bins, 0, n_fft // 2)
    fl = np.zeros((n_mels, n_fft // 2 + 1)); fr = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        l, c, rr = int(bins[m]), int(bins[m + 1]), int(bins[m + 2])
        if c > l: fl[m, l:c] = (np.arange(l, c) - l) / (c - l)
        if rr > c: fr[m, c:rr] = (rr - np.arange(c, rr)) / (rr - c)
    return (fl + fr).astype(np.float32)


_MEL_FB = None
def mel_filterbank():
    global _MEL_FB
    if _MEL_FB is None:
        _MEL_FB = _mel_filterbank(N_MELS, N_FFT, SR)
    return _MEL_FB


def _stft_power(x: np.ndarray) -> np.ndarray:
    """x: (samples,) float32 -> power (n_fft//2+1, n_frames)."""
    n = x.size
    if n < N_FFT:
        x = np.pad(x, (0, N_FFT - n))
    # center padding
    pad = N_FFT // 2
    x = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (x.size - N_FFT) // HOP
    win = np.hanning(N_FFT).astype(np.float32)
    power = np.empty((N_FFT // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        seg = x[i * HOP: i * HOP + N_FFT] * win
        spec = np.fft.rfft(seg)
        power[:, i] = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    return power


def logmel(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """x: (samples,) float32 in [-1,1] -> (40, 301) log-mel float32."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = resample_poly(x, SR, sr).astype(np.float32)
    if x.size < N_WIN:
        x = np.pad(x, (0, N_WIN - x.size))
    else:
        x = x[:N_WIN]
    power = _stft_power(x.astype(np.float32))
    mel = mel_filterbank() @ power
    return np.log(mel + 1e-10).astype(np.float32)


def logmel_from_pcm16(pcm: bytes, sr: int = SR) -> np.ndarray:
    """pcm: int16 mono bytes -> (40, 301) log-mel."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return logmel(x, sr)
