"""Feature extraction: 16 kHz waveform -> 39-dim PocketSphinx-style cepstral
features.

This is a faithful, trimmed port of the PocketSphinx front-end
(`pocketsphinx/src/fe/fe_sigproc.c` + `fe_com.c`) so that the acoustic model
sees the *same* input shape the classic HMM/GMM was designed around:

    pre-emphasis (0.97)
    -> Hamming window, 25.6 ms frame / 10 ms hop
    -> 512-point FFT, power spectrum
    -> 26-band mel filterbank, 133.34 .. 6855.5 Hz
    -> log
    -> orthonormal DCT-II -> 13 cepstra (c0..c12)
    -> cepstral mean normalization (CMN, per utterance)
    -> delta + delta-of-delta (context 2)
    -> (T, 39)

The same pipeline runs at training time and on the Pi at runtime, so the
acoustic model sees identical inputs in both.
"""
from __future__ import annotations
import numpy as np
import scipy.io.wavfile as wavio

SR = 16000
N_CEP = 13          # cepstra kept (c0..c12)
N_NFILT = 26        # mel filters
N_FFT = 512
FREQ_LOW = 133.34   # Hz
FREQ_HIGH = 6855.497  # Hz
PRE_EMP = 0.97
WIN_S = 0.0256      # 25.6 ms
HOP_S = 0.010       # 10 ms
DELTA_CTX = 2

_FRAME = int(round(WIN_S * SR))    # 410
_HOP = int(round(HOP_S * SR))      # 160


def _hz2mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel2hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_filt: int,
                    f_low: float, f_high: float) -> np.ndarray:
    """Triangular mel filterbank (unit-area-ish), shape (n_filt, n_fft//2+1).

    Mirrors PocketSphinx's fe_com.c filterbank construction: mel-spaced points
    across [f_low, f_high], mapped to DFT bin indices, triangular weights.
    """
    n_freq = n_fft // 2 + 1
    mel_pts = np.linspace(_hz2mel(f_low), _hz2mel(f_high), n_filt + 2)
    hz_pts = _mel2hz(mel_pts)
    bins = np.round((n_fft + 1) * hz_pts / sr).astype(int)
    bins = np.clip(bins, 0, n_freq)
    fb = np.zeros((n_filt, n_freq), dtype=np.float64)
    for m in range(n_filt):
        l, c, r = int(bins[m]), int(bins[m + 1]), int(bins[m + 2])
        if c > l:
            fb[m, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[m, c:r] = (r - np.arange(c, r)) / (r - c)
    # normalize each filter to unit area (PocketSphinx does not strictly, but
    # this keeps the log energies on a stable scale); harmless for GMM training
    area = fb.sum(axis=1, keepdims=True)
    area[area == 0] = 1.0
    fb = fb / area
    return fb


_FB = None


def _get_fb() -> np.ndarray:
    global _FB
    if _FB is None:
        _FB = _mel_filterbank(SR, N_FFT, N_NFILT, FREQ_LOW, FREQ_HIGH)
    return _FB


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    """Linear-resample to 16 kHz (scipy.signal). No-op for 16 kHz input."""
    if sr == SR:
        return x
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr, SR)
    return resample_poly(x, SR // g, sr // g).astype(np.float64)


def read_wav(path: str) -> np.ndarray:
    """Read any wav -> mono float64 at 16 kHz, peak-normalized to [-0.9, 0.9]."""
    sr, data = wavio.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        x = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        x = (data.astype(np.float64) - 128.0) / 128.0
    else:
        x = data.astype(np.float64)
    x = _resample(x, sr)
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak * 0.9
    return x


def _dct2_orthonormal(x: np.ndarray, n_out: int) -> np.ndarray:
    """Orthonormal DCT-II. x: (T, M). Returns (T, n_out) coefficients c0..c(n-1).

        c0   = (1/sqrt(M)) * sum_n x[n]
        ck   = sqrt(2/M)   * sum_n x[n] cos(pi*k*(n+0.5)/M),  k>=1
    """
    T, M = x.shape
    n = np.arange(M)
    k = np.arange(n_out)
    # cos matrix: (n_out, M)
    cos = np.cos(np.pi * np.outer(k, (n + 0.5)) / M)
    c = x @ cos.T                       # (T, n_out)
    scale = np.full(n_out, np.sqrt(2.0 / M))
    scale[0] = np.sqrt(1.0 / M)
    c = c * scale[None, :]
    return c


def _delta(feat: np.ndarray, context: int = DELTA_CTX) -> np.ndarray:
    n = feat.shape[0]
    denom = 2.0 * sum(i * i for i in range(1, context + 1))
    d = np.zeros_like(feat)
    for t in range(n):
        num = np.zeros_like(feat[0])
        for i in range(1, context + 1):
            if t - i >= 0:
                num += i * (feat[t - i] * -1.0)
            if t + i < n:
                num += i * feat[t + i]
        d[t] = num / denom
    return d


def cepstral(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """(T, 13) cepstral features (pre-CMN) from a 16 kHz mono signal."""
    if x.dtype != np.float64:
        x = x.astype(np.float64)
    if len(x) < _FRAME:
        x = np.pad(x, (0, _FRAME - len(x)))
    # pre-emphasis
    y = np.concatenate(([x[0]], x[1:] - PRE_EMP * x[:-1]))
    n_frames = 1 + (len(y) - _FRAME) // _HOP
    idx = np.arange(_FRAME)[None, :] + _HOP * np.arange(n_frames)[:, None]
    frames = y[idx]
    frames *= np.hamming(_FRAME)
    spec = np.abs(np.fft.rfft(frames, N_FFT, axis=1)) ** 2
    fb = _get_fb()
    mel = spec @ fb.T
    mel = np.log(mel + 1e-10)
    cep = _dct2_orthonormal(mel, N_CEP)
    return cep.astype(np.float64)


def features(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """Full feature vector: (T, 39) = [cep(13), delta(13), delta2(13)]."""
    c = cepstral(x, sr)
    # cepstral mean normalization (per utterance, over frames)
    c = c - c.mean(axis=0, keepdims=True)
    d = _delta(c)
    dd = _delta(d)
    return np.concatenate([c, d, dd], axis=1).astype(np.float32)


def features_from_wav(path: str) -> np.ndarray:
    return features(read_wav(path))
