"""Mel-domain augmentation for VCM training (numpy only, no GPU recompute).

Why mel-domain and not waveform-domain:
  * The training fast-path reads precomputed log-mel features
    (data/features/*.npz) and never touches the audio files, so waveform
    augmentation (resample the wav, re-run log_mel) would force a full audio
    re-read per epoch. Mel-domain augmentation is a cheap per-frame op that
    keeps the fast path.
  * The transforms below are the standard, well-validated ones for CTC ASR:
    - pitch shift  : the single biggest speaker-diversity lever. Shifting the
                     log-frequency axis by a few semitones simulates different
                     voices without changing the words. This directly attacks
                     the overfit-to-training-speakers failure mode.
    - time stretch : speaking-rate invariance (resample the time axis).
    - gain         : level invariance (add a constant in the log1p domain).
    - time shift   : robustness to where the utterance sits in the window.
    - SpecAugment  : time + frequency masking (regularization).

Everything operates on a [T, 40] log-mel (log1p) array and returns the same
shape, so it drops straight into the existing collate().

All randomness is driven by numpy's global RNG, which train.py reseeds per
epoch (via random.seed + np.random.seed) so runs are reproducible.
"""
import numpy as np

from vcm.features import _get_mel, N_MELS, SR

_N_FFT = 512  # must match vcm.features.log_mel default


def _mel_centers():
    """Center frequency (Hz) of each of the 40 mel bands, ascending."""
    f = _get_mel(_N_FFT, SR)                 # [40, n_fft//2+1]
    bins = np.arange(f.shape[1]) * (SR / _N_FFT)
    num = (f * bins[None, :]).sum(axis=1)
    den = f.sum(axis=1) + 1e-9
    return (num / den).astype(np.float64)


_MEL_HZ = None


def mel_centers():
    global _MEL_HZ
    if _MEL_HZ is None:
        _MEL_HZ = _mel_centers()
    return _MEL_HZ


def _resample_freq_axis(mel, src_pos):
    """Gather mel[:, j] from fractional source positions src_pos[j] in [0, F-1]."""
    F = mel.shape[1]
    lo = np.floor(src_pos).astype(np.int64)
    hi = np.clip(lo + 1, 0, F - 1)
    frac = (src_pos - lo).astype(np.float32)[None, :]
    return mel[:, lo] * (1 - frac) + mel[:, hi] * frac


def _resample_time_axis(mel, src_pos):
    """Gather mel[t, :] from fractional source positions src_pos[t] in [0, T-1]."""
    T = mel.shape[0]
    lo = np.floor(src_pos).astype(np.int64)
    hi = np.clip(lo + 1, 0, T - 1)
    frac = (src_pos - lo).astype(np.float32)[:, None]
    return mel[lo, :] * (1 - frac) + mel[hi, :] * frac


def pitch_shift(mel, semitones):
    """Shift the frequency content by `semitones` (can be negative).

    Output bin j (center freq f_j) is filled from the source frequency
    f_j / 2^(st/12), interpolated on the mel grid. Output shape unchanged.
    """
    if abs(semitones) < 1e-3:
        return mel
    hz = mel_centers()
    src_hz = hz / (2.0 ** (semitones / 12.0))
    src_pos = np.interp(src_hz, hz, np.arange(len(hz)))
    return _resample_freq_axis(mel, src_pos).astype(np.float32)


def time_stretch(mel, factor):
    """Compress/expand the time axis by `factor` (1.0 = no change).

    factor > 1 -> faster speech (time compressed); output length preserved.
    """
    if abs(factor - 1.0) < 1e-3:
        return mel
    T = mel.shape[0]
    src = np.arange(T) * factor
    src = np.clip(src, 0, T - 1)
    return _resample_time_axis(mel, src).astype(np.float32)


def gain(mel, db):
    """Add a constant in the log1p domain == multiply power by 10^(db/10)."""
    if abs(db) < 1e-3:
        return mel
    c = db * np.log(10.0) / 10.0
    return (mel + c).astype(np.float32)


def time_shift(mel, frames):
    """Roll the time axis by `frames` (negative = earlier). Edges wrap."""
    if frames == 0:
        return mel
    return np.roll(mel, frames, axis=0).astype(np.float32)


def spec_augment(mel, n_time=1, n_freq=1, max_time_frac=0.2, max_freq_frac=0.25):
    """Mask out random time bands and frequency bands (set to 0)."""
    out = mel.copy()
    T, F = out.shape
    for _ in range(n_time):
        l = np.random.randint(1, max(2, int(T * max_time_frac) + 1))
        t0 = np.random.randint(0, max(1, T - l + 1))
        out[t0:t0 + l, :] = 0.0
    for _ in range(n_freq):
        l = np.random.randint(1, max(2, int(F * max_freq_frac) + 1))
        f0 = np.random.randint(0, max(1, F - l + 1))
        out[:, f0:f0 + l] = 0.0
    return out.astype(np.float32)


def augment_mel(mel, cfg=None):
    """Apply a random, mild augmentation to one [T,40] log-mel.

    cfg keys (all optional, defaults below):
        pitch_semitones : 3.0   uniform half-range for pitch shift
        time_stretch    : 0.10  uniform half-range for the stretch factor
        gain_db         : 3.0   uniform half-range for gain
        time_shift_frames: 8    uniform half-range for the time roll
        spec_aug_prob   : 0.5   probability of applying SpecAugment
        spec_aug_time   : 1     number of time masks
        spec_aug_freq   : 1     number of frequency masks
    """
    cfg = cfg or {}
    out = mel
    ps = cfg.get("pitch_semitones", 3.0)
    if ps > 0:
        out = pitch_shift(out, float(np.random.uniform(-ps, ps)))
    ts = cfg.get("time_stretch", 0.10)
    if ts > 0:
        out = time_stretch(out, float(np.random.uniform(1 - ts, 1 + ts)))
    gd = cfg.get("gain_db", 3.0)
    if gd > 0:
        out = gain(out, float(np.random.uniform(-gd, gd)))
    tf = int(cfg.get("time_shift_frames", 8))
    if tf > 0:
        out = time_shift(out, int(np.random.randint(-tf, tf + 1)))
    if np.random.rand() < cfg.get("spec_aug_prob", 0.5):
        out = spec_augment(out, n_time=int(cfg.get("spec_aug_time", 1)),
                           n_freq=int(cfg.get("spec_aug_freq", 1)))
    return out
