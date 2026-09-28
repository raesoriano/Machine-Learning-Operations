#!/usr/bin/env python3
"""Build the new-speaker training subset from `additional_test_data`.

The 171 clips are one NEW speaker (the one the ASR has never heard). We turn
them into a reusable training subset by generating, per clip, a set of
waveform variants that span the acoustic conditions the device will meet:

    clean    noisereduce denoise of the raw clip (the "cleaned denoised" version)
    hp80     4th-order Butterworth high-pass @ 80 Hz (removes the 0-100 Hz rumble)
    rumble   + low-frequency (0-200 Hz) noise, SNR 15 dB
    white    + white noise, SNR 20 dB
    babble   + other-speaker speech (babble), SNR 20 dB
    reverb   + synthetic room impulse response (RT60 ~ 0.4 s)
    pitch_up   +3 semitones (resample-resample)
    pitch_down -3 semitones
    stretch    +10% time stretch (faster)

The RAW clip itself is NEVER used for training -- it is held out as the
leakage-free test set (this is the whole point: we measure generalization to
the new speaker on audio the model has not seen in any form).

Outputs (under --out, default VCM-v2/data_new):
    wavs/<variant>/<folder>/<stem>.wav     the 171 x 9 = 1539 variant clips
    subset_manifest.csv                    one row per variant clip
                                            (source, folder, spoken, text, variant, split, path)
    features/train.npz  val.npz            precomputed log-mel + token ids,
                                            MERGED with the original ME2 splits:
                                              train = original train + 8 variants x 171
                                                      (clean held out of train)
                                              val   = original val   + clean x 171
                                                      (new-speaker val signal)
    build_report.json                      counts, per-variant stats, target check

The npz format matches the ME2 model/train.py fast path exactly
(ids / mel_data / mel_off / tok_data / tok_off).

Usage:
    python scripts/build_subset.py --src ../additional_test_data
"""
import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, sosfiltfilt, resample_poly

_HERE = os.path.dirname(os.path.abspath(__file__))      # .../VCM-v2/scripts
_VCM2 = os.path.dirname(_HERE)                          # .../VCM-v2
_SANDBOX = os.path.dirname(_VCM2)                       # .../sandbox
_ME2 = os.path.join(_SANDBOX, "Machine-Learning-Operations",
                    "ME2 - Voice Controlled Smart Device")
sys.path.insert(0, _VCM2)
sys.path.insert(0, _ME2)

from vcm.features import log_mel            # noqa: E402  (16 kHz [T,40])
from vcm2.ground_truth import (             # noqa: E402
    build_ground_truth, spoken_phrase_from_filename)
from model.train import tokenize            # noqa: E402  (exact training tokenizer)

SR = 16000

# OOV spoken phrases -> in-vocab canonical target. The CTC model can only emit
# words in the constrained vocab, so a target that tokenizes to empty (or drops
# an OOV word) is unlearnable. These two are the only such cases in the 171:
#   "end playback"  -> end/playback not in vocab (tokenizes empty)
#   "voluyme up"    -> 'voluyme' is a slip-of-the-tongue non-word
# We map them to the in-vocab command they mean. Documented in the README.
TARGET_FIX = {
    "end playback": "stop playing",
    "voluyme up": "volume up",
}


def _load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1).astype(np.float32)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def _norm(x):
    p = np.max(np.abs(x))
    return x / p if p > 0 else x


def v_clean(x, rng, src_dir):
    import noisereduce as nr
    return _norm(nr.reduce_noise(y=x, sr=SR, stationary=True, prop_decrease=0.85))


def v_hp80(x, rng, src_dir):
    sos = butter(4, 80.0, btype="high", fs=SR, output="sos")
    return _norm(sosfiltfilt(sos, x))


def _add_noise(x, noise, snr_db):
    px = np.mean(x ** 2) + 1e-12
    pn = np.mean(noise ** 2) + 1e-12
    n = noise * np.sqrt(px / pn * (10 ** (-snr_db / 10.0)))
    return _norm(x + n.astype(np.float32))


def v_rumble(x, rng, src_dir):
    n = np.random.default_rng(rng.integers(1, 2 ** 31)).standard_normal(len(x)).astype(np.float32)
    sos = butter(3, 200.0, btype="low", fs=SR, output="sos")
    return _add_noise(x, sosfilt(sos, n), 15.0)


def v_white(x, rng, src_dir):
    n = np.random.default_rng(rng.integers(1, 2 ** 31)).standard_normal(len(x)).astype(np.float32)
    return _add_noise(x, n, 20.0)


def v_babble(x, rng, src_dir, pool):
    # mix 2 random other-speaker clips as babble noise (wrap to fill length)
    r = np.random.default_rng(rng.integers(1, 2 ** 31))
    n = np.zeros(len(x), dtype=np.float32)
    for _ in range(2):
        p = pool[r.integers(len(pool))]
        y = _load(p)
        off = int(r.integers(0, max(1, len(y))))
        seg = np.concatenate([y[off:], y[:off]])
        while len(seg) < len(x):          # pool clip shorter than target
            seg = np.concatenate([seg, y])
        n += seg[:len(x)]
    return _add_noise(x, n, 20.0)


def v_reverb(x, rng, src_dir, rt60=0.4):
    r = np.random.default_rng(rng.integers(1, 2 ** 31))
    decay = int(rt60 * SR)
    ir = r.standard_normal(decay).astype(np.float32) * np.exp(
        -np.arange(decay) / (rt60 / 6.0 * SR))
    ir[0] += 1.0  # dry
    ir = ir / (np.linalg.norm(ir) + 1e-9)
    out = np.convolve(x, ir, mode="full")[:len(x)]
    return _norm(out)


def _pitch(x, semis):
    """Resample-resample pitch shift: changes pitch by `semis`, keeps duration.

    Raise pitch by factor f: stretch time by f (lower pitch), then compress
    back by f (restore duration; net pitch up by f). Lower pitch: reverse.
    """
    f = 2.0 ** (semis / 12.0)
    num, den = int(round(1000 * f)), 1000
    if semis > 0:
        y = resample_poly(x, num, den)      # stretch (lower pitch)
        y = resample_poly(y, den, num)      # compress (restore duration)
    else:
        y = resample_poly(x, den, num)      # compress (higher pitch)
        y = resample_poly(y, num, den)      # stretch (restore duration)
    return _norm(y[:len(x)] if len(y) >= len(x)
                 else np.pad(y, (0, len(x) - len(y))))


def v_pitch_up(x, rng, src_dir):
    return _pitch(x, +3.0)


def v_pitch_down(x, rng, src_dir):
    return _pitch(x, -3.0)


def v_stretch(x, rng, src_dir):
    # faster speech: compress time by 10%
    y = resample_poly(x, 10, 11)
    return _norm(y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y))))


VARIANTS = {
    "clean": v_clean, "hp80": v_hp80, "rumble": v_rumble, "white": v_white,
    "babble": v_babble, "reverb": v_reverb, "pitch_up": v_pitch_up,
    "pitch_down": v_pitch_down, "stretch": v_stretch,
}
# which variants go to train (clean is held out of train -> used as val only)
TRAIN_VARIANTS = ["hp80", "rumble", "white", "babble", "reverb",
                  "pitch_up", "pitch_down", "stretch"]


def _one_clip(args):
    """Generate all variants for one source clip. Returns list of rows."""
    src_path, folder, spoken, out_dir, pool = args
    stem = re.sub(r" \d+$", "", spoken)
    x = _load(src_path)
    rng = np.random.default_rng(abs(hash((folder, stem))) % (2 ** 32))
    rows = []
    for name, fn in VARIANTS.items():
        try:
            y = fn(x, rng, os.path.dirname(src_path), pool=pool)
        except TypeError:
            y = fn(x, rng, os.path.dirname(src_path))
        outp = os.path.join(out_dir, "wavs", name, folder, stem + ".wav")
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        sf.write(outp, y, SR)
        text = TARGET_FIX.get(spoken, spoken)
        rows.append({"source": os.path.basename(src_path), "folder": folder,
                     "spoken": spoken, "text": text, "variant": name,
                     "path": outp})
    return rows


def _feat_row(path, text):
    x = _load(path)
    mel = log_mel(x, sr=SR)
    toks = tokenize(text)
    return path, mel, toks


def _feat_row_wrapped(pt):
    return _feat_row(pt[0], pt[1])


def build_features(rows, split, out_dir, orig_npz):
    """Compute mel+toks for `rows` and MERGE with the original ME2 npz."""
    print(f"  computing features for {len(rows)} {split} rows ...")
    mels, toks, ids = [], [], []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for path, mel, tok in ex.map(_feat_row_wrapped,
                                     [(r["path"], r["text"]) for r in rows],
                                     chunksize=8):
            mels.append(mel)
            toks.append(tok)
            ids.append(os.path.basename(path))
    # merge with original
    d = np.load(orig_npz)
    o_ids = d["ids"].tolist()
    o_mel_off = d["mel_off"].tolist()
    o_tok_off = d["tok_off"].tolist()
    o_mel = d["mel_data"]
    o_tok = d["tok_data"]
    all_ids = o_ids + ids
    # rebuild mel_data
    new_mel = np.zeros((int(o_mel_off[-1]) + sum(m.shape[0] for m in mels), 40),
                       dtype=np.float32)
    new_mel[:int(o_mel_off[-1])] = o_mel
    mel_off = list(o_mel_off)
    base = int(o_mel_off[-1])
    for m in mels:
        base += m.shape[0]
        mel_off.append(base)
    # rebuild tok_data
    new_tok = np.zeros(int(o_tok_off[-1]) + sum(len(t) for t in toks), dtype=np.int32)
    new_tok[:int(o_tok_off[-1])] = o_tok
    tok_off = list(o_tok_off)
    tb = int(o_tok_off[-1])
    for t in toks:
        tb += len(t)
        tok_off.append(tb)
    np.savez_compressed(os.path.join(out_dir, "features", f"{split}.npz"),
                        ids=np.array(all_ids, dtype=d["ids"].dtype),
                        mel_data=new_mel, mel_off=np.array(mel_off),
                        tok_data=new_tok, tok_off=np.array(tok_off))
    print(f"  {split}.npz: {len(all_ids)} rows "
          f"(orig {len(o_ids)} + new {len(ids)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=os.path.join(_VCM2, "data_new"))
    ap.add_argument("--orig-features",
                    default=os.path.join(_ME2, "data", "features"))
    args = ap.parse_args()
    out = args.out
    os.makedirs(os.path.join(out, "features"), exist_ok=True)

    rows_gt = build_ground_truth(args.src)
    print(f"source clips: {len(rows_gt)}")
    pool = [r["path"] for r in rows_gt]

    # generate variants (parallel over clips)
    jobs = [(r["path"], r["folder"], r["spoken"], out, pool) for r in rows_gt]
    all_rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for res in ex.map(_one_clip, jobs, chunksize=4):
            all_rows.extend(res)
    print(f"variant clips: {len(all_rows)}")

    # assign splits
    for r in all_rows:
        r["split"] = "val" if r["variant"] == "clean" else "train"

    # write manifest
    man = os.path.join(out, "subset_manifest.csv")
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "folder", "spoken", "text",
                                          "variant", "split", "path"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"manifest -> {man}")

    # features
    train_rows = [r for r in all_rows if r["split"] == "train"]
    val_rows = [r for r in all_rows if r["split"] == "val"]
    build_features(train_rows, "train", out,
                   os.path.join(args.orig_features, "train.npz"))
    build_features(val_rows, "val", out,
                   os.path.join(args.orig_features, "val.npz"))

    # report
    per_variant = {}
    for r in all_rows:
        per_variant.setdefault(r["variant"], 0)
        per_variant[r["variant"]] += 1
    empty_targets = [r for r in all_rows if not tokenize(r["text"])]
    report = {
        "source_clips": len(rows_gt),
        "variant_clips": len(all_rows),
        "per_variant": per_variant,
        "train_new": len(train_rows),
        "val_new": len(val_rows),
        "empty_targets_after_fix": len(empty_targets),
        "target_fixes_applied": {k: sum(1 for r in all_rows
                                        if r["spoken"] == k)
                                 for k in TARGET_FIX},
        "note": "raw 171 clips held out (test); clean variant -> val; "
                "8 variants -> train",
    }
    with open(os.path.join(out, "build_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
