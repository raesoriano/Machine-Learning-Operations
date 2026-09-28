#!/usr/bin/env python3
"""ME2 in-domain regression guardrail for the fine-tuned Whisper.

Transcribes a random sample of the ME2 TEST split (original speakers,
never used for the backbone fine-tune) with the fine-tuned model and
computes WER against the manifest transcripts. Confirms the fine-tune
did not degrade in-domain ASR.

Usage:
    CUDA_VISIBLE_DEVICES=0 python backbone/scripts/eval_whisper_ft_me2.py \
        --model backbone/artifacts/whisper_base_ft/best \
        --n 400 --seed 42 \
        --report backbone/reports/me2_regression_whisper_ft.json
"""
import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch
from transformers import (AutoTokenizer, WhisperForConditionalGeneration,
                          WhisperFeatureExtractor)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACK = os.path.dirname(_HERE)
_REPO = os.path.dirname(_BACK)
_SANDBOX = os.path.dirname(os.path.dirname(_ME2))  # .../sandbox
_ME2 = os.path.dirname(_REPO)  # vcm-v2 lives inside the ME2 folder
sys.path.insert(0, _ME2)

MANIFEST = os.path.join(_ME2, "data", "manifests",
                        "positive_negative_manifest.csv")
DATA_ROOT = os.path.join(_ME2, "data")


def wer(ref, hyp):
    """Word-level Levenshtein WER (0..1+)."""
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[0, :] = np.arange(len(h) + 1)
    d[:, 0] = np.arange(len(r) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + cost)
    return d[len(r), len(h)] / len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(_BACK, "artifacts",
                                                    "whisper_base_ft",
                                                    "best"))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report", default=os.path.join(_BACK, "reports",
                                                     "me2_regression_"
                                                     "whisper_ft.json"))
    args = ap.parse_args()

    rows = []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["polarity"] != "positive" or r["split"] != "test":
                continue
            audio = r["audio"]
            if not os.path.isabs(audio):
                audio = os.path.join(DATA_ROOT, audio)
            if r["transcript"].strip() and os.path.isfile(audio):
                rows.append((audio, r["transcript"].strip()))
    random.seed(args.seed)
    sample = random.sample(rows, min(args.n, len(rows)))
    print(f"ME2 test positives: {len(rows)}  sampled: {len(sample)}",
          flush=True)

    device = args.device
    print(f"loading fine-tuned whisper from {args.model} ...", flush=True)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.to(device)
    model.eval()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        args.model if os.path.isfile(os.path.join(args.model,
                                                  "preprocessor_config.json"))
        else os.path.dirname(os.path.abspath(args.model)))
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    t0 = time.perf_counter()
    wer_sum, n, exact = 0.0, 0, 0
    for k, (path, ref) in enumerate(sample):
        a, sr = sf.read(path, dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != 16000:
            import torchaudio
            a = torchaudio.functional.resample(
                torch.from_numpy(a), sr, 16000).numpy().astype(np.float32)
        inputs = feature_extractor(a, sampling_rate=16000,
                                   return_tensors="pt",
                                   padding="max_length").to(device)
        # The extractor no longer returns an attention mask; build it from
        # the waveform length (50 mel frames/s, 3000-frame max).
        n_frames = min(int(round(len(a) / 16000 * 50)), 3000)
        mask = torch.zeros(1, 3000, dtype=torch.long)
        mask[0, :n_frames] = 1
        mask = mask.to(device)
        with torch.no_grad():
            ids = model.generate(inputs.input_features,
                                 attention_mask=mask,
                                 max_new_tokens=64,
                                 do_sample=False)
        if ids[0, 0] == tokenizer.pad_token_id:
            ids = ids[:, 1:]
        hyp = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        w = wer(ref, hyp)
        wer_sum += w
        n += 1
        if w == 0:
            exact += 1
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(sample)}  "
                  f"running WER {wer_sum / (k + 1):.4f}  "
                  f"({(time.perf_counter() - t0) / (k + 1):.2f}s/clip)",
                  flush=True)

    report = {
        "model": "whisper base.en FINE-TUNED (HF, ME2 domain)",
        "split": "ME2 test (original speakers, in-domain guardrail)",
        "n_rows": n,
        "wer": round(wer_sum / n, 4),
        "word_exact_match": round(exact / n, 4),
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
