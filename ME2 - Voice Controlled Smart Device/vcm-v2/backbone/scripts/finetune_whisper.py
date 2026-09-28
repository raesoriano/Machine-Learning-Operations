#!/usr/bin/env python3
"""Fine-tune a PRETRAINED Whisper (openai/whisper-base.en) on the ME2 command
domain, seq2seq teacher-forcing.

Why: zero-shot whisper base.en already hits 81.9% command accuracy on the
held-out 171 new-speaker clips (vs 24.6% for the from-scratch CTC). Its
remaining errors are (a) domain-word mishearings (PAUSE->'boss', STOP->'and
labor', TIME->'dang') and (b) digit/percent forms the classifier rejects
('alarm 8am', 'brightness 100%'). Fine-tuning on the ME2 transcripts teaches
it the command vocabulary and the canonical text form the stage-2 classifier
was trained on.

LEAKAGE-FREE: trains on backbone/data/{train,val}.jsonl (ME2 optionb
positives only, built with --no-newspk). The 171 additional_test_data clips
(raw + every denoised/noise/reverb/pitch variant) are NEVER in train or val,
so the held-out test set is genuinely unseen.

Input: 16 kHz waveform -> WhisperFeatureExtractor (padded to the fixed
      30 s / 3000-mel-frame Whisper input; attention mask from wav length)
Target: the ME2 transcript (the spoken words, e.g. 'alarm six am' / 'Alarm 6 AM')

Usage:
    CUDA_VISIBLE_DEVICES=0 python backbone/scripts/finetune_whisper.py \
        --epochs 5 --batch-size 32 --lr 1e-5 \
        --out backbone/artifacts/whisper_base_ft
"""
import argparse
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoTokenizer, WhisperForConditionalGeneration,
                          WhisperFeatureExtractor)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACK = os.path.dirname(_HERE)
_REPO = os.path.dirname(_BACK)

MAX_SECONDS = 4.0          # clips are ~2 s; cap at 4 s
SR = 16000


class WavText(Dataset):
    def __init__(self, jsonl_path, max_samples):
        self.max_samples = max_samples
        self.rows = []
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                if r["text"].strip():
                    self.rows.append(r)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x, sr = sf.read(r["audio"], dtype="float32")
        x = x.mean(axis=1) if x.ndim > 1 else x
        if sr != SR:
            import torchaudio
            x = torchaudio.functional.resample(
                torch.from_numpy(x), sr, SR).numpy().astype(np.float32)
        if len(x) > self.max_samples:
            x = x[:self.max_samples]
        return x, r["text"].strip()


def collate(batch, feature_extractor, tokenizer):
    wavs, texts = zip(*batch)
    # Whisper's encoder REQUIRES a fixed 30 s input (3000 mel frames);
    # the extractor pads to 30 s. It no longer returns an attention mask
    # (transformers >= 5), so build one from the waveform lengths and pass
    # it as attention_mask so the model can ignore the padding.
    inputs = feature_extractor(
        list(wavs), sampling_rate=SR, return_tensors="pt",
        padding="max_length")
    n = len(wavs)
    mask = torch.zeros(n, 3000, dtype=torch.long)
    for i, w in enumerate(wavs):
        n_frames = min(int(round(len(w) / SR * 50)), 3000)
        mask[i, :n_frames] = 1
    inputs["attention_mask"] = mask
    labels = tokenizer(texts, return_tensors="pt",
                       padding=True)["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    return inputs, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.path.join(_BACK, "data", "train.jsonl"))
    ap.add_argument("--val", default=os.path.join(_BACK, "data", "val.jsonl"))
    ap.add_argument("--out", default=os.path.join(_BACK, "artifacts",
                                                  "whisper_base_ft"))
    ap.add_argument("--model", default="openai/whisper-base.en")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    print(f"loading {args.model} ...", flush=True)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    max_samples = int(MAX_SECONDS * SR)
    train_ds = WavText(args.train, max_samples)
    val_ds = WavText(args.val, max_samples)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=lambda b: collate(b, feature_extractor,
                                                       tokenizer),
                          num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: collate(b, feature_extractor,
                                                     tokenizer),
                        num_workers=4, pin_memory=True)
    print(f"train rows: {len(train_ds)}  val rows: {len(val_ds)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    def run_epoch(dl, training):
        model.train(training)
        total, n, t0 = 0.0, 0, time.time()
        for inputs, labels in dl:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            out = model(input_features=inputs["input_features"],
                        attention_mask=inputs["attention_mask"],
                        labels=labels)
            loss = out.loss
            if training:
                opt.zero_grad()
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            if torch.isfinite(loss):
                total += loss.item() * len(labels)
                n += len(labels)
        return total / max(n, 1), time.time() - t0

    best, bad, best_ep = float("inf"), 0, 0
    history = []
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_t = run_epoch(train_dl, True)
        val_loss, _ = run_epoch(val_dl, False)
        history.append({"epoch": ep, "train_loss": round(tr_loss, 4),
                        "val_loss": round(val_loss, 4),
                        "seconds": round(tr_t, 0)})
        print(f"ep {ep:02d}  train {tr_loss:.4f}  val {val_loss:.4f}  "
              f"({tr_t:.0f}s)", flush=True)
        if val_loss < best - 1e-4:
            best, bad, best_ep = val_loss, 0, ep
            model.save_pretrained(os.path.join(args.out, "best"))
            feature_extractor.save_pretrained(os.path.join(args.out, "best"))
            tokenizer.save_pretrained(os.path.join(args.out, "best"))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at ep {ep} (best ep {best_ep}, "
                      f"val {best:.4f})", flush=True)
                break
    model.save_pretrained(os.path.join(args.out, "last"))
    feature_extractor.save_pretrained(os.path.join(args.out, "last"))
    tokenizer.save_pretrained(os.path.join(args.out, "last"))
    feature_extractor.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    report = {"model": args.model, "epochs_run": len(history),
              "best_epoch": best_ep, "best_val_loss": round(best, 4),
              "history": history,
              "hyperparams": {k: getattr(args, k) for k in
                              ("epochs", "batch_size", "lr", "patience",
                               "seed")},
              "train_rows": len(train_ds), "val_rows": len(val_ds),
              "note": "LEAKAGE-FREE: ME2 optionb positives only "
                      "(--no-newspk); additional_test_data never in train/val"}
    with open(os.path.join(args.out, "train_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved -> {args.out}  (best ep {best_ep}, val {best:.4f})")


if __name__ == "__main__":
    main()
