#!/usr/bin/env python3
"""Fine-tune Whisper base.en on the ME2 command domain -- v2 (staged).

Why v2 exists: the v1 fine-tune (finetune_whisper.py, single LR 1e-5,
6 epochs, early-stopped at epoch 3 on CE loss) was DEGENERATE. CE loss
looked fine (val 0.131) but the model had learned to hallucinate:
  - held-out 171 new-speaker clips:  WER 10.24 (garbage streams)
  - ME2 test guardrail (400 clips):  WER 5.25, exact match 6%
  - zero-shot (no fine-tune) baseline: WER 0.225, 81.9% command acc
CE is a terrible early-stopping signal for seq2seq ASR: a model can
drive CE down by emitting high-probability filler while still being
useless. v2 therefore:

  (1) more training with a proper LR schedule:
      Stage A (warm-start, encoder FROZEN): 3 epochs, decoder only,
              LR 3e-5 -- teaches the head the domain text forms without
              disturbing the pretrained encoder.
      Stage B (encoder UNFROZEN): up to 12 more epochs, encoder LR 1e-5,
              decoder LR 3e-5, cosine schedule with 6% linear warmup.
  (2) early stopping on VALIDATION WER (300 sampled clips, greedy
      generation) instead of CE -- patience 3. Best checkpoint = lowest
      val WER.
  (3) bf16 mixed precision for speed.

LEAKAGE-FREE: identical data to v1 -- backbone/data/{train,val}.jsonl
(ME2 optionb positives only, built with --no-newspk). The 171
additional_test_data clips (raw + every variant) are NEVER in train or
val, so the held-out test set is genuinely unseen.

Usage:
    CUDA_VISIBLE_DEVICES=0 python backbone/scripts/finetune_whisper_v2.py \
        --out backbone/artifacts/whisper_base_ft_v2
"""
import argparse
import json
import os
import random
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoTokenizer, WhisperForConditionalGeneration,
                          WhisperFeatureExtractor,
                          get_cosine_schedule_with_warmup)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACK = os.path.dirname(_HERE)

MAX_SECONDS = 4.0          # clips are ~2 s; cap at 4 s
SR = 16000
VAL_WER_N = 300            # val clips sampled for the WER early-stop signal


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
    # Whisper's encoder REQUIRES a fixed 30 s input (3000 mel frames).
    inputs = feature_extractor(
        list(wavs), sampling_rate=SR, return_tensors="pt",
        padding="max_length")
    n = len(wavs)
    mask = torch.zeros(n, 3000, dtype=torch.long)
    for i, w in enumerate(wavs):
        n_frames = min(int(round(len(w) / SR * 50)), 3000)
        mask[i, :n_frames] = 1
    inputs["attention_mask"] = mask
    labels = tokenizer(texts, return_tensors="pt", padding=True)["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    return inputs, labels


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
    ap.add_argument("--train", default=os.path.join(_BACK, "data",
                                                    "train.jsonl"))
    ap.add_argument("--val", default=os.path.join(_BACK, "data", "val.jsonl"))
    ap.add_argument("--out", default=os.path.join(_BACK, "artifacts",
                                                  "whisper_base_ft_v2"))
    ap.add_argument("--model", default="openai/whisper-base.en")
    ap.add_argument("--stage-a-epochs", type=int, default=3)
    ap.add_argument("--stage-b-epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-dec", type=float, default=3e-5)
    ap.add_argument("--lr-enc", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    print(f"loading {args.model} ...", flush=True)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}",
          flush=True)

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

    # Fixed val subset for the WER early-stop signal (greedy generation).
    random.Random(args.seed).shuffle(val_ds.rows)
    val_wer_rows = val_ds.rows[:VAL_WER_N]
    print(f"val-WER early-stop sample: {len(val_wer_rows)} clips", flush=True)

    history = []

    def run_epoch(dl, training, opt):
        model.train(training)
        total, n, t0 = 0.0, 0, time.time()
        for inputs, labels in dl:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=training):
                out = model(input_features=inputs["input_features"],
                            attention_mask=inputs["attention_mask"],
                            labels=labels)
                loss = out.loss
            if training:
                opt.zero_grad()
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        1.0)
                    opt.step()
            if torch.isfinite(loss):
                total += loss.item() * len(labels)
                n += len(labels)
        return total / max(n, 1), time.time() - t0

    @torch.no_grad()
    def val_wer():
        """Greedy-generation WER on the fixed val subset (0..1+)."""
        model.eval()
        s, n = 0.0, 0
        for r in val_wer_rows:
            x, sr = sf.read(r["audio"], dtype="float32")
            x = x.mean(axis=1) if x.ndim > 1 else x
            if sr != SR:
                import torchaudio
                x = torchaudio.functional.resample(
                    torch.from_numpy(x), sr, SR).numpy().astype(np.float32)
            if len(x) > max_samples:
                x = x[:max_samples]
            inputs = feature_extractor(x, sampling_rate=SR,
                                       return_tensors="pt",
                                       padding="max_length").to(device)
            n_frames = min(int(round(len(x) / SR * 50)), 3000)
            mask = torch.zeros(1, 3000, dtype=torch.long)
            mask[0, :n_frames] = 1
            mask = mask.to(device)
            ids = model.generate(inputs.input_features, attention_mask=mask,
                                 max_new_tokens=64, do_sample=False)
            if ids[0, 0] == tokenizer.pad_token_id:
                ids = ids[:, 1:]
            hyp = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
            s += wer(r["text"].strip(), hyp)
            n += 1
        model.train(True)
        return s / max(n, 1)

    # ---------------- Stage A: encoder frozen, decoder only ----------------
    for p in model.model.encoder.parameters():
        p.requires_grad = False
    opt_a = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr_dec, weight_decay=0.0)
    print(f"=== STAGE A: encoder frozen, decoder LR {args.lr_dec}, "
          f"{args.stage_a_epochs} epochs ===", flush=True)

    best_wer, bad, best_ep = float("inf"), 0, 0
    for ep in range(1, args.stage_a_epochs + 1):
        tr_loss, tr_t = run_epoch(train_dl, True, opt_a)
        val_loss, _ = run_epoch(val_dl, False, opt_a)
        vw = val_wer()
        history.append({"epoch": ep, "stage": "A",
                        "train_loss": round(tr_loss, 4),
                        "val_loss": round(val_loss, 4),
                        "val_wer": round(vw, 4),
                        "seconds": round(tr_t, 0)})
        print(f"ep {ep:02d} [A]  train {tr_loss:.4f}  val {val_loss:.4f}  "
              f"valWER {vw:.4f}  ({tr_t:.0f}s)", flush=True)
        if vw < best_wer - 1e-4:
            best_wer, bad, best_ep = vw, 0, ep
            model.save_pretrained(os.path.join(args.out, "best"))
            feature_extractor.save_pretrained(os.path.join(args.out, "best"))
            tokenizer.save_pretrained(os.path.join(args.out, "best"))
            print(f"  -> saved best (valWER {vw:.4f})", flush=True)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop (stage A)", flush=True)
                break

    # ---------------- Stage B: encoder unfrozen, cosine LR ----------------
    for p in model.parameters():
        p.requires_grad = True
    enc_params, dec_params = [], []
    for name, p in model.named_parameters():
        (enc_params if name.startswith("model.encoder") else dec_params
         ).append(p)
    opt_b = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr_enc},
        {"params": dec_params, "lr": args.lr_dec},
    ], weight_decay=0.0)
    steps_b = args.stage_b_epochs * len(train_dl)
    sched_b = get_cosine_schedule_with_warmup(
        opt_b, num_warmup_steps=int(0.06 * steps_b),
        num_training_steps=steps_b)
    print(f"=== STAGE B: encoder unfrozen (enc {args.lr_enc} / dec "
          f"{args.lr_dec}, cosine), up to {args.stage_b_epochs} epochs ===",
          flush=True)
    for ep in range(1, args.stage_b_epochs + 1):
        model.train(True)
        total, n, t0 = 0.0, 0, time.time()
        for inputs, labels in train_dl:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                out = model(input_features=inputs["input_features"],
                            attention_mask=inputs["attention_mask"],
                            labels=labels)
                loss = out.loss
            opt_b.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt_b.step()
                sched_b.step()
            if torch.isfinite(loss):
                total += loss.item() * len(labels)
                n += len(labels)
        tr_loss, tr_t = total / max(n, 1), time.time() - t0
        val_loss, _ = run_epoch(val_dl, False, opt_b)
        vw = val_wer()
        history.append({"epoch": ep + args.stage_a_epochs, "stage": "B",
                        "train_loss": round(tr_loss, 4),
                        "val_loss": round(val_loss, 4),
                        "val_wer": round(vw, 4),
                        "seconds": round(tr_t, 0)})
        print(f"ep {ep:02d} [B]  train {tr_loss:.4f}  val {val_loss:.4f}  "
              f"valWER {vw:.4f}  ({tr_t:.0f}s)", flush=True)
        if vw < best_wer - 1e-4:
            best_wer, bad, best_ep = vw, 0, ep + args.stage_a_epochs
            model.save_pretrained(os.path.join(args.out, "best"))
            feature_extractor.save_pretrained(os.path.join(args.out, "best"))
            tokenizer.save_pretrained(os.path.join(args.out, "best"))
            print(f"  -> saved best (valWER {vw:.4f})", flush=True)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop (stage B)", flush=True)
                break

    model.save_pretrained(os.path.join(args.out, "last"))
    feature_extractor.save_pretrained(os.path.join(args.out, "last"))
    tokenizer.save_pretrained(os.path.join(args.out, "last"))

    report = {
        "model": "whisper base.en FINE-TUNED v2 (staged: frozen-encoder "
                 "warm-start -> unfrozen cosine)",
        "base": args.model,
        "best_epoch": best_ep,
        "best_val_wer": round(best_wer, 4),
        "history": history,
        "hyperparams": {
            "stage_a_epochs": args.stage_a_epochs,
            "stage_b_epochs": args.stage_b_epochs,
            "batch_size": args.batch_size,
            "lr_dec": args.lr_dec, "lr_enc": args.lr_enc,
            "patience": args.patience, "seed": args.seed,
        },
        "train_rows": len(train_ds), "val_rows": len(val_ds),
        "note": "LEAKAGE-FREE: ME2 optionb positives only (--no-newspk); "
                "additional_test_data never in train/val. Early stop on "
                "val WER (300 clips), not CE.",
    }
    with open(os.path.join(args.out, "train_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in
                      ("best_epoch", "best_val_wer")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
