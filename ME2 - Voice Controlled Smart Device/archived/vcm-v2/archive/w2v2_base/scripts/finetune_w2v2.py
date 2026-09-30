#!/usr/bin/env python3
"""Fine-tune a pretrained wav2vec2 backbone with CTC over the constrained
ME2 command vocabulary.

Why this (vs the from-scratch 422k CTC in archive/ctc_v8):
    The tiny CTC underfits even its own in-domain data (47.7% WER on the ME2
    test split). A pretrained wav2vec2-base-960h (94M params, 960k hours of
    general speech) already knows English acoustics; we only retrain the
    output head (and fine-tune the encoder) to emit the 1033-word constrained
    command vocabulary. Same constrained-vocab CTC loss, same stage-2
    classifier, same held-out 171-clip new-speaker test set -> comparable.

Design:
    * backbone: facebook/wav2vec2-base-960h (AutoModelForCTC)
    * head:     nn.Linear(hidden=768, 1034) -- 1033 vocab words + CTC blank(0)
    * targets:  ME2 `tokenize()` word ids (1-based, blank=0) -- the EXACT
                same tokenizer the CTC pipeline used, so the stage-2
                classifier and the comparison are unchanged
    * data:     backbone/data/{train,val}.jsonl (raw 16 kHz wav + transcript)
                train = ME2 train positives + 8-variant new-speaker subset
                val   = ME2 val positives   + 171 clean new-speaker clips
                (the 171 RAW clips are the held-out test set, never trained)
    * augment:  random gain (+-6 dB) + random time shift (+-200 ms) on the
                waveform (the mel-domain augmentation the CTC pipeline used)
    * loss:     nn.CTCLoss (zero_infinity)
    * early stop on val CTC loss (patience 3)

Output (backbone/artifacts/w2v2_base/):
    model.safetensors   fine-tuned weights (fp32, ~376 MB, git-lfs)
    config.json         architecture config
    vocab.json          word-id mapping (1-based) + blank
    train_report.json   hyperparams + best/last val metrics

Usage:
    python backbone/scripts/finetune_w2v2.py --epochs 4 --batch-size 32
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCTC, AutoProcessor

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACK = os.path.dirname(_HERE)
_REPO = os.path.dirname(_BACK)
_SANDBOX = os.path.dirname(_REPO)
_ME2 = os.path.join(_SANDBOX, "Machine-Learning-Operations",
                    "ME2 - Voice Controlled Smart Device")
sys.path.insert(0, _ME2)

from vcm.vocab import VOCAB, WORD2ID  # noqa: E402

BACKBONE = "facebook/wav2vec2-base-960h"
SR = 16000
BLANK = 0
VOCAB_SIZE = len(VOCAB) + 1  # 186
MAX_SECONDS = 10.0


class Wav2Text(Dataset):
    def __init__(self, jsonl_path, max_samples):
        self.rows = []
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                self.rows.append((r["audio"], r["text"]))
        self.max_samples = max_samples

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        audio, text = self.rows[i]
        x, sr = sf.read(audio, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != SR:
            import torchaudio
            x = torchaudio.functional.resample(
                torch.from_numpy(x), sr, SR).numpy().astype(np.float32)
        if len(x) > self.max_samples:
            off = random.randint(0, len(x) - self.max_samples)
            x = x[off:off + self.max_samples]
        return x, text


def collate(batch, max_samples):
    """waveforms -> padded [B, T] + padded target ids [B, L] + lengths."""
    texts = [b[1] for b in batch]
    from model.train import tokenize
    targets = [torch.tensor(tokenize(t), dtype=torch.long) for t in texts]
    wavs = [torch.from_numpy(b[0]) for b in batch]
    T = max(w.shape[0] for w in wavs)
    L = max(t.shape[0] for t in targets)
    wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
    padded = torch.zeros(len(targets), L, dtype=torch.long)
    for i, t in enumerate(targets):
        padded[i, :t.shape[0]] = t
    return wavs, padded, torch.tensor([w.shape[0] for w in wavs]), \
        torch.tensor([t.shape[0] for t in targets])


def augment_waveform(x):
    """Random gain +-6 dB, random time shift +-200 ms (zero-padded)."""
    if random.random() < 0.5:
        db = random.uniform(-6.0, 6.0)
        x = x * (10.0 ** (db / 20.0))
    shift = random.randint(-int(0.2 * SR), int(0.2 * SR))
    z = torch.zeros
    if shift > 0:
        x = torch.cat([z(shift, device=x.device), x])[:len(x)]
    elif shift < 0:
        x = torch.cat([x[-shift:], z(-shift, device=x.device)])
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.path.join(_BACK, "data", "train.jsonl"))
    ap.add_argument("--val", default=os.path.join(_BACK, "data", "val.jsonl"))
    ap.add_argument("--out", default=os.path.join(_BACK, "artifacts", "w2v2_base"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="encoder LR (3e-5 stalled, 3e-4 destroyed; 1e-4 is the "
                         "standard moderate wav2vec2 fine-tune regime)")
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5,
                    help="AdamW default 0.01 is 1000x too high for fine-tuning")
    ap.add_argument("--warmup-steps", type=int, default=1000,
                    help="linear warmup steps before cosine decay")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--head-only-epochs", type=int, default=1,
                    help="epochs to train ONLY the head (encoder frozen) "
                         "before full fine-tune; stabilizes the random head")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    print(f"loading backbone {BACKBONE} ...")
    model = AutoModelForCTC.from_pretrained(BACKBONE)
    # ROOT-CAUSE FIX (the one that unblocks training): the base-960h CTC
    # checkpoint does NOT ship `wav2vec2.masked_spec_embed` (that parameter
    # only exists in the *pretraining* model). Under low_cpu_mem_usage it
    # materializes as UNINITIALIZED memory (garbage up to ~1e38, often NaN).
    # `_init_weights` has no branch for this bare nn.Parameter, so it is never
    # re-initialized. SpecAugment (mask_time_prob=0.05) writes it into masked
    # positions -> encoder activations blow up -> NaN loss. Re-init to a small
    # normal. (Verified: baseline masked_spec_embed max=nan -> loss nan;
    #  re-init max=0.067 -> loss finite & decreasing, 0 NaN grads.)
    with torch.no_grad():
        nn.init.normal_(model.wav2vec2.masked_spec_embed, mean=0.0, std=0.02)
    model.lm_head = nn.Linear(model.config.hidden_size, VOCAB_SIZE)
    # Zero-init the head: a from-scratch head has logits ~N(0, 768)
    # (std ~27) which drives CTC to inf/NaN within a few dozen steps.
    # Zero init -> all-zero logits -> uniform posterior -> finite, stable
    # start (the standard wav2vec2 fine-tune recipe).
    nn.init.zeros_(model.lm_head.weight)
    nn.init.zeros_(model.lm_head.bias)
    n_enc = sum(p.numel() for n, p in model.named_parameters()
                if not n.startswith("lm_head"))
    n_head = sum(p.numel() for p in model.lm_head.parameters())
    print(f"params: encoder {n_enc:,}  head {n_head:,}  device={device}")
    model.to(device)
    # Gradient checkpointing: trade ~25% compute for ~4x less activation
    # memory (wav2vec2's encoder activations are large at batch 16-32).
    model.gradient_checkpointing_enable()

    max_samples = int(MAX_SECONDS * SR)
    train_ds = Wav2Text(args.train, max_samples)
    val_ds = Wav2Text(args.val, max_samples)
    print(f"train rows: {len(train_ds)}  val rows: {len(val_ds)}")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=lambda b: collate(b, max_samples),
                          num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: collate(b, max_samples),
                        num_workers=4, pin_memory=True)

    head_ids = {id(p) for p in model.lm_head.parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in head_ids],
         "lr": args.lr},
        {"params": list(model.lm_head.parameters()), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    # Linear warmup -> cosine decay. The constant-LR run (lr=3e-5) never
    # adapted the encoder (val CTC stuck ~243 = near-uniform) and the head
    # collapsed to the marginal word prior. Warmup + decay at the standard
    # 3e-4 encoder LR is the proven wav2vec2 fine-tune recipe.
    steps_per_epoch = len(train_dl)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup = max(1, args.warmup_steps)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    enc_params = [p for p in model.parameters() if id(p) not in head_ids]
    def freeze_encoder(frozen):
        for p in enc_params:
            p.requires_grad = not frozen

    def run_epoch(dl, training):
        model.train(training)
        total, n = 0.0, 0
        t0 = time.time()
        # NOTE: fp32 training (no autocast). ctc_loss on GPU is fragile
        # under fp16 (inf logits -> NaN -> illegal memory access), and the
        # A100 handles fp32 fine at this scale.
        for wavs, targets, ilens, tlens in dl:
                wavs = wavs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                ilens = ilens.to(device, non_blocking=True)
                tlens = tlens.to(device, non_blocking=True)
                if training:
                    wavs = torch.stack([augment_waveform(w) for w in wavs])
                # CTC input_lengths must be the LOGIT-frame length, not the
                # raw waveform length. wav2vec2 provides the exact mapping.
                ilens = model._get_feat_extract_output_lengths(ilens)
                logits = model(wavs).logits
                # ctc_loss_gpu crashes under fp16 autocast -> compute in fp32
                logp = torch.log_softmax(logits.float(), dim=-1)
                logp = logp.permute(2, 0, 1).contiguous()  # [T, B, V]
                # F.ctc_loss reads input_lengths[i] frames; if any value
                # exceeds the padded logit length T the GPU kernel faults
                # (illegal memory access). Clamp to T (a no-op when already
                # within bounds, a guard when the length map overshoots).
                ilens = ilens.clamp(max=logp.size(0))
                loss = ctc(logp, targets, ilens, tlens)
                if training:
                    opt.zero_grad()
                    if torch.isfinite(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        opt.step()
                        sched.step()
                    else:
                        print("  [warn] non-finite loss, skipping batch",
                              flush=True)
                if torch.isfinite(loss):
                    total += loss.item() * len(wavs)
                    n += len(wavs)
        return total / max(n, 1), time.time() - t0

    best, bad, best_ep = float("inf"), 0, 0
    history = []
    for ep in range(1, args.epochs + 1):
        frozen = ep <= args.head_only_epochs
        freeze_encoder(frozen)
        # gradient checkpointing only in the full (encoder-unfrozen)
        # phase; the head-only warmup has no activation-memory pressure
        if frozen:
            model.gradient_checkpointing_disable()
        else:
            model.gradient_checkpointing_enable()
        tr_loss, tr_t = run_epoch(train_dl, training=True)
        val_loss, _ = run_epoch(val_dl, training=False)
        history.append({"epoch": ep, "train_loss": round(tr_loss, 4),
                        "val_loss": round(val_loss, 4),
                        "seconds": round(tr_t, 0),
                        "phase": "head" if frozen else "full"})
        print(f"ep {ep:02d} [{'head' if frozen else 'full'}]  "
              f"train {tr_loss:.4f}  val {val_loss:.4f}  ({tr_t:.0f}s)",
              flush=True)
        if val_loss < best - 1e-4:
            best, bad, best_ep = val_loss, 0, ep
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at ep {ep} (best ep {best_ep}, "
                      f"val {best:.4f})")
                break
    torch.save(model.state_dict(), os.path.join(out, "last.pt"))

    # ---- persist a loadable model (config + weights + vocab) ----
    # Update the config's vocab_size so from_pretrained rebuilds the new
    # head shape (the original wav2vec2 config says 32).
    model.config.vocab_size = VOCAB_SIZE
    sd = torch.load(os.path.join(out, "best.pt"), map_location="cpu")
    model.load_state_dict(sd)
    model.save_pretrained(out)  # config.json + model.safetensors
    with open(os.path.join(out, "vocab.json"), "w") as f:
        json.dump({"blank": BLANK, "vocab": VOCAB,
                   "word2id": WORD2ID}, f, indent=1)
    report = {
        "backbone": BACKBONE, "vocab_size": VOCAB_SIZE,
        "n_vocab_words": len(VOCAB), "max_seconds": MAX_SECONDS,
        "epochs_run": len(history), "best_epoch": best_ep,
        "best_val_ctc": round(best, 4), "history": history,
        "hyperparams": {k: getattr(args, k) for k in
                        ("epochs", "batch_size", "lr", "head_lr", "patience",
                         "seed", "head_only_epochs", "warmup_steps")},
        "train_rows": len(train_ds), "val_rows": len(val_ds),
    }
    with open(os.path.join(out, "train_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved -> {out}  (best ep {best_ep}, val ctc {best:.4f})")


if __name__ == "__main__":
    main()
