#!/usr/bin/env python3
"""Fine-tune v2: pretrained wav2vec2 backbone, CTC over the constrained ME2
command vocabulary -- improved recipe over scripts/finetune_w2v2.py.

What changed vs the v1 recipe (and why):

1. Cosine LR schedule with linear warmup (was: constant 3e-5).
   The v1 run plateaued at val CTC ~243-245 from epoch 2 to 15 -- a constant
   LR neither converges the head fast enough nor anneals the encoder.
   Cosine annealing lets the head learn the vocab quickly (warmup + high
   start) and then settle the encoder (low end).

2. Lower encoder LR (1e-5, was 3e-5). The backbone already knows English
   acoustics; 3e-5 for 15 epochs on ~38k short clips overwrites pretrained
   features faster than the task can exploit them. 1e-5 + cosine is the
   standard wav2vec2 fine-tune regime.

3. More epochs (default 40, patience 10). v1 early-stopped at ep 15 of 20
   with the loss still moving; the model was undertrained (smoke test
   emitted incoherent word streams).

4. Stronger data augmentation (was: gain +-6 dB + time shift only):
     * pitch shift +-3 semitones (50%) -- speaker robustness
     * time stretch 0.9-1.1 (30%) -- speaker robustness
     * waveform SpecAugment: 1 time mask + 1 freq mask (50%) -- robustness
     * gain +-8 dB (70%), time shift +-200 ms (always)
   All applied on the raw waveform before the feature extractor.

5. Early stopping on val WER (greedy CTC decode) instead of val CTC loss.
   CTC loss is a poor proxy for decode quality on constrained vocabs; WER
   on the val split is the actual objective we care about.

Unchanged (proven to be required):
   * masked_spec_embed re-init (NaN root cause, see finetune_w2v2.py)
   * zero-init head
   * ilens clamp to logit length
   * fp32, gradient checkpointing, head-only warmup epoch

Usage:
    CUDA_VISIBLE_DEVICES=7 python backbone/scripts/finetune_w2v2_v2.py \
        --out backbone/artifacts/w2v2_v2 [--train X --val Y]
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
from transformers import AutoModelForCTC

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../backbone/scripts
_BACK = os.path.dirname(_HERE)                              # .../backbone
_REPO = os.path.dirname(_BACK)                              # .../VCM-v2
_SANDBOX = os.path.dirname(_REPO)                           # .../sandbox
_ME2 = os.path.join(_SANDBOX, "Machine-Learning-Operations",
                    "ME2 - Voice Controlled Smart Device")
if _ME2 not in sys.path:
    sys.path.insert(0, _ME2)

from vcm.vocab import VOCAB, WORD2ID  # noqa: E402
from model.decode import decode_to_text  # noqa: E402
from model.train import tokenize  # noqa: E402

BACKBONE = "facebook/wav2vec2-base-960h"
SR = 16000
MAX_SECONDS = 10.0
VOCAB_SIZE = len(VOCAB) + 1  # + blank
BLANK = 0


def _edit_distance(a, b):
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n]


class Wav2Text(Dataset):
    def __init__(self, path, max_samples):
        self.rows, self.max_samples = [], max_samples
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                self.rows.append((r["audio"], r["text"]))

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
    texts = [b[1] for b in batch]
    targets = [torch.tensor(tokenize(t), dtype=torch.long) for t in texts]
    wavs = [torch.from_numpy(b[0]) for b in batch]
    wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
    L = max(t.shape[0] for t in targets)
    padded = torch.zeros(len(targets), L, dtype=torch.long)
    for i, t in enumerate(targets):
        padded[i, :t.shape[0]] = t
    return wavs, padded, torch.tensor([w.shape[0] for w in wavs]), \
        torch.tensor([t.shape[0] for t in targets])


def _spec_augment_wav(x):
    """Waveform SpecAugment (1 time mask + 1 freq mask) via torch STFT."""
    win = torch.hann_window(2048, periodic=False, device=x.device)
    stft = torch.stft(x[None], n_fft=2048, hop_length=512, win_length=2048,
                      window=win, return_complex=True)
    F, T = stft.shape[1], stft.shape[2]
    t0 = random.randint(0, max(T - 1, 0))
    tlen = random.randint(1, max(1, T // 8))
    stft[:, :, t0:min(t0 + tlen, T)] = 0
    f0 = random.randint(0, max(F - 1, 0))
    flen = random.randint(1, max(1, F // 8))
    stft[:, f0:min(f0 + flen, F), :] = 0
    y = torch.istft(stft, n_fft=2048, hop_length=512, win_length=2048,
                    window=win, length=x.shape[0])
    return y.squeeze(0)


def _match_len(x, n, z):
    if x.shape[0] < n:
        x = torch.cat([x, z(n - x.shape[0], device=x.device)])
    else:
        x = x[:n]
    return x


def augment_waveform(x):
    """Stronger augmentation than v1 (gain +-6 dB + shift only).

    Runs on the DEVICE tensor (GPU): torchaudio 2.11 pitch_shift (new
    signature: waveform, sample_rate, n_steps), double-resample time
    stretch, and a manual STFT SpecAugment. CPU versions of these are
    ~30 s/batch (measured) -- GPU is the only viable option.
    """
    n = x.shape[0]
    if random.random() < 0.7:
        db = random.uniform(-8.0, 8.0)
        x = x * (10.0 ** (db / 20.0))
    shift = random.randint(-int(0.2 * SR), int(0.2 * SR))
    z = torch.zeros
    if shift > 0:
        x = torch.cat([z(shift, device=x.device), x])[:n]
    elif shift < 0:
        x = torch.cat([x[-shift:], z(-shift, device=x.device)])
    if random.random() < 0.5:  # pitch shift +-3 semitones
        import torchaudio
        semis = random.choice([-3, -2, -1, 1, 2, 3])
        xp = torchaudio.functional.pitch_shift(x[None], SR, semis).squeeze(0)
        x = _match_len(xp, n, z)
    if random.random() < 0.3:  # time stretch 0.9-1.1 (double resample)
        import torchaudio
        # Discrete factors so 16000*s stays integer (resample requirement).
        f = random.choice([14400, 15200, 16800, 17600])  # 0.9/0.95/1.05/1.1
        xt = torchaudio.functional.resample(x[None], SR, f).squeeze(0)
        xt = torchaudio.functional.resample(xt[None], f, SR).squeeze(0)
        x = _match_len(xt, n, z)
    if random.random() < 0.5:  # waveform SpecAugment
        x = _match_len(_spec_augment_wav(x), n, z)
    return x


def val_wer(model, dl, device, max_clips=2000):
    """Greedy-decoded WER on a (capped) val subset. The real objective."""
    model.eval()
    dist, ref_n, hyp_n, n = 0, 0, 0, 0
    with torch.no_grad():
        for wavs, targets, ilens, tlens in dl:
            if n >= max_clips:
                break
            wavs = wavs.to(device)
            logits = model(wavs).logits
            ilens = model._get_feat_extract_output_lengths(ilens).clamp(
                max=logits.size(2))
            for i in range(len(wavs)):
                hyp = decode_to_text(
                    torch.tensor(logits[i, :ilens[i]]).cpu())
                ref = " ".join(VOCAB[t - 1] for t in
                               targets[i, :tlens[i]].tolist())
                dist += _edit_distance(ref.split(), hyp.split())
                ref_n += len(ref.split())
                hyp_n += len(hyp.split())
                n += 1
    return dist / max(ref_n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.path.join(_BACK, "data", "train.jsonl"))
    ap.add_argument("--val", default=os.path.join(_BACK, "data", "val.jsonl"))
    ap.add_argument("--out", default=os.path.join(_BACK, "artifacts", "w2v2_v2"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5,
                    help="encoder LR (cosine from here to 1% of it)")
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--head-only-epochs", type=int, default=1)
    ap.add_argument("--val-wer-clips", type=int, default=2000)
    ap.add_argument("--resume", default=None,
                    help="path to a best.pt/last.pt to resume from "
                         "(model weights only; schedule restarts)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    print(f"loading backbone {BACKBONE} ...")
    model = AutoModelForCTC.from_pretrained(BACKBONE)
    # NaN root-cause fix (see finetune_w2v2.py docstring).
    with torch.no_grad():
        nn.init.normal_(model.wav2vec2.masked_spec_embed, mean=0.0, std=0.02)
    model.lm_head = nn.Linear(model.config.hidden_size, VOCAB_SIZE)
    nn.init.zeros_(model.lm_head.weight)
    nn.init.zeros_(model.lm_head.bias)
    n_enc = sum(p.numel() for n, p in model.named_parameters()
                if not n.startswith("lm_head"))
    n_head = sum(p.numel() for p in model.lm_head.parameters())
    print(f"params: encoder {n_enc:,}  head {n_head:,}  device={device}")
    if args.resume:
        sd = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(sd)
        print(f"resumed weights from {args.resume}")
    model.to(device)

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
    enc_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr},
        {"params": list(model.lm_head.parameters()), "lr": args.head_lr},
    ])
    steps_per_epoch = len(train_dl)
    total_steps = max(steps_per_epoch * args.epochs, 1)
    warmup_steps = max(int(total_steps * args.warmup_frac), 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    def freeze_encoder(frozen):
        for p in enc_params:
            p.requires_grad = not frozen

    def run_epoch(dl, training):
        model.train(training)
        total, n = 0.0, 0
        t0 = time.time()
        for wavs, targets, ilens, tlens in dl:
            wavs = wavs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            ilens = ilens.to(device, non_blocking=True)
            tlens = tlens.to(device, non_blocking=True)
            if training:
                wavs = torch.stack([augment_waveform(w) for w in wavs])
            ilens = model._get_feat_extract_output_lengths(ilens)
            logits = model(wavs).logits
            logp = torch.log_softmax(logits.float(), dim=-1)
            logp = logp.permute(2, 0, 1).contiguous()
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

    best_wer, bad, best_ep = float("inf"), 0, 0
    history = []
    for ep in range(1, args.epochs + 1):
        frozen = ep <= args.head_only_epochs
        freeze_encoder(frozen)
        if frozen:
            model.gradient_checkpointing_disable()
        else:
            model.gradient_checkpointing_enable()
        tr_loss, tr_t = run_epoch(train_dl, training=True)
        val_loss, _ = run_epoch(val_dl, training=False)
        vwer = val_wer(model, val_dl, device, args.val_wer_clips)
        history.append({"epoch": ep, "train_loss": round(tr_loss, 4),
                        "val_loss": round(val_loss, 4),
                        "val_wer": round(vwer, 4),
                        "lr": round(opt.param_groups[0]["lr"], 8),
                        "seconds": round(tr_t, 0),
                        "phase": "head" if frozen else "full"})
        print(f"ep {ep:02d} [{'head' if frozen else 'full'}]  "
              f"train {tr_loss:.4f}  val {val_loss:.4f}  "
              f"val_wer {vwer:.4f}  lr {opt.param_groups[0]['lr']:.2e}  "
              f"({tr_t:.0f}s)", flush=True)
        if vwer < best_wer - 1e-4:
            best_wer, bad, best_ep = vwer, 0, ep
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at ep {ep} (best ep {best_ep}, "
                      f"val_wer {best_wer:.4f})")
                break
    torch.save(model.state_dict(), os.path.join(out, "last.pt"))

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
        "best_val_wer": round(best_wer, 4), "history": history,
        "hyperparams": {k: getattr(args, k) for k in
                        ("epochs", "batch_size", "lr", "head_lr",
                         "warmup_frac", "patience", "seed",
                         "head_only_epochs", "val_wer_clips")},
        "train_rows": len(train_ds), "val_rows": len(val_ds),
        "train_file": os.path.abspath(args.train),
        "val_file": os.path.abspath(args.val),
    }
    with open(os.path.join(out, "train_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved -> {out}  (best ep {best_ep}, val_wer {best_wer:.4f})")


if __name__ == "__main__":
    main()
