"""Train the tiny VCM (CTC over the constrained command vocab).

Runs on a laptop/GPU once the dataset manifest is ready (WP3). The code is
complete and tested for shape/logic; it just needs real audio rows.

Input manifest (JSONL, from data/generate): one row per utterance
    {"id": ..., "audio": "abs/path.wav", "text": "set a timer for 5 minutes",
     "intent": ..., "slots": {...}}
The transcript `text` is tokenized with vcm.vocab (constrained: every word is
in VOCAB by construction of the grammar).

Usage:
    python -m model.train --manifest data/manifests/train_aug.jsonl \
        --val-manifest data/manifests/val.jsonl \
        --config model/configs/tiny.yaml --out model/checkpoints/tiny

Outputs: best.pt (torch) + a small eval report. Export to ONNX afterwards
with deploy/export_onnx.py.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from model.decode import greedy_decode  # noqa: E402
from vcm.augment import augment_mel  # noqa: E402
from model.model_def import VCMEncoder, ctc_loss, BLANK  # noqa: E402
from vcm.features import log_mel  # noqa: E402
from vcm.vocab import to_ids, VOCAB  # noqa: E402


def load_manifest(path, max_rows=None):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if max_rows:
        rows = rows[:max_rows]
    return rows


def load_features(fdir, name):
    """Load precomputed {name}.npz (ids/mel_data/mel_off/tok_data/tok_off)
    into rows with _mel/_toks so training never touches the audio files."""
    d = np.load(os.path.join(fdir, f"{name}.npz"))
    ids, mel_data, mel_off = d["ids"], d["mel_data"], d["mel_off"]
    tok_data, tok_off = d["tok_data"], d["tok_off"]
    rows = []
    for i in range(len(ids)):
        rows.append({
            "id": str(ids[i]), "audio": None, "text": "",
            "_mel": mel_data[mel_off[i]:mel_off[i + 1]],
            "_toks": tok_data[tok_off[i]:tok_off[i + 1]].tolist(),
        })
    return rows


def tokenize(text):
    """transcript -> 1-based token ids (constrained vocab).

    Mirrors ``vcm.parser._normalize`` (contractions, punctuation) and expands
    Arabic digits to number-words, since the constrained vocab only contains
    number-words. Unknown words are dropped (constrained decoding).
    """
    from vcm.parser import _normalize
    from vcm.numbers import int_to_words
    toks = _normalize(text).split()
    toks = [int_to_words(int(t)) if t.isdigit() and int(t) <= 120 else t for t in toks]
    ids = [i + 1 for i in to_ids(toks) if i >= 0]
    return ids


def _row_mel(row):
    m = row.get("_mel")
    return m if m is not None else log_mel(_load_audio(row["audio"]))


def _row_toks(row):
    t = row.get("_toks")
    return t if t is not None else tokenize(row["text"])


def collate(batch, max_frames=400):
    """Pad a list of (mels[T,40], target_ids) to a batch."""
    mels = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    T = min(max(m.shape[0] for m in mels), max_frames)
    B = len(mels)
    X = np.zeros((B, T, mels[0].shape[1]), dtype=np.float32)
    ilens, tlist, tlens = [], [], []
    for i, (m, t) in enumerate(zip(mels, targets)):
        X[i, : min(T, m.shape[0])] = m[:T]
        ilens.append(min(T, m.shape[0]))
        tlist.append(t)
        tlens.append(len(t))
    L = max(tlens) if tlens else 1
    Y = np.zeros((B, L), dtype=np.int32)
    for i, t in enumerate(tlist):
        Y[i, : len(t)] = t
    return (torch.from_numpy(X), torch.from_numpy(Y),
            torch.tensor(ilens, dtype=torch.long),
            torch.tensor(tlens, dtype=torch.long))


def eval_word_acc(model, rows, device, n=200):
    """Quick greedy-decode accuracy on a sample (for early stopping)."""
    model.eval()
    correct = 0
    rng = random.Random(0)
    sample = rng.sample(rows, min(n, len(rows)))
    with torch.no_grad():
        for row in sample:
            mels = _row_mel(row)
            logits = model(torch.from_numpy(mels)[None].to(device))
            pred = greedy_decode(logits[0].cpu().numpy(), blank=BLANK)
            gold = _row_toks(row)
            if pred == gold:
                correct += 1
    return correct / max(len(sample), 1)


def _load_audio(path):
    """Load wav/flac -> float32 mono at 16000 Hz (log_mel's native rate).

    Every dataset in the manifest is 16 kHz, so this is a no-op on the real
    pipeline; the resample is defensive for stray files only.
    """
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        g = __import__("math").gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x


def eval_val_loss(model, rows, device, n=200):
    """Mean CTC loss on a val sample (model-selection signal).

    Strict exact-match is the wrong selector for CTC: greedy decode is blank
    for most of training, so exact-match stays 0.0 and a `best = -1.0` init
    freezes best.pt at epoch 1. CTC loss descends monotonically and is the
    standard, reliable signal.
    """
    model.eval()
    rng = random.Random(0)
    sample = rng.sample(rows, min(n, len(rows)))
    total, count = 0.0, 0
    with torch.no_grad():
        for row in sample:
            mels = _row_mel(row)
            t = _row_toks(row)
            if not t:
                continue
            X = torch.from_numpy(mels)[None].to(device)
            Y = torch.tensor([t], dtype=torch.int32, device=device)
            il = torch.tensor([mels.shape[0]], dtype=torch.long, device=device)
            tl = torch.tensor([len(t)], dtype=torch.long, device=device)
            total += ctc_loss(model(X), Y, il, tl).item()
            count += 1
    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--val-manifest", default=None)
    ap.add_argument("--features", default=None,
                    help="dir with precomputed {train,val}.npz features "
                         "(fast path; skips --manifest audio I/O)")
    ap.add_argument("--config", default=str(_REPO / "model/configs/tiny.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None,
                    help="debug: cap rows per split")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg.get("device", "cpu"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Persist the config next to the checkpoint so the benchmark torch backend
    # and deploy/export_onnx can rebuild the exact architecture.
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))

    model = VCMEncoder(
        n_mels=cfg["n_mels"], channels=cfg["channels"],
        blocks=cfg["blocks"], dropout=cfg.get("dropout", 0.1)).to(device)
    print(f"model params: {model.count_params():,}  device={device}")

    if args.features:
        train_rows = load_features(args.features, "train")
        val_rows = load_features(args.features, "val")
    else:
        train_rows = load_manifest(args.manifest, args.max_rows)
        val_rows = load_manifest(args.val_manifest, args.max_rows) if args.val_manifest else []
    print(f"train rows: {len(train_rows)}  val rows: {len(val_rows)}")

    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    epochs = args.epochs or cfg["epochs"]
    bs = cfg["batch_size"]
    # ReduceLROnPlateau (not fixed cosine): a cosine T_max=200 decays too
    # slowly, so early-stopping at ep 25-77 kills the run while the LR is
    # still 68-96% of its peak and the train loss is still falling. Dropping
    # the LR on val plateau lets the model keep fitting the data.
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg.get("lr_factor", 0.5),
        patience=cfg.get("lr_patience", 6), min_lr=cfg.get("min_lr", 1e-5))
    patience = cfg.get("patience", 24)
    # Mel-domain augmentation (on-the-fly, per epoch). Attacks the
    # overfit-to-training-speakers failure mode; see vcm/augment.py.
    do_aug = bool(cfg.get("augment", False))
    aug_cfg = cfg.get("augment_cfg", {})
    if do_aug:
        print(f"augmentation: ON {aug_cfg}")
    best = float("inf")  # best val CTC loss (lower is better)
    bad = 0
    for ep in range(1, epochs + 1):
        random.shuffle(train_rows)
        model.train()
        t0 = time.time()
        total = 0
        for i in range(0, len(train_rows), bs):
            batch = train_rows[i:i + bs]
            prepared = []
            for row in batch:
                mels = _row_mel(row)
                if do_aug:
                    mels = augment_mel(mels, aug_cfg)
                prepared.append((mels, _row_toks(row)))
            X, Y, ilens, tlens = collate(prepared)
            X, Y, ilens, tlens = (X.to(device), Y.to(device),
                                  ilens.to(device), tlens.to(device))
            logits = model(X)
            loss = ctc_loss(logits, Y, ilens, tlens)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(batch)
        avg = total / max(len(train_rows), 1)
        msg = f"ep {ep:02d} loss {avg:.4f} ({time.time() - t0:.0f}s)"
        if val_rows and ep % cfg.get("eval_every", 1) == 0:
            evn = cfg.get("eval_n", 200)
            vloss = eval_val_loss(model, val_rows, device, n=evn)
            acc = eval_word_acc(model, val_rows, device, n=evn)
            msg += f"  val_ctc {vloss:.4f}  val_word_acc {acc * 100:.1f}%"
            if vloss < best:
                best = vloss
                bad = 0
                torch.save(model.state_dict(), out / "best.pt")
                print(f"  saved best.pt (val_ctc {vloss:.4f})")
            else:
                bad += 1
                if bad >= patience:
                    print(f"  early stop @ ep {ep} (no val improvement "
                          f"for {patience} eps)")
                    break
            sched.step(vloss)
        print(msg)
    torch.save(model.state_dict(), out / "last.pt")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
