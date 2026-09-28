#!/usr/bin/env python3
"""Fine-tune the VCM ASR (CTC) on the original ME2 data + the new-speaker
subset, starting from the me2_v6 checkpoint.

This is the "retrain including the new data" step. It reuses the EXACT
training machinery from the ME2 repo (model/train.py: same model, same
collate, same mel-domain augmentation, same CTC loss, same early-stopping
on val CTC loss) so the new model is directly comparable to me2_v6. The only
addition is initializing the weights from me2_v6/best.pt instead of from
scratch -- fine-tuning from the converged v6 is fast and low-regression,
which is what we want when adding one new speaker's 171 clips (x9 variants).

Inputs:
    --features  VCM-v2/data_new/features   (merged train.npz / val.npz)
    --init      me2_v6/best.pt             (starting weights)
    --config    me2_v6/config.yaml         (architecture + hyperparams)
    --out       VCM-v2/artifacts/asr/me2_v8

The config is persisted next to the checkpoint (config.yaml) so the VCM-v2
ASR wrapper (vcm2/asr.py) and any ONNX export can rebuild the exact model.

Usage:
    python scripts/train_asr_v8.py \
        --features data_new/features \
        --init  ../Machine-Learning-Operations/ME2 - Voice Controlled Smart Device/model/checkpoints/me2_v6/best.pt \
        --config ../Machine-Learning-Operations/ME2 - Voice Controlled Smart Device/model/checkpoints/me2_v6/config.yaml \
        --out artifacts/asr/me2_v8
"""
import argparse
import os
import sys
import time

import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))      # .../VCM-v2/scripts
_VCM2 = os.path.dirname(_HERE)                          # .../VCM-v2
_SANDBOX = os.path.dirname(_VCM2)                       # .../sandbox
_ME2 = os.path.join(_SANDBOX, "Machine-Learning-Operations",
                    "ME2 - Voice Controlled Smart Device")
if _ME2 not in sys.path:
    sys.path.insert(0, _ME2)

# Reuse the EXACT ME2 training machinery (no re-implementation drift).
from model.train import (                      # noqa: E402
    load_features, collate, eval_val_loss, _row_mel, _row_toks)
from model.model_def import VCMEncoder, ctc_loss  # noqa: E402
from vcm.augment import augment_mel            # noqa: E402

import random                                   # noqa: E402
import numpy as np                              # noqa: E402
import torch.nn as nn                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--init", required=True, help="starting checkpoint (.pt)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None,
                    help="override LR (fine-tuning usually wants a lower LR)")
    ap.add_argument("--patience", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.lr is not None:
        cfg["lr"] = args.lr
    device = torch.device(cfg.get("device", "cpu"))
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    (open(os.path.join(out, "config.yaml"), "w")).write(yaml.safe_dump(cfg))

    model = VCMEncoder(n_mels=cfg["n_mels"], channels=cfg["channels"],
                       blocks=cfg["blocks"], dropout=cfg.get("dropout", 0.1))
    sd = torch.load(args.init, map_location="cpu")
    model.load_state_dict(sd)
    model.to(device).train()
    print(f"model params: {model.count_params():,}  device={device}")
    print(f"init from: {os.path.basename(os.path.dirname(args.init))}/best.pt")

    train_rows = load_features(args.features, "train")
    val_rows = load_features(args.features, "val")
    print(f"train rows: {len(train_rows)}  val rows: {len(val_rows)}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    epochs = args.epochs or cfg["epochs"]
    bs = cfg["batch_size"]
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg.get("lr_factor", 0.5),
        patience=cfg.get("lr_patience", 6), min_lr=cfg.get("min_lr", 1e-5))
    patience = args.patience or cfg.get("patience", 24)
    do_aug = bool(cfg.get("augment", False))
    aug_cfg = cfg.get("augment_cfg", {})
    if do_aug:
        print(f"augmentation: ON {aug_cfg}")

    best = float("inf")
    bad = 0
    for ep in range(1, epochs + 1):
        random.seed(ep)
        np.random.seed(ep)
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
            evn = cfg.get("eval_n", 500)
            vloss = eval_val_loss(model, val_rows, device, n=evn)
            acc = eval_val_acc(model, val_rows, device, n=evn)
            msg += f"  val_ctc {vloss:.4f}  val_word_acc {acc * 100:.1f}%"
            if vloss < best:
                best = vloss
                bad = 0
                torch.save(model.state_dict(), os.path.join(out, "best.pt"))
                print(f"  saved best.pt (val_ctc {vloss:.4f})")
            else:
                bad += 1
                if bad >= patience:
                    print(f"  early stop @ ep {ep} (no val improvement "
                          f"for {patience} eps)")
                    break
            sched.step(vloss)
        print(msg)
    torch.save(model.state_dict(), os.path.join(out, "last.pt"))
    print(f"done -> {out}")


def eval_val_acc(model, rows, device, n=200):
    """Greedy word exact-match on a val sample (diagnostic, not the selector)."""
    from model.decode import greedy_decode
    from model.model_def import BLANK
    model.eval()
    rng = random.Random(0)
    sample = rng.sample(rows, min(n, len(rows)))
    correct = 0
    with torch.no_grad():
        for row in sample:
            mels = _row_mel(row)
            logits = model(torch.from_numpy(mels)[None].to(device))
            pred = greedy_decode(logits[0].cpu().numpy(), blank=BLANK)
            gold = _row_toks(row)
            if pred == gold:
                correct += 1
    model.train()
    return correct / max(len(sample), 1)


if __name__ == "__main__":
    main()
