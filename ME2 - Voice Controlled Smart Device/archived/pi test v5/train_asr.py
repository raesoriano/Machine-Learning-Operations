#!/usr/bin/env python3
"""pi test v5 -- train the limited-vocabulary CTC ASR model.

Trains ASRModel on the in-domain (optionb) command clips only, using the
command transcripts as CTC targets.  Out-of-domain clips are NOT used for the
ASR (they are for the classifier's REJECT class).  A held-out validation split
is used for early stopping + WER reporting.

Outputs:
    models/asr.pt        -- torch checkpoint (best val)
    models/asr.onnx      -- ONNX export
    data/asr_report.json -- val WER / CER + sample decodes
"""
from __future__ import annotations
import os, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
import models as M


class FeatDS(Dataset):
    def __init__(self, X, words, wlen, idx):
        self.X = X[idx]; self.words = words[idx]; self.wlen = wlen[idx]
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (torch.from_numpy(self.X[i]),
                torch.from_numpy(self.words[i]),
                int(self.wlen[i]))


def collate(batch):
    X = torch.stack([b[0] for b in batch])
    words = torch.stack([b[1] for b in batch])
    wlen = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return X, words, wlen


def wer(ref, hyp):
    """Word error rate via Levenshtein edit distance (ref, hyp: word-index lists)."""
    r, h = list(ref), list(hyp)
    dp = list(range(len(h) + 1))
    for a in range(1, len(r) + 1):
        prev = dp[0]; dp[0] = a
        for b in range(1, len(h) + 1):
            cur = dp[b]
            dp[b] = min(dp[b] + 1, dp[b - 1] + 1, prev + (0 if r[a - 1] == h[b - 1] else 1))
            prev = cur
    return dp[len(h)] / max(1, len(r))


def eval_wer(model, dl, blank, device):
    model.eval()
    err = 0; nref = 0; n = 0
    with torch.no_grad():
        for Xb, wb, wl in dl:
            logits = model(Xb.to(device))
            dec = M.ctc_greedy_decode(logits, blank)
            for i in range(len(dec)):
                ref = wb[i, :wl[i]].tolist()
                if len(ref) == 0:
                    continue
                err += wer(ref, dec[i]) * len(ref)
                nref += len(ref); n += 1
    return err / max(1, nref), n


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    d = np.load(os.path.join(HERE, "data", "features.npz"), allow_pickle=True)
    w = np.load(os.path.join(HERE, "data", "words.npz"))
    vocab = json.load(open(os.path.join(HERE, "data", "vocab.json")))
    X, y, src, split = d["X"], d["y"], d["src"], d["split"]
    words, wlen, maxlen = w["words"], w["wlen"], int(w["maxlen"])
    blank = vocab["blank"]
    print(f"X {X.shape} | vocab {len(vocab['idx2word'])} | maxlen {maxlen} | device {device}")

    # in-domain only
    cmd = (src == 0)
    tr = np.where(cmd & (split == 0))[0]
    va = np.where(cmd & (split == 1))[0]
    te = np.where(cmd & (split == 2))[0]
    print(f"in-domain train/val/test = {len(tr)}/{len(va)}/{len(te)}")

    ds_tr = FeatDS(X, words, wlen, tr)
    ds_va = FeatDS(X, words, wlen, va)
    dl_tr = DataLoader(ds_tr, batch_size=128, shuffle=True, num_workers=4, collate_fn=collate)
    dl_va = DataLoader(ds_va, batch_size=256, shuffle=False, num_workers=4, collate_fn=collate)

    # Fixed feature normalization stats (in-domain train set) -- baked into
    # the model so train / ONNX / Pi inference all use the same scaling.
    Xtr = X[(src == 0) & (split == 0)]
    feat_mean = float(Xtr.mean())
    feat_std = float(Xtr.std())
    print(f"feat normalization: mean {feat_mean:.4f} std {feat_std:.4f}")

    model = M.ASRModel(n_mels=40, n_frames=X.shape[2], vocab=len(vocab["idx2word"]),
                       feat_mean=feat_mean, feat_std=feat_std).to(device)
    print(f"ASRModel params: {model.param_count():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)

    best_wer = 1.0; best_state = None; patience = 0
    for epoch in range(1, 41):
        model.train(); t0 = time.time(); run = 0.0
        for Xb, wb, wl in dl_tr:
            Xb = Xb.to(device); wb = wb.to(device); wl = wl.to(device)
            opt.zero_grad()
            logits = model(Xb)
            loss = M.ctc_loss(logits, wb, wl, blank)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += loss.item()
        sched.step()
        # val WER
        val_wer, n = eval_wer(model, dl_va, blank, device)
        dt = time.time() - t0
        print(f"epoch {epoch:2d} loss {run/len(dl_tr):.4f} valWER {val_wer:.3f} ({dt:.1f}s)", flush=True)
        if val_wer < best_wer - 0.005:
            best_wer = val_wer; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 12:
                print("early stop"); break

    model.load_state_dict(best_state)
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    torch.save({"state_dict": best_state, "vocab_size": len(vocab["idx2word"]),
                "n_mels": 40, "n_frames": X.shape[2], "blank": blank,
                "feat_mean": feat_mean, "feat_std": feat_std},
               os.path.join(HERE, "models", "asr.pt"))

    # test WER + sample decodes
    ds_te = FeatDS(X, words, wlen, te)
    dl_te = DataLoader(ds_te, batch_size=128, shuffle=False, num_workers=4, collate_fn=collate)
    model.eval()
    we = 0.0; nref = 0; samples = []
    i2w = {int(k): v for k, v in vocab["idx2word"].items()}
    with torch.no_grad():
        for Xb, wb, wl in dl_te:
            logits = model(Xb.to(device))
            dec = M.ctc_greedy_decode(logits, blank)
            for i in range(len(dec)):
                ref = wb[i, :wl[i]].tolist()
                if len(ref) == 0:
                    continue
                we += wer(ref, dec[i]) * len(ref)
                nref += len(ref)
                if len(samples) < 8:
                    samples.append({"ref": " ".join(i2w[t] for t in ref),
                                    "hyp": " ".join(i2w[t] for t in dec[i])})
    test_wer = we / max(1, nref)
    report = {"val_wer": best_wer, "test_wer": test_wer,
              "test_clips": len(te), "samples": samples}
    json.dump(report, open(os.path.join(HERE, "data", "asr_report.json"), "w"), indent=1)
    print(f"\nASR val WER {best_wer:.3f} | test WER {test_wer:.3f}")
    for s in samples:
        print(f"  ref: {s['ref']}\n  hyp: {s['hyp']}")

    # ONNX export (model must be on CPU for torch.onnx.export)
    try:
        model = model.cpu()
        model.eval()
        dummy = torch.randn(1, 40, X.shape[2], device="cpu")
        torch.onnx.export(model, (dummy,), os.path.join(HERE, "models", "asr.onnx"),
                          input_names=["mel"], output_names=["logits"],
                          dynamic_axes={"mel": {0: "B"}, "logits": {0: "B"}},
                          opset_version=14)
        print("exported models/asr.onnx", os.path.getsize(os.path.join(HERE, "models", "asr.onnx")), "bytes")
    except Exception as e:
        print("ONNX export failed:", e)


if __name__ == "__main__":
    main()
