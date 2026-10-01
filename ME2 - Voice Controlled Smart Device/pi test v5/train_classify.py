#!/usr/bin/env python3
"""pi test v5 -- train the phrase classifier (31 commands + REJECT).

The classifier is trained on the ASR's DECODED word sequences, so it sees
exactly the distribution it will get at inference (ASR errors included):

    in-domain (optionb) clip  ->  ASR decode  ->  command class
    out-of-domain (fsc/...)   ->  ASR decode  ->  REJECT

This is the cascade the user asked for: limited-vocab ASR (unknown -> <unk>)
then phrase classification including a real REJECT class.

Outputs:
    models/classify.pt      -- torch checkpoint (best val)
    models/classify.onnx    -- ONNX export
    data/decoded_words.npz  -- cached ASR decodes for every clip (for the
                                end-to-end test + the Pi's word cache)
    data/classify_report.json
"""
from __future__ import annotations
import os, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
import models as M


def decode_all(model, X, blank, device, bs=256):
    model.eval()
    out = np.zeros((len(X), 16), dtype=np.int64)
    wlen = np.zeros(len(X), dtype=np.int32)
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(device)
            dec = M.ctc_greedy_decode(model(xb), blank)
            for j, seq in enumerate(dec):
                seq = seq[:16]
                out[i + j, :len(seq)] = seq
                wlen[i + j] = len(seq)
    return out, wlen


class WordDS(Dataset):
    def __init__(self, words, wlen, y, idx):
        self.words = words[idx]; self.wlen = wlen[idx]; self.y = y[idx]
    def __len__(self): return len(self.words)
    def __getitem__(self, i):
        return (torch.from_numpy(self.words[i]), int(self.wlen[i]), int(self.y[i]))


def collate(batch):
    words = torch.stack([b[0] for b in batch])
    wlen = torch.tensor([b[1] for b in batch], dtype=torch.long)
    y = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return words, wlen, y


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    d = np.load(os.path.join(HERE, "data", "features.npz"), allow_pickle=True)
    vocab = json.load(open(os.path.join(HERE, "data", "vocab.json")))
    X, y, src, split = d["X"], d["y"], d["src"], d["split"]
    blank = vocab["blank"]
    n_classes = vocab["n_classes"]
    i2w = vocab["idx2word"]

    # load ASR
    ck = torch.load(os.path.join(HERE, "models", "asr.pt"), map_location="cpu")
    asr = M.ASRModel(n_mels=40, n_frames=X.shape[2], vocab=ck["vocab_size"],
                     feat_mean=ck.get("feat_mean", 0.0),
                     feat_std=ck.get("feat_std", 1.0)).to(device)
    asr.load_state_dict(ck["state_dict"]); asr.eval()
    print(f"loaded ASR (params {asr.param_count():,}); decoding {len(X)} clips ...")
    words, wlen = decode_all(asr, X, blank, device)
    np.savez_compressed(os.path.join(HERE, "data", "decoded_words.npz"),
                        words=words, wlen=wlen)
    print("  decoded -> data/decoded_words.npz")

    # quick decode quality
    cmd = (src == 0)
    print(f"  in-domain mean decoded len {wlen[cmd].mean():.2f} | "
          f"OOD mean decoded len {wlen[~cmd].mean():.2f}")

    tr = np.where(split == 0)[0]
    va = np.where(split == 1)[0]
    te = np.where(split == 2)[0]
    print(f"train/val/test = {len(tr)}/{len(va)}/{len(te)}")

    ds_tr = WordDS(words, wlen, y, tr)
    ds_va = WordDS(words, wlen, y, va)
    dl_tr = DataLoader(ds_tr, batch_size=128, shuffle=True, num_workers=4, collate_fn=collate)
    dl_va = DataLoader(ds_va, batch_size=256, shuffle=False, num_workers=4, collate_fn=collate)

    model = M.PhraseModel(vocab=len(i2w), embed=64, maxlen=16, num_classes=n_classes).to(device)
    print(f"PhraseModel params: {model.param_count():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)

    def eval_acc(m, dl):
        m.eval(); correct = 0; tot = 0
        rej = 0; ood = 0; ood_rej = 0; cmd_rej = 0
        with torch.no_grad():
            for words_b, wl, yb in dl:
                pred = m(words_b.to(device)).argmax(-1).cpu()
                correct += (pred == yb).sum().item(); tot += len(yb)
        return correct / max(1, tot)

    best = -1; best_state = None; patience = 0
    for epoch in range(1, 26):
        model.train(); run = 0.0
        for words_b, wl, yb in dl_tr:
            opt.zero_grad()
            logits = model(words_b.to(device))
            loss = nn.functional.cross_entropy(logits, yb.to(device))
            loss.backward()
            opt.step()
            run += loss.item()
        sched.step()
        acc = eval_acc(model, dl_va)
        if acc > best:
            best = acc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 6:
                break
        if epoch % 5 == 0 or patience == 0:
            print(f"epoch {epoch:2d} loss {run/len(dl_tr):.4f} valAcc {acc:.4f}", flush=True)

    model.load_state_dict(best_state)
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    torch.save({"state_dict": best_state, "vocab_size": len(i2w), "maxlen": 16,
                "num_classes": n_classes}, os.path.join(HERE, "models", "classify.pt"))

    # ---- detailed test evaluation ----
    model.eval()
    te_words = words[te]; te_wlen = wlen[te]; te_y = y[te]; te_src = src[te]
    preds = []
    with torch.no_grad():
        for i in range(0, len(te), 256):
            preds.append(model(torch.from_numpy(te_words[i:i+256]).to(device)).argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)

    REJ = vocab["class2idx"]["REJECT"]
    cmd_mask = te_src == 0
    ood_mask = te_src == 1
    cmd_acc = (preds[cmd_mask] == te_y[cmd_mask]).mean()
    false_rej = (preds[cmd_mask] == REJ).mean()
    ood_rej = (preds[ood_mask] == REJ).mean()
    # confusion among command classes (in-domain, excluding reject)
    import collections
    conf = collections.Counter()
    for p, t in zip(preds[cmd_mask], te_y[cmd_mask]):
        conf[(int(t), int(p))] += 1
    # per-class acc
    per_class = {}
    for c in vocab["classes"]:
        if c == "REJECT":
            continue
        ci = vocab["class2idx"][c]
        sel = (te_y[cmd_mask] == ci)
        if sel.sum() > 0:
            per_class[c] = round(float((preds[cmd_mask][sel] == ci).mean()), 3)

    report = {
        "val_acc": best,
        "test_in_domain_acc": round(float(cmd_acc), 4),
        "test_false_reject_rate": round(float(false_rej), 4),
        "test_ood_reject_rate": round(float(ood_rej), 4),
        "test_n_in_domain": int(cmd_mask.sum()),
        "test_n_ood": int(ood_mask.sum()),
        "per_class_acc": per_class,
    }
    json.dump(report, open(os.path.join(HERE, "data", "classify_report.json"), "w"), indent=1)
    print(f"\nClassifier val acc {best:.4f}")
    print(f"  in-domain acc {cmd_acc:.4f} | false-reject {false_rej:.4f} | OOD reject {ood_rej:.4f}")
    print("  per-class (worst 8):")
    for c, a in sorted(per_class.items(), key=lambda kv: kv[1])[:8]:
        print(f"    {c:30s} {a:.3f}")

    # ONNX export (model must be on CPU for torch.onnx.export)
    try:
        model = model.cpu()
        model.eval()
        dummy = torch.zeros(1, 16, dtype=torch.long)
        torch.onnx.export(model, (dummy,), os.path.join(HERE, "models", "classify.onnx"),
                          input_names=["words"], output_names=["logits"],
                          dynamic_axes={"words": {0: "B"}, "logits": {0: "B"}},
                          opset_version=14)
        print("exported models/classify.onnx",
              os.path.getsize(os.path.join(HERE, "models", "classify.onnx")), "bytes")
    except Exception as e:
        print("ONNX export failed:", e)


if __name__ == "__main__":
    main()
