#!/usr/bin/env python3
"""pi test v5 -- data prep (torch-free; numpy + scipy only).

Reads the OptionB manifest (Voice-Recognition-Mark-Dataset/manifest.csv),
builds the limited-vocabulary word set from the in-domain command transcripts,
extracts log-Mel features (feats.py -- the SAME pipeline the Pi uses), and
writes a compact training bundle:

    data/features.npz   -- {X: (N,40,301), y: (N,) class idx, src: (N,) 0=cmd 1=ood,
                           split: (N,) 0=train 1=val 2=test, path: (N,) str}
    data/vocab.json     -- {"words": [...], "word2idx": {...}, "idx2word": {...},
                           "n_classes": 32, "classes": [...]}
    data/words.npz      -- {words: (N, maxlen) word idx, wlen: (N,) true len}

In-domain (source==optionb) clips carry one of the 31 command classes.
Out-of-domain (fsc/slurp/librispeech) free-text clips are the REJECT class.

A per-(class,split) subsample keeps the file reads tractable (set V5_MAX_CMD /
V5_MAX_OOD).  read + features run in a single torch-free process pool, so there
is no fork-with-torch deadlock.
"""
from __future__ import annotations
import os, sys, csv, json, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # so `import feats` resolves
import feats as F                                   # numpy/scipy log-Mel

MD = "/home/ron.andrei.soriano/sandbox/Voice-Recognition-Mark-Dataset"
# OOD free-text clips (fsc/librispeech/slurp) live under the ME2 repo root
# (their manifest paths are prefixed with "data/external/..."); the in-domain
# optionb clips live under the dataset root (MD).
ME2 = "/home/ron.andrei.soriano/sandbox/Machine-Learning-Operations/ME2 - Voice Controlled Smart Device"
MANIFEST = os.path.join(MD, "manifest.csv")

BLANK = "<blank>"
UNK = "<unk>"

# 31 in-domain command classes (fixed order = the v3/v4 ontology).
CLASSES = [
    "ALARM_6_00AM", "ALARM_8_00AM", "ALARM_9_00PM",
    "BRIGHTNESS_100", "BRIGHTNESS_20", "BRIGHTNESS_60",
    "CALL", "COLOR_BLUE", "COLOR_GREEN", "COLOR_RED",
    "CREATE_REMINDER_DRINK_WATER", "CREATE_REMINDER_EXERCISE", "CREATE_REMINDER_STUDY",
    "LIGHT_OFF", "LIGHT_ON", "LIST_REMINDERS", "MESSAGE", "NEXT", "PAUSE",
    "PLAY_MUSIC", "STOP", "TEMPERATURE_18", "TEMPERATURE_22", "TEMPERATURE_26",
    "TIME", "TIMER_10s", "TIMER_1m", "TIMER_30s", "VOLUME_DOWN", "VOLUME_UP",
    "WEATHER",
]
REJECT = "REJECT"
ALL_CLASSES = CLASSES + [REJECT]          # 32
CLASS2IDX = {c: i for i, c in enumerate(ALL_CLASSES)}


def tokenize(t: str) -> list[str]:
    t = t.strip().lower()
    return re.findall(r"[a-z0-9']+", t)


def load_manifest():
    with open(MANIFEST) as f:
        return list(csv.DictReader(f))


def build_vocab(rows):
    words = set()
    for r in rows:
        if r["source"] == "optionb":
            words.update(tokenize(r["transcript"]))
    words = sorted(words)
    all_tokens = [BLANK, UNK] + words
    w2i = {w: i for i, w in enumerate(all_tokens)}
    i2w = {i: w for w, i in w2i.items()}
    return words, w2i, i2w


def _read_feat(p):
    """Module-level worker (picklable, torch-free): read + log-Mel features."""
    try:
        import soundfile as sf
        x, sr = sf.read(p, dtype="float32")
        mel = F.logmel(x, sr)
        return mel, None
    except Exception as e:
        return None, repr(e)


def main():
    import collections
    from concurrent.futures import ProcessPoolExecutor

    print("loading manifest ...", flush=True)
    rows = load_manifest()
    print(f"  {len(rows)} rows", flush=True)

    # ---- subsample to keep the file reads tractable ----
    MAX_CMD = int(os.environ.get("V5_MAX_CMD", "120"))
    MAX_OOD = int(os.environ.get("V5_MAX_OOD", "150"))
    buckets = collections.defaultdict(list)
    for r in rows:
        key = (r["label"] if r["source"] == "optionb" else "OOD", r["split"])
        buckets[key].append(r)
    kept = []
    for key in sorted(buckets):
        lst = sorted(buckets[key], key=lambda r: r["path"])
        cap = MAX_CMD if key[0] != "OOD" else MAX_OOD
        kept.extend(lst[:cap])
    order = {id(r): i for i, r in enumerate(rows)}
    kept.sort(key=lambda r: order[id(r)])
    rows = kept
    print(f"  subsampled to {len(rows)} clips "
          f"(<= {MAX_CMD}/cmd-split, <= {MAX_OOD}/ood-split)", flush=True)

    words, w2i, i2w = build_vocab(rows)
    print(f"  vocab: {len(words)} words (+blank+unk = {len(i2w)} tokens)", flush=True)

    # ---- parallel read + features (torch-free) ----
    def _full(r):
        base = MD if r["source"] == "optionb" else ME2
        return os.path.join(base, r["path"])
    paths = [_full(r) for r in rows]
    missing = 0
    for p in paths:
        if not os.path.exists(p):
            missing += 1
    print(f"  {len(paths) - missing}/{len(paths)} files exist", flush=True)
    print(f"  reading + features for {len(paths)} clips (parallel) ...", flush=True)
    with ProcessPoolExecutor(max_workers=24) as ex:
        results = list(ex.map(_read_feat, paths, chunksize=16))

    X_list, y_list, src_list, split_list, path_list = [], [], [], [], []
    word_seqs, wlen_list = [], []
    missing = 0
    for i, (r, (mel, err)) in enumerate(zip(rows, results)):
        if mel is None:
            missing += 1
            continue
        X_list.append(mel)
        if r["source"] == "optionb":
            y_list.append(CLASS2IDX[r["label"]]); src_list.append(0)
        else:
            y_list.append(CLASS2IDX[REJECT]); src_list.append(1)
        split_list.append({"train": 0, "val": 1, "test": 2}[r["split"]])
        path_list.append(r["path"])
        toks = tokenize(r["transcript"])
        wlen_list.append(len(toks))
        word_seqs.append([w2i.get(t, w2i[UNK]) for t in toks])
        if (i + 1) % 2000 == 0:
            print(f"  assembled {i+1}/{len(rows)}", flush=True)
    print(f"  extracted {len(X_list)} clips (missing {missing})", flush=True)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    src = np.array(src_list, dtype=np.int8)
    split = np.array(split_list, dtype=np.int8)
    path = np.array(path_list, dtype=object)

    maxlen = max(1, min(16, max(wlen_list)))
    W = np.zeros((len(X), maxlen), dtype=np.int64)
    for i, seq in enumerate(word_seqs):
        seq = seq[:maxlen]
        W[i, :len(seq)] = seq
    wlen = np.array(wlen_list, dtype=np.int32)

    np.savez_compressed(os.path.join(HERE, "features.npz"),
                        X=X, y=y, src=src, split=split, path=path)
    np.savez_compressed(os.path.join(HERE, "words.npz"),
                        words=W, wlen=wlen, maxlen=np.array(maxlen))
    vocab = {"words": words, "word2idx": w2i, "idx2word": i2w,
             "n_classes": len(ALL_CLASSES), "classes": ALL_CLASSES,
             "class2idx": CLASS2IDX, "blank": w2i[BLANK], "unk": w2i[UNK],
             "maxlen": maxlen}
    with open(os.path.join(HERE, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=1)

    cc = collections.Counter(y.tolist())
    print("\n=== class counts (all splits) ===")
    for c in ALL_CLASSES:
        print(f"  {c:32s} {cc[CLASS2IDX[c]]}")
    print(f"\nX {X.shape} {X.dtype} | src cmd/ood = {(src==0).sum()}/{(src==1).sum()}")
    print(f"word maxlen {maxlen} | splits train/val/test = "
          f"{(split==0).sum()}/{(split==1).sum()}/{(split==2).sum()}")


if __name__ == "__main__":
    main()
