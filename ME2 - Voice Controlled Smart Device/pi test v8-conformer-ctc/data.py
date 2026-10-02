"""pi test v8 -- data loading for the v6 dataset (CTC targets).

  * 16 kHz wav -> 80-dim log-mel (torchaudio, 25 ms / 10 ms)
  * transcript -> spoken word list (the SAME hgm.spoken tokenizer as v6)
  * CTC target: word indices, consecutive duplicates collapsed (blanks are
    implicit; the loss uses torch.nn.CTCLoss with blank=0)

Word 0 in the CTC output is the BLANK; word i+1 is vocab[i].
"""
from __future__ import annotations
import csv
import os
import random
import sys
import wave

import numpy as np
import torch
import torchaudio

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from hgm.spoken import tokenize_spoken          # noqa: E402  (vendored, self-contained)
from hgm.commands import phrase_to_class        # noqa: E402
from hgm.dict import build_dictionary           # noqa: E402

MEL_BINS = 80
FREQ_MIN, FREQ_MAX = 133.0, 6855.0
N_FFT = 512
WIN = 400          # 25 ms
HOP = 160          # 10 ms
MAX_FRAMES = 800   # 8 s at 10 ms hop (longer clips are truncated)


def log_mel(x: np.ndarray, sr: int = 16000) -> torch.Tensor:
    """float32 mono waveform -> [T, 80] log-mel."""
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
        f_min=FREQ_MIN, f_max=FREQ_MAX, n_mels=MEL_BINS,
        window_fn=lambda n: torch.hann_window(n))
    m = tf(t.unsqueeze(0)).squeeze(0)            # [n_mels, T]
    return torch.log(m.clamp_min(1e-5)).transpose(0, 1)  # [T, 80]


def read_wav16(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return x


class V8Dataset(torch.utils.data.Dataset):
    def __init__(self, items):
        # items: list of (wav_path, [word, ...])
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, words = self.items[i]
        x = read_wav16(path)
        mel = log_mel(x)
        if mel.shape[0] > MAX_FRAMES:
            mel = mel[:MAX_FRAMES]
        return mel, torch.tensor(words, dtype=torch.long)


def collate(batch):
    mels, labels = zip(*batch)
    T = max(m.shape[0] for m in mels)
    B = len(mels)
    mel = torch.full((B, T, MEL_BINS), float(torch.log(torch.tensor(1e-5))))
    for i, m in enumerate(mels):
        mel[i, :m.shape[0], :] = m
    lab = torch.full((B, max(l.numel() for l in labels)), 0, dtype=torch.long)
    for i, l in enumerate(labels):
        lab[i, :l.numel()] = l
    lens = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)
    llens = torch.tensor([l.numel() for l in labels], dtype=torch.long)
    return mel, lab, lens, llens


def ctc_target(words_idx: list[int]) -> list[int]:
    """Collapse consecutive duplicates (CTC convention)."""
    out = []
    for w in words_idx:
        if not out or out[-1] != w:
            out.append(w)
    return out


def load_split(data: str, dictionary: dict, word2idx: dict,
               max_numerals: int = 20000, seed: int = 0):
    """Return items [(wav_path, [ctc word idx ...])] for train+numerals."""
    items = []
    n_ooo = n_bad = 0
    mpath = os.path.join(data, "train", "manifest.csv")
    with open(mpath) as f:
        for r in csv.DictReader(f):
            if int(r.get("out_of_scope") or 0) == 1:
                n_ooo += 1
                continue
            tr = (r.get("transcript") or "").strip()
            if not tr:
                continue
            ws = tokenize_spoken(tr)
            if not ws or any(w not in dictionary for w in ws):
                n_bad += 1
                continue
            idx = ctc_target([word2idx[w] for w in ws])
            items.append((os.path.join(data, "train", "audio",
                                       os.path.basename(r["file"])), idx))
    n_train = len(items)
    random.seed(seed)
    num_rows = []
    mpath = os.path.join(data, "numerals", "manifest.csv")
    with open(mpath) as f:
        for r in csv.DictReader(f):
            tr = (r.get("transcript") or "").strip()
            if not tr:
                continue
            ws = tokenize_spoken(tr)
            if ws and all(w in dictionary for w in ws):
                num_rows.append((os.path.join(data, "numerals", "audio",
                                              os.path.basename(r["file"])),
                                 ctc_target([word2idx[w] for w in ws])))
    if len(num_rows) > max_numerals:
        num_rows = random.sample(num_rows, max_numerals)
    items += num_rows
    print(f"v8 train items: {len(items)} (in-scope {n_train} + numerals "
          f"{len(num_rows)}); {n_ooo} OOS + {n_bad} OOV excluded", flush=True)
    return items


def load_test(data: str, split: str = "test"):
    """Return rows [(wav_path, gold_coarse, is_synthetic)] for eval."""
    rows = []
    mpath = os.path.join(data, split, "manifest.csv")
    with open(mpath) as f:
        for r in csv.DictReader(f):
            gold = (r.get("command") or "").strip()
            if not gold:
                continue
            p = os.path.join(data, split, r["file"])
            if os.path.exists(p):
                rows.append((p, gold, int(r.get("is_synthetic") or 0)))
    return rows


def build_vocab(data: str):
    """The v6 dictionary (109 words: in-scope + number words), loaded from
    the v6 model artifacts so the data selection matches v6 exactly."""
    dictionary = {}
    with open(os.path.join(_HERE, "base_dictionary.txt")) as f:
        for line in f:
            parts = line.split()
            if len(parts) > 1:
                dictionary[parts[0]] = parts[1:]
    return dictionary


def build_vocab_v8(data: str):
    """Expanded v8 vocabulary: the 109 base words (in-scope + numbers) PLUS
    every word that appears in the out-of-scope (OOS) train transcripts.

    Why: the base 109-word vocab cannot decode OOS audio, so the free (greedy)
    decode comes out empty/garbage and the reject rule has nothing to work
    with. By adding the OOS words to the CTC vocabulary we let the model
    DECODE the out-of-scope speech accurately (the free decode produces the
    real words); rejection then happens at the command-matching stage, where
    those words do not form any of the 93 in-scope command phrases. This is
    what fixes the false rejections: in-scope commands still decode to a
    high-scoring constrained phrase, OOS decodes to words that match no
    command.

    The base 109 words keep their original (sorted) order so their indices
    are stable; OOS words are appended after them (also sorted). OOS words
    get a placeholder phone (SIL) -- the CTC head learns their acoustic
    emission directly from the OOS training clips, no dictionary needed.
    """
    dictionary = build_vocab(data)          # base 109, phones from v6
    base_words = sorted(dictionary)
    oos_words = set()
    mpath = os.path.join(data, "train", "manifest.csv")
    with open(mpath) as f:
        for r in csv.DictReader(f):
            if int(r.get("out_of_scope") or 0) != 1:
                continue
            tr = (r.get("transcript") or "").strip()
            if not tr:
                continue
            for w in tokenize_spoken(tr):
                oos_words.add(w)
    new_words = [w for w in sorted(oos_words) if w not in dictionary]
    for w in new_words:
        dictionary[w] = ["SIL"]            # placeholder; CTC learns it
    words = base_words + new_words         # base indices stay stable
    return dictionary, words


def oos_train_items(data: str, word2idx: dict):
    """The out-of-scope train clips, as (path, ctc word idx) items with their
    REAL word targets (so the model learns to decode them, not emit blank)."""
    items = []
    n_bad = 0
    mpath = os.path.join(data, "train", "manifest.csv")
    with open(mpath) as f:
        for r in csv.DictReader(f):
            if int(r.get("out_of_scope") or 0) != 1:
                continue
            tr = (r.get("transcript") or "").strip()
            if not tr:
                continue
            ws = tokenize_spoken(tr)
            if not ws or any(w not in word2idx for w in ws):
                n_bad += 1
                continue
            items.append((os.path.join(data, "train", "audio",
                                       os.path.basename(r["file"])),
                          ctc_target([word2idx[w] for w in ws])))
    print(f"v8 OOS train items: {len(items)} (skipped {n_bad} empty/OOV)",
          flush=True)
    return items


def oos_items(data: str, word2idx: dict, split: str = "holdout"):
    """OOS clips from `split` as (path, ctc word idx) with their REAL word
    targets. The model learns to DECODE out-of-scope speech accurately (the
    free decode produces the real words); rejection then happens at the
    command-matching / score-gap stage, where those words match no command.
    This is the opposite of the all-blank trick: we WANT the model to speak
    the OOS words, so the score gap (free vs constrained) is what rejects.
    """
    items = []
    n_bad = 0
    mpath = os.path.join(data, split, "manifest.csv")
    with open(mpath) as f:
        for r in csv.DictReader(f):
            if int(r.get("out_of_scope") or 0) != 1:
                continue
            tr = (r.get("transcript") or "").strip()
            if not tr:
                continue
            ws = tokenize_spoken(tr)
            if not ws or any(w not in word2idx for w in ws):
                n_bad += 1
                continue
            p = os.path.join(data, split, r["file"])
            if os.path.exists(p):
                items.append((p, ctc_target([word2idx[w] for w in ws])))
    return items
