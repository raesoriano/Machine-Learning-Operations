"""pi test v5 -- model definitions (torch).

Two small, ONNX-exportable models:

1. ASRModel  -- limited-vocabulary CTC speech recognition.
   Input  (B, 40, 301) log-Mel  ->  (B, T, vocab) logits.
   Decodes to a word sequence over the ~82-word command vocabulary;
   anything it cannot match is emitted as <unk>.

2. PhraseModel -- phrase classifier over the decoded word sequence.
   Input  (B, maxlen) word indices  ->  (B, 32) logits
   (31 command classes + REJECT).

Both are 1-D CNNs, a few hundred KB to a couple of MB in fp32.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASRModel(nn.Module):
    """CRNN (1-D CNN + BiLSTM) encoder + CTC head over log-Mel frames.

    Input  (B, 40, 301) log-Mel  ->  (B, T, vocab) logits, T = 75 after 2
    maxpools (T/L ~ 4.7x for <=16-word targets -- enough room for CTC).
    Decodes to a word sequence over the ~82-word command vocabulary;
    anything it cannot match is emitted as <unk>.
    """

    def __init__(self, n_mels=40, n_frames=301, vocab=84,
                 conv_channels=(64, 128), rnn_hidden=128, rnn_layers=2,
                 dropout=0.2, feat_mean=None, feat_std=None):
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.vocab = vocab
        # Fixed feature normalization (computed on the training set). The raw
        # log-Mel has a wide range and large variance, which makes the first
        # conv layer + BatchNorm diverge under AdamW. Zero-mean / unit-variance
        # input keeps the network stable. The SAME stats are used at inference
        # (infer.py) so train/test match.
        if feat_mean is None:
            feat_mean = 0.0
        if feat_std is None:
            feat_std = 1.0
        self.register_buffer("feat_mean",
                             torch.tensor(feat_mean, dtype=torch.float32))
        self.register_buffer("feat_std",
                             torch.tensor(feat_std, dtype=torch.float32))
        # CNN feature extractor: 301 -> 150 -> 75
        c0, c1 = conv_channels
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mels, c0, 5, padding=2),
            nn.BatchNorm1d(c0), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c0, c1, 5, padding=2),
            nn.BatchNorm1d(c1), nn.ReLU(), nn.MaxPool1d(2),
        )
        # BiLSTM sequence model (the standard for CTC ASR)
        self.rnn = nn.LSTM(c1, rnn_hidden, num_layers=rnn_layers,
                           bidirectional=True, batch_first=True,
                           dropout=dropout)
        self.head = nn.Linear(rnn_hidden * 2, vocab)

    def forward(self, x):
        # x: (B, n_mels, n_frames)
        if x.shape[-2:] != (self.n_mels, self.n_frames):
            raise ValueError(f"expected (B,{self.n_mels},{self.n_frames}), got {tuple(x.shape)}")
        x = (x - self.feat_mean) / self.feat_std     # fixed normalization
        x = self.cnn(x)                 # (B, C, 75)
        x = x.permute(0, 2, 1)          # (B, 75, C)
        x, _ = self.rnn(x)              # (B, 75, 2*hidden)
        return self.head(x)             # (B, 75, vocab)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class PhraseModel(nn.Module):
    """1-D CNN over embedded word indices -> command / REJECT."""

    def __init__(self, vocab=84, embed=64, maxlen=8, num_classes=32,
                 channels=(128, 128), dropout=0.3):
        super().__init__()
        self.vocab = vocab
        self.maxlen = maxlen
        self.embed = nn.Embedding(vocab, embed, padding_idx=0)
        layers = []
        prev = embed
        for ch in channels:
            layers += [
                nn.Conv1d(prev, ch, 3, padding=1),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            prev = ch
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(prev, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, w):
        # w: (B, maxlen) int64 word indices
        e = self.embed(w).permute(0, 2, 1)   # (B, embed, maxlen)
        x = self.cnn(e)
        x = self.pool(x).squeeze(-1)         # (B, C)
        return self.fc(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# CTC helpers (used by training + inference)
# --------------------------------------------------------------------------- #
def ctc_greedy_decode(logits, blank):
    """logits: (B, T, vocab) -> list of word-index lists (repeats collapsed,
    blanks removed)."""
    with torch.no_grad():
        idx = logits.argmax(dim=-1)          # (B, T)
        out = []
        for b in range(idx.shape[0]):
            seq = []
            prev = None
            for t in range(idx.shape[1]):
                tok = int(idx[b, t])
                if tok != prev:
                    if tok != blank:
                        seq.append(tok)
                prev = tok
            out.append(seq)
    return out


def ctc_loss(logits, targets, target_len, blank, reduction="mean"):
    """logits: (B, T, vocab); targets: (B, maxL) int64; target_len: (B,) int."""
    # torch ctc_loss wants log_probs as (T, B, C). logits is (B, T, C), so
    # permute(1, 0, 2). (permute(2, 0, 1) would give (C, B, T) which torch
    # misreads as (T=C, B, C=T) -- the loss and its gradient are then
    # computed on the wrong axes and diverge to NaN.)
    log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
    T = logits.shape[1]
    return F.ctc_loss(log_probs, targets,
        input_lengths=torch.full((logits.shape[0],), T, dtype=torch.long,
                                 device=logits.device),
        target_lengths=target_len, blank=blank, reduction=reduction,
        zero_infinity=True)
