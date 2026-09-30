"""Constrained CTC decoder (greedy + optional beam).

Shared by:
  * training-time evaluation (model/train.py)
  * ONNX export (deploy/export_onnx.py) — the exported graph ends in ids
  * the RPi service (deploy/rpi_service/server.py)

Because the output vocabulary IS the command vocab (vcm.VOCAB), every decoded
transcript is a string the parser understands. This is what makes the VCM a
"pure" command model rather than a general ASR.
"""
import numpy as np


def collapse_ctc(ids, blank=0):
    """Collapse a raw CTC frame sequence: remove blanks, merge repeats."""
    out, prev = [], None
    for i in ids:
        if i != blank and i != prev:
            out.append(i)
        prev = i
    return out


def greedy_decode(logits, blank=0):
    """logits [T,V] (log-probs or raw) -> list of token ids (1-based vocab)."""
    if isinstance(logits, np.ndarray):
        frame_ids = logits.argmax(axis=-1).tolist()
    else:
        frame_ids = logits.argmax(dim=-1).tolist()
    return collapse_ctc(frame_ids, blank=blank)


def confidence(logits, blank=0):
    """Mean log-probability of the argmax token over NON-BLANK frames.

    A confident command produces frames where the top word is strongly
    favored; an out-of-distribution or empty utterance produces frames where
    the top token is a weak guess (often blank). Averaging over the
    non-blank argmax frames (blanks excluded so silence can't inflate the
    score) gives a single scalar in (-inf, 0] that separates in-domain from
    OOD. This is the reject signal used by the RPi service and the benchmark
    (see decode_with_confidence / the reject gate in deploy + benchmark).
    """
    import math
    if isinstance(logits, np.ndarray):
        logp = _log_softmax_np(logits)
        frame_ids = np.argmax(logits, axis=-1)
    else:
        import torch
        logp = torch.log_softmax(logits, dim=-1)
        frame_ids = logits.argmax(dim=-1)
    vals = []
    for t in range(len(frame_ids)):
        i = int(frame_ids[t])
        if i == blank:
            continue
        vals.append(float(logp[t, i]))
    if not vals:
        return -math.inf
    return sum(vals) / len(vals)


def _log_softmax_np(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return (x - m) - np.log(e.sum(axis=-1, keepdims=True))


def decode_with_confidence(logits, blank=0, min_conf=None):
    """Greedy decode + confidence gate.

    Returns (text, conf). If min_conf is set and the utterance confidence is
    below it, the transcript is rejected to "" (empty) — the parser maps an
    empty transcript to {intent: "unknown"}, so the device takes no action
    instead of emitting a wrong command. This is the OOD-reject guard.
    """
    text = decode_to_text(logits, blank=blank)
    conf = confidence(logits, blank=blank)
    if min_conf is not None and conf < min_conf:
        return "", conf
    return text, conf


def ids_to_text(ids):
    """1-based token ids -> words (skip blank 0)."""
    from vcm.vocab import ID2WORD
    return [ID2WORD[i - 1] for i in ids if i > 0]


def decode_to_text(logits, blank=0):
    """logits [T,V] -> transcript string (words joined by spaces)."""
    ids = greedy_decode(logits, blank=blank)
    return " ".join(ids_to_text(ids))


def beam_decode(logits, beam=4, blank=0):
    """Small beam search over CTC frames. Returns best token-id sequence.

    Kept intentionally simple (class-scale); greedy is usually enough for a
    constrained command vocab, but this gives a knob for hard cases.
    """
    import math
    T, V = logits.shape
    # state: list of (logprob, last_id, seq)
    beams = [(0.0, blank, [])]
    for t in range(T):
        new_beams = []
        for lp, last, seq in beams:
            for v in range(1, V):
                if v == last:
                    continue  # CTC: repeats of a non-blank emit nothing
                nlp = lp + float(logits[t, v])
                new_beams.append((nlp, v, seq + [v]))
            # blank transition
            new_beams.append((lp + float(logits[t, blank]), blank, seq))
        new_beams.sort(key=lambda b: b[0], reverse=True)
        beams = new_beams[:beam]
    best = max(beams, key=lambda b: b[0])
    return best[2]
