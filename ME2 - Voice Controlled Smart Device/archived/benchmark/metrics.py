"""Benchmark metrics. Pure Python (no jiwer dependency).

Gold and predicted commands are both ``{intent, slots}`` dicts produced by
the SAME parser (vcm.parser.parse), so scoring is a deterministic comparison.
"""
import re


def _norm(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def _lev(a, b):
    """Word-level Levenshtein distance."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(ref, hyp):
    """Word error rate, 0.0 = perfect."""
    r = _norm(ref).split()
    h = _norm(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    return _lev(r, h) / len(r)


def slots_equal(gold, pred):
    """Exact match on the slots dict (values compared as normalized strings)."""
    if set(gold) != set(pred):
        return False
    return all(_norm(gold[k]) == _norm(pred[k]) for k in gold)


def slot_f1(gold, pred):
    """Micro F1 over (slot_key, value) pairs.

    A slot counts as correct only if the key is present AND the value matches
    exactly (after normalization). Extra predicted slots hurt precision.
    """
    g = {k: _norm(v) for k, v in gold.items()}
    p = {k: _norm(v) for k, v in pred.items()}
    tp = sum(1 for k, v in g.items() if p.get(k) == v)
    precision = tp / len(p) if p else (1.0 if not g else 0.0)
    recall = tp / len(g) if g else (1.0 if not p else 0.0)
    if precision + recall == 0:
        return 1.0 if not g and not p else 0.0
    return 2 * precision * recall / (precision + recall)


class Metrics:
    """Accumulate per-row outcomes, then summarize.

    Row kinds:
      * in-vocab  (intent != unknown):  correct = exact match on (intent,slots)
      * OOV       (intent == unknown):  correct = model also said unknown
    """

    def __init__(self):
        self.rows = []

    def add(self, row_id, gold, pred, transcript, gold_text, subset="core"):
        self.rows.append({
            "id": row_id, "subset": subset,
            "gold": gold, "pred": pred,
            "transcript": transcript, "gold_text": gold_text,
        })

    def _split(self):
        in_vocab = [r for r in self.rows if r["gold"]["intent"] != "unknown"]
        oov = [r for r in self.rows if r["gold"]["intent"] == "unknown"]
        return in_vocab, oov

    def summary(self):
        in_vocab, oov = self._split()
        s = {}
        if in_vocab:
            exact = [r for r in in_vocab
                     if r["pred"]["intent"] == r["gold"]["intent"]
                     and slots_equal(r["gold"]["slots"], r["pred"]["slots"])]
            intent_ok = [r for r in in_vocab
                         if r["pred"]["intent"] == r["gold"]["intent"]]
            s["n_in_vocab"] = len(in_vocab)
            s["intent_accuracy"] = len(intent_ok) / len(in_vocab)
            s["slot_f1"] = sum(slot_f1(r["gold"]["slots"], r["pred"]["slots"])
                               for r in in_vocab) / len(in_vocab)
            s["exact_match"] = len(exact) / len(in_vocab)
            s["wer"] = sum(wer(r["gold_text"], r["transcript"])
                           for r in in_vocab) / len(in_vocab)
        if oov:
            rejected = [r for r in oov if r["pred"]["intent"] == "unknown"]
            s["n_oov"] = len(oov)
            s["rejection_rate"] = len(rejected) / len(oov)
            s["false_accept_rate"] = 1.0 - s["rejection_rate"]
        if in_vocab and oov:
            s["command_accuracy"] = (
                sum(1 for r in in_vocab
                    if r["pred"]["intent"] == r["gold"]["intent"]
                    and slots_equal(r["gold"]["slots"], r["pred"]["slots"]))
                + sum(1 for r in oov if r["pred"]["intent"] == "unknown")
            ) / len(self.rows)
        return s

    def per_intent(self):
        in_vocab, _ = self._split()
        out = {}
        by = {}
        for r in in_vocab:
            by.setdefault(r["gold"]["intent"], []).append(r)
        for intent, rs in sorted(by.items()):
            exact = sum(1 for r in rs
                        if r["pred"]["intent"] == r["gold"]["intent"]
                        and slots_equal(r["gold"]["slots"], r["pred"]["slots"]))
            out[intent] = {
                "n": len(rs),
                "exact_match": exact / len(rs),
                "slot_f1": sum(slot_f1(r["gold"]["slots"], r["pred"]["slots"])
                               for r in rs) / len(rs),
            }
        return out

    def per_subset(self):
        out = {}
        for r in self.rows:
            out.setdefault(r["subset"], []).append(r)
        res = {}
        for subset, rs in sorted(out.items()):
            m = Metrics()
            for r in rs:
                m.add(r["id"], r["gold"], r["pred"], r["transcript"],
                      r["gold_text"], subset)
            res[subset] = m.summary()
        return res
