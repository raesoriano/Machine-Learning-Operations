"""Stage 2: learned text classifier  transcript -> 31 commands | REJECT.

Replaces v1's hand-written rule parser + canonicalizer. Trained on the ME2
manifest:
    positives (49,786 rows): transcript -> command   (one of the 31)
    negatives (32,992 rows): transcript -> REJECT    (OOD speech / noise)

Model: TF-IDF (word unigrams+bigrams  +  char 2-5 grams) + multinomial
logistic regression with class_weight='balanced'.

Why this config (val-validated, see reports/tune_results.json):
  * class_weight='balanced'  -- the command classes are imbalanced
      (STOP 'stop' x4167 vs ALARM_9_00PM x599; LIGHT_ON 4828 vs LIGHT_OFF 4915).
      Balanced weights cut false-REJECT on the held-out val split 0.90% -> 0.30%
      and lift val command accuracy 98.89% -> 99.12%.
  * char ngrams (2,5)        -- bag-of-words has ZERO tolerance to typos /
      ASR errors; char 2-5 grams let "voluyme up" still match "volume up".
  * word ngrams (1,2), C=4   -- best val command accuracy in the grid.

The classifier sees only words, so it is speaker-independent by construction
(the fix for v1's speaker-overfitting failure) and has an explicit REJECT
class (v1 had no reject training signal).
"""
import csv
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from .normalize import normalize

REJECT = "REJECT"
_HERE = os.path.dirname(os.path.abspath(__file__))
_ARTIFACT = os.path.join(_HERE, "..", "classifier.pkl")

# Chosen config (val-validated; see reports/tune_results.json).
CONFIG = dict(word_ngram=(1, 2), char_ngram=(2, 5),
              class_weight="balanced", C=4.0, min_df=3)


def load_classifier(path=None):
    with open(path or _ARTIFACT, "rb") as f:
        return pickle.load(f)


def build_pipeline(cfg=None):
    """TF-IDF (word unigrams+bigrams + char 2-5 grams, normalized text)
    -> multinomial logistic (class_weight='balanced').

    LogisticRegression (lbfgs) is used (not LinearSVC) so we get calibrated
    probabilities for a reject-confidence threshold.
    """
    cfg = dict(CONFIG, **(cfg or {}))
    vecs = [("word", TfidfVectorizer(
        analyzer="word", ngram_range=cfg["word_ngram"], sublinear_tf=True,
        min_df=cfg["min_df"], strip_accents="unicode"))]
    if cfg["char_ngram"]:
        vecs.append(("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=cfg["char_ngram"],
            sublinear_tf=True, min_df=cfg["min_df"],
            strip_accents="unicode")))
    return Pipeline([
        ("norm", FunctionTransformer(normalize, validate=False)),
        ("vec", FeatureUnion(vecs)),
        ("clf", LogisticRegression(C=cfg["C"], max_iter=2000, solver="lbfgs",
                                   class_weight=cfg["class_weight"])),
    ])


def load_rows(manifest_csv, split=None):
    """manifest -> (texts, labels, n_pos, n_neg).

    split in {None, 'train', 'val', 'test'}; None = all rows (production).
    """
    texts, labels, n_pos, n_neg = [], [], 0, 0
    with open(manifest_csv) as f:
        for r in csv.DictReader(f):
            if split is not None and r["split"] != split:
                continue
            t = r["transcript"].strip()
            if not t:
                continue
            texts.append(t)
            if r["polarity"] == "positive":
                labels.append(r["command"])
                n_pos += 1
            else:
                labels.append(REJECT)
                n_neg += 1
    return texts, labels, n_pos, n_neg


def train_classifier(manifest_csv, out_path=None, split=None, cfg=None,
                     val_split="val"):
    """Train on the ME2 manifest.

    split=None    -> train on ALL rows (production model).
    split='train' -> train on the train split only and report accuracy on the
                     HELD-OUT val split (the honest tuning number).

    Returns (pipeline, report).
    """
    texts, labels, n_pos, n_neg = load_rows(manifest_csv, split=split)
    pipe = build_pipeline(cfg)
    pipe.fit(texts, labels)
    report = {"n_positives": n_pos, "n_negatives": n_neg,
              "n_classes": len(set(labels)), "train_split": split,
              "config": {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in dict(CONFIG, **(cfg or {})).items()}}
    if split == "train":
        vx, vy, _, _ = load_rows(manifest_csv, split=val_split)
        pred = pipe.predict(vx)
        acc = sum(p == g for p, g in zip(pred, vy)) / len(vy)
        pos_idx = [i for i, g in enumerate(vy) if g != REJECT]
        fr = sum(pred[i] == REJECT for i in pos_idx) / len(pos_idx)
        neg_idx = [i for i, g in enumerate(vy) if g == REJECT]
        fa = sum(pred[i] != REJECT for i in neg_idx) / len(neg_idx) if neg_idx else 0.0
        report.update({"val_n": len(vy), "val_acc": round(acc, 4),
                       "val_false_reject": round(fr, 4),
                       "val_false_accept": round(fa, 4)})
    else:
        report["in_sample_acc"] = round(pipe.score(texts, labels), 4)
    out = out_path or _ARTIFACT
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(pipe, f)
    report["artifact"] = out
    return pipe, report


def predict(pipe, transcript):
    """transcript -> (command, prob). command is one of the 31 or REJECT."""
    norm = normalize(transcript)
    if not norm:
        return REJECT, 0.0
    proba = pipe.predict_proba([norm])[0]
    classes = list(pipe.classes_)
    i = int(proba.argmax())
    return classes[i], float(proba[i])


if __name__ == "__main__":
    pipe = load_classifier()
    tests = [
        "alarm six am", "turn off the lights", "set the temperature to twenty two degrees",
        "play music", "what's the weather", "turn the volume down", "change the color to red",
        "set a timer for ten seconds", "call mom", "send a message", "list my reminders",
        "create a reminder to drink water", "pause", "stop", "next", "lights on",
        "lights out", "end playback", "voluyme up",
        "brightness to one hundred percent", "turn the heat up",
        "the weather is nice today", "",
    ]
    for t in tests:
        c, p = predict(pipe, t)
        print(f"  {t!r:45s} -> {c:30s} ({p:.3f})")
