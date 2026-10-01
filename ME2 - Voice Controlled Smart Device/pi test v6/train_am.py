#!/usr/bin/env python3
"""pi test v6 -- build the three recognizer components and train the acoustic
model on the AI231 ME2 voice-command dataset.

  1. Phonetic dictionary : in-scope command words + number words -> ARPAbet
                           (from the CMU dict, used only as a build resource).
  2. Language model      : word bigram + constrained FSA over the 93 command
                           variations (spoken form) + a free unigram for
                           "what I actually heard".
  3. Acoustic model      : one 3-state L2R HMM per phone (diagonal-GMM
                           emissions), trained by bootstrap EM on the real
                           train clips + the numerals split (digit phones) +
                           real silence (for the SIL model).

The dataset transcripts are in *written* form ("Alarm 6:00 AM"); they are
converted to *spoken* form ("alarm six am") by hgm.spoken before phones are
looked up, so the model hears what the speaker actually says.

Outputs (in models/):
    acoustic_model.npz   -- the trained HMM/GMM acoustic model
    dictionary.txt       -- word -> phoneme
    lm.npz               -- language model + FSA
    phones.txt           -- phone inventory
    phrases.json         -- the 93 spoken phrases + their command classes
"""
from __future__ import annotations
import os, sys, json, csv, time, argparse
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hgm.feats import features, read_wav
from hgm.dict import build_dictionary, phones_in_dict, save as save_dict, SIL
from hgm.lm import LanguageModel
from hgm.acoustic import AcousticModel
from hgm.commands import phrase_to_class
from hgm.spoken import tokenize_spoken

DEFAULT_DATA = "/home/ron.andrei.soriano/sandbox/data/external/me2-v6/dataset"
VARIATIONS = os.path.join(HERE, "..", "..", "..",
                          "me2_reference_files", "variations.csv")
MODELS = os.path.join(HERE, "models")


def _load_variations(path):
    """[(label, variation, value, phrase, spoken_words)] for the 93 rows."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            phrase = r["phrase"]
            rows.append((r["label"], r["variation"], r["value"], phrase,
                         tokenize_spoken(phrase)))
    return rows


def _in_scope_vocab(rows):
    return sorted({w for _, _, _, _, ws in rows for w in ws})


def _numeral_vocab(data):
    """Number words spoken in the numerals split (zero..billion)."""
    words = set()
    mpath = os.path.join(data, "numerals", "manifest.csv")
    with open(mpath) as f:
        for r in csv.DictReader(f):
            tr = (r.get("transcript") or "").strip()
            if tr:
                words.update(tokenize_spoken(tr))
    return sorted(words)


def _extract(args):
    path, transcript = args
    try:
        x = read_wav(path)
        F = features(x)
        return (path, transcript, F)
    except Exception as e:
        return (path, transcript, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--n-comp", type=int, default=4)
    ap.add_argument("--max-numerals", type=int, default=20000,
                    help="cap on numerals clips used for the digit phones")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    data = args.data
    os.makedirs(MODELS, exist_ok=True)
    t0 = time.time()

    rows = _load_variations(VARIATIONS)
    phrases = [" ".join(ws) for _, _, _, _, ws in rows]
    class_of = [phrase_to_class(ph) for ph in phrases]
    vocab = _in_scope_vocab(rows)
    num_vocab = _numeral_vocab(data)
    all_words = sorted(set(vocab) | set(num_vocab))
    print(f"variations: {len(rows)} phrases, {len(vocab)} in-scope words, "
          f"{len(num_vocab)} number words = {len(all_words)} vocab", flush=True)

    # ---------------- 1. dictionary ---------------- #
    dictionary = build_dictionary(all_words)
    save_dict(dictionary, os.path.join(MODELS, "dictionary.txt"))
    phones = phones_in_dict(dictionary)
    print(f"dictionary: {len(dictionary)} words, {len(phones)} phones "
          f"(incl {SIL})", flush=True)
    open(os.path.join(MODELS, "phones.txt"), "w").write("\n".join(phones) + "\n")
    ncls = len(set(class_of) - {"REJECT"})
    print(f"language model: {len(phrases)} phrases -> {ncls} command classes",
          flush=True)

    # ---------------- 2. language model ---------------- #
    lm = LanguageModel(phrases, class_of, all_words)
    lm.save(os.path.join(MODELS, "lm.npz"))
    json.dump([{"label": l, "variation": v, "value": val,
                "phrase": ph, "class": c}
               for (l, v, val, _, ws), ph, c in
               zip(rows, phrases, class_of)],
              open(os.path.join(MODELS, "phrases.json"), "w"), indent=1)

    # ---------------- 3. acoustic model ---------------- #
    # in-scope train clips (out_of_scope rows are the reject class and are
    # NOT fed to the constrained model) + numerals for the digit phones.
    train = []
    mpath = os.path.join(data, "train", "manifest.csv")
    n_ooo = 0
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
                continue
            train.append((os.path.join(data, "train", "audio",
                                       os.path.basename(r["file"])),
                          " ".join(ws)))
    n_train = len(train)
    # numerals (cap for speed; every number word is well covered)
    import random
    random.seed(0)
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
                                 " ".join(ws)))
    if len(num_rows) > args.max_numerals:
        num_rows = random.sample(num_rows, args.max_numerals)
    train += num_rows
    print(f"training clips: {len(train)} (in-scope train {n_train} + "
          f"numerals {len(num_rows)}); {n_ooo} out-of-scope rows excluded",
          flush=True)

    # extract features in a pool
    print("extracting features ...", flush=True)
    with Pool(args.workers) as pool:
        results = pool.map(_extract, train, chunksize=16)
    Xs, seqs, sil_frames = [], [], []
    bad = 0
    for path, phrase, F in results:
        if F is None:
            bad += 1
            continue
        seq = []
        for w in phrase.split():
            if w in dictionary:
                seq.extend(dictionary[w])
        if not seq:
            bad += 1
            continue
        Xs.append(F)
        seqs.append(seq)
        n = F.shape[0]
        edge = max(1, int(0.12 / 0.010))
        sil_frames.append(F[:edge])
        sil_frames.append(F[-edge:])
    sil = (np.concatenate(sil_frames, axis=0) if sil_frames
           else np.zeros((200, Xs[0].shape[1])))
    print(f"features done: {len(Xs)} utterances, {sil.shape[0]} silence "
          f"frames, {bad} bad", flush=True)

    am = AcousticModel(phones, dim=Xs[0].shape[1], n_comp=args.n_comp, seed=0)
    print("training acoustic model (bootstrap EM) ...", flush=True)
    am.train(list(zip(Xs, seqs)), rounds=args.rounds, verbose=True)
    # train the SIL HMM on real silence
    print("training SIL on silence ...", flush=True)
    silh = am.hmms[SIL]
    for s, st in enumerate(silh.states):
        st.initialize_from_data(sil, seed=7 + s)
    for _ in range(6):
        log_b = silh.log_emit(sil)
        silh.baum_welch(sil, log_b, n_iter=1)
    am.save(os.path.join(MODELS, "acoustic_model.npz"))
    print(f"DONE in {time.time()-t0:.0f}s -> {MODELS}", flush=True)


if __name__ == "__main__":
    main()
