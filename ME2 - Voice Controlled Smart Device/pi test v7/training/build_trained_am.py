#!/usr/bin/env python3
"""Prepare PocketSphinx training data + build a command-domain acoustic model
with sphinxtrain, trained on OUR dataset (the clean optionb clips).

NOTE (2026-09-29): the ACTUAL training run used the modern sphinxtrain
task/stage interface (sphinxtrain -t vcm setup / -s <stage> run) with the
task dir at <sandbox>/vcm_task (sphinx_train.cfg + resolved.json), NOT the
classic 4-stage command line below. The classic `-1 sphinx2fe -2
train_sequencemodel -3 train_hmm -4 train_idcd` interface was removed from
sphinxtrain master. The resulting model (vcm.cd_cont_200) was packaged into
artifacts/pocketsphinx_trained/ and evaluated: see
backbone/reports/pocketsphinx_trained_{cmd,free}.json and
backbone/reports/pocketsphinx_comparison.json. This script's data-prep and
dict/LM steps are still valid reference code.

This is the "PocketSphinx trained from our dataset" model. It produces:
    artifacts/pocketsphinx_trained/
        mdef, means, variances, transition_matrices, sendump, noisedict,
        feat.params, ...          (the trained HMM acoustic model)
        dict                      (word -> phones, built from cmudict)
        vcm.lm.bin                (bigram LM over the command phrases)
        vcm_commands.jsgf         (the command grammar, copied)

Pipeline
--------
1. data prep : clean optionb clips -> train_dir/{wav,text}
2. dict      : distinct words -> phones (cmudict + digit handling)
3. mdef      : reuse the en-us phone set
4. sphinxtrain: sphinx2fe -> train_sequencemodel -> train_hmm -> train_idcd
5. lmtool    : build a bigram LM from the command phrase list

Usage
-----
    python backbone/pocketsphinx/build_trained_am.py \
        --sphinxtrain /home/.../sphinx_src/install/bin/sphinxtrain \
        --lmtool      /home/.../sphinx_src/install/bin/lmtool \
        --out backbone/artifacts/pocketsphinx_trained
"""
import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import wave

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACK = os.path.dirname(_HERE)
_REPO = os.path.dirname(_BACK)
_ME2 = os.path.dirname(_REPO)

# bundled en-us model (stock acoustic model + cmudict) provides the phone set
import pocketsphinx
_PS = os.path.dirname(pocketsphinx.__file__)
_ENUS = os.path.join(_PS, "model", "en-us")
_CMU = os.path.join(_ENUS, "cmudict-en-us.dict")
_ENUS_AM = os.path.join(_ENUS, "en-us")


def load_clean_clips(manifest):
    """manifest (train.jsonl) -> list of (id, wav_path, transcript) for _clean."""
    out = []
    for line in open(manifest):
        r = json.loads(line)
        if r["audio"].endswith("_clean.wav"):
            wav = os.path.join(_REPO, r["audio"])
            if os.path.exists(wav):
                out.append((r["id"], wav, r["text"].strip()))
    return out


def ensure_16k_mono(src, dst):
    """Copy wav to dst, resampling to 16 kHz 16-bit mono if needed."""
    with wave.open(src, "rb") as w:
        sr, sw, ch = w.getframerate(), w.getsampwidth(), w.getnchannels()
        if sr == 16000 and sw == 2 and ch == 1:
            shutil.copyfile(src, dst)
            return
    # otherwise resample with scipy
    import scipy.io.wavfile as swf
    import numpy as np
    rate, data = swf.read(src)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype != np.int16:
        if data.dtype == np.float32 or data.dtype == np.float64:
            data = (np.clip(data, -1, 1) * 32767).astype(np.int16)
        else:
            data = data.astype(np.int16)
    if rate != 16000:
        n = int(len(data) * 16000 / rate)
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, n)
        data = np.interp(x_new, x_old, data).astype(np.int16)
    swf.write(dst, 16000, data)


def build_dict(transcripts, out_path):
    """Distinct words across transcripts -> phones, using cmudict.
    Returns (dict_text, oov_words)."""
    cmu = {}
    with open(_CMU) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                cmu[parts[0].lower()] = parts[1]
    words = collections.Counter()
    for t in transcripts:
        for w in t.lower().split():
            w = w.strip(".,!?;:'\"()")
            if w:
                words[w] += 1
    oov = []
    lines = []
    for w in sorted(words):
        if w in cmu:
            lines.append(f"{w} {cmu[w]}")
        elif w.isdigit():
            # map digits to cmudict digit phones if present
            if w in cmu:
                lines.append(f"{w} {cmu[w]}")
            else:
                oov.append(w)
        else:
            oov.append(w)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return oov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sphinxtrain", required=True)
    ap.add_argument("--lmtool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=os.path.join(_REPO, "data", "features", "train.jsonl"))
    ap.add_argument("--max-clips", type=int, default=0, help="0 = all clean clips")
    ap.add_argument("--skip-train", action="store_true", help="reuse existing train_dir")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    train_dir = os.path.join(out, "train")
    wav_dir = os.path.join(train_dir, "wav")
    text_dir = os.path.join(train_dir, "text")

    clips = load_clean_clips(args.manifest)
    if args.max_clips:
        clips = clips[:args.max_clips]
    print(f"[prep] {len(clips)} clean clips", flush=True)

    if not args.skip_train:
        os.makedirs(wav_dir, exist_ok=True)
        os.makedirs(text_dir, exist_ok=True)
        for i, (cid, wav, text) in enumerate(clips):
            ensure_16k_mono(wav, os.path.join(wav_dir, f"{cid}.wav"))
            with open(os.path.join(text_dir, f"{cid}.txt"), "w") as f:
                f.write(text + "\n")
            if (i + 1) % 1000 == 0:
                print(f"[prep] {i + 1}/{len(clips)}", flush=True)
    else:
        print("[prep] reusing existing train_dir", flush=True)

    # dictionary
    transcripts = [t for _, _, t in clips]
    oov = build_dict(transcripts, os.path.join(out, "dict"))
    print(f"[dict] {len(set(w for t in transcripts for w in t.split()))} words, "
          f"OOV={oov}", flush=True)

    # mdef: reuse en-us phone set
    shutil.copyfile(os.path.join(_ENUS_AM, "mdef"), os.path.join(out, "mdef"))

    # copy the command grammar
    shutil.copyfile(os.path.join(_HERE, "vcm_commands.jsgf"),
                    os.path.join(out, "vcm_commands.jsgf"))

    # ---- sphinxtrain ----
    # sphinxtrain reads a config file. Build one.
    conf = os.path.join(out, "training.conf")
    with open(conf, "w") as f:
        f.write(f"""
# sphinxtrain configuration
train_dir = {train_dir}
mdef = {os.path.join(out, 'mdef')}
dict = {os.path.join(out, 'dict')}
# output
hmm_dir = {out}
# feature extraction (sphinx2fe)
fs = 16000
# HMM training
n_emit = 3
# number of phones
""")
    print(f"[train] running sphinxtrain ...", flush=True)
    cmd = [args.sphinxtrain, "-1", "sphinx2fe", "-2", "train_sequencemodel",
           "-3", "train_hmm", "-4", "train_idcd",
           "-c", conf]
    print("[train] " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("---- sphinxtrain stdout (tail) ----")
    print("\n".join(r.stdout.splitlines()[-40:]))
    print("---- sphinxtrain stderr (tail) ----")
    print("\n".join(r.stderr.splitlines()[-40:]))
    if r.returncode != 0:
        print(f"[train] sphinxtrain FAILED rc={r.returncode}", flush=True)
        # keep going to attempt LM build
    else:
        print("[train] sphinxtrain OK", flush=True)

    # ---- LM (bigram) over the command phrases ----
    print(f"[lm] building bigram LM ...", flush=True)
    # lmtool wants a text file of sentences (one per line)
    lm_text = os.path.join(out, "lm_text.txt")
    with open(lm_text, "w") as f:
        for t in sorted(set(transcripts)):
            f.write(t + "\n")
    cmd2 = [args.lmtool, "-lm_type", "ngram", "-dict", os.path.join(out, "dict"),
            "-lm_format", "bin", "-lm_output", os.path.join(out, "vcm.lm.bin"),
            "-arg", "-bestfirst", "1", "-arg", "-beam", "1e-60",
            "-arg", "-warp", "1", "-arg", "-kldist", "0",
            "-arg", "-discount", "0.1",
            lm_text]
    print("[lm] " + " ".join(cmd2), flush=True)
    r2 = subprocess.run(cmd2, capture_output=True, text=True)
    print("\n".join(r2.stdout.splitlines()[-15:]))
    print("\n".join(r2.stderr.splitlines()[-15:]))
    if r2.returncode == 0:
        print("[lm] LM OK ->", os.path.join(out, "vcm.lm.bin"), flush=True)
    else:
        print(f"[lm] lmtool FAILED rc={r2.returncode}", flush=True)

    print(f"\n[done] artifacts in {out}", flush=True)
    for fn in sorted(os.listdir(out)):
        fp = os.path.join(out, fn)
        if os.path.isfile(fp):
            print(f"  {os.path.getsize(fp) // 1024:>8} KB  {fn}")


if __name__ == "__main__":
    main()
