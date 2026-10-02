#!/usr/bin/env python3
"""pi test v7 -- prepare the v6-dataset clips into a sphinxtrain task.

Builds <task>/etc/{vcm.dic, vcm.phone, vcm.filler, vcm.ngram.txt,
vcm_train.fileids, vcm_train.transcription, feat.params} + <task>/wav/*.wav
from the AI231 ME2 v6 dataset (the new HuggingFace set):

  * in-scope train clips (out_of_scope rows are the reject class -> excluded)
  * the numerals split (single digits, real speech) for the digit phones

Transcripts are converted to *spoken* form ("Alarm 6:00 AM" -> "alarm six am")
with the SAME tokenizer the v6 pipeline uses (hgm.spoken), so the AM hears
what the speaker actually says.

The task layout is identical to the v3 custom-AM task (sandbox/vcm_task) so
sphinxtrain runs unchanged; only the data is the new v6 dataset.

Usage
-----
    python training/prep_v6_data.py \
        --data /home/ron.andrei.soriano/sandbox/data/external/me2-v6/dataset \
        --task /home/ron.andrei.soriano/sandbox/vcm_task_v6 \
        [--max-numerals 20000] [--workers 32]
"""
import argparse
import csv
import os
import random
import shutil
import sys
import wave
from concurrent.futures import ProcessPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACK = os.path.dirname(_HERE)          # pi test v7/
_ME2 = os.path.dirname(_BACK)           # ME2 - Voice Controlled Smart Device/
sys.path.insert(0, os.path.join(_ME2, "pi test v6"))
from hgm.spoken import tokenize_spoken          # noqa: E402
from hgm.dict import build_dictionary           # noqa: E402

PHONES = ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH",
          "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N",
          "NG", "OW", "P", "R", "S", "SH", "SIL", "T", "TH", "UW", "V",
          "W", "Y", "Z"]

# same feature front-end as the v3 custom AM (feat.params in vcm_task/etc)
FEAT_PARAMS = """-lowerf 130
-upperf 6800
-nfilt 25
-transform dct
-lifter 22
-feat 1s_c_d_dd
-agc none
-cmn batch
-varnorm no
"""


def _load_clips(data, dictionary, max_numerals, seed=0):
    clips = []
    mpath = os.path.join(data, "train", "manifest.csv")
    n_ooo = n_bad = 0
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
            clips.append((os.path.join(data, "train", "audio",
                                       os.path.basename(r["file"])),
                          " ".join(ws)))
    n_train = len(clips)
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
                                 " ".join(ws)))
    if len(num_rows) > max_numerals:
        num_rows = random.sample(num_rows, max_numerals)
    clips += num_rows
    print(f"clips: {len(clips)} (in-scope train {n_train} + numerals "
          f"{len(num_rows)}); {n_ooo} OOS + {n_bad} OOV excluded", flush=True)
    return clips


def _copy_one(args):
    src, dst = args
    try:
        shutil.copyfile(src, dst)
        return 0
    except Exception:
        return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",
                    default="/home/ron.andrei.soriano/sandbox/data/"
                            "external/me2-v6/dataset")
    ap.add_argument("--task",
                    default="/home/ron.andrei.soriano/sandbox/vcm_task_v6")
    ap.add_argument("--max-numerals", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    etc = os.path.join(args.task, "etc")
    wavdir = os.path.join(args.task, "wav")
    os.makedirs(etc, exist_ok=True)
    os.makedirs(wavdir, exist_ok=True)

    # dictionary: the v6 dictionary (superset of the v3 dict3)
    dictionary = {}
    with open(os.path.join(_ME2, "pi test v6", "models",
                           "dictionary.txt")) as f:
        for line in f:
            parts = line.split()
            if len(parts) > 1:
                dictionary[parts[0]] = parts[1:]
    print(f"dictionary: {len(dictionary)} words", flush=True)

    clips = _load_clips(args.data, dictionary, args.max_numerals)

    # copy wavs (short unique names; keep a name->src map for the fileids)
    print("copying wavs ...", flush=True)
    jobs = [(src, os.path.join(wavdir, f"c{i:06d}.wav"))
            for i, (src, _) in enumerate(clips)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        bad = sum(ex.map(_copy_one, jobs, chunksize=64))
    if bad:
        print(f"WARNING: {bad} wav copies failed", flush=True)

    # etc files
    with open(os.path.join(etc, "vcm.dic"), "w") as f:
        for w in sorted(dictionary):
            f.write(f"{w} {' '.join(dictionary[w])}\n")
    with open(os.path.join(etc, "vcm.phone"), "w") as f:
        f.write("\n".join(PHONES) + "\n")
    with open(os.path.join(etc, "vcm.filler"), "w") as f:
        f.write("<s>\tSIL\n<sil>\tSIL\n</s>\tSIL\n")
    with open(os.path.join(etc, "vcm.ngram.txt"), "w") as f:
        for _, tr in clips:
            f.write(f"<s> {tr} </s>\n")
    with open(os.path.join(etc, "vcm_train.fileids"), "w") as f:
        for i in range(len(clips)):
            f.write(f"c{i:06d}\n")
    with open(os.path.join(etc, "vcm_train.transcription"), "w") as f:
        for i, (_, tr) in enumerate(clips):
            f.write(f"<s> {tr} </s> (c{i:06d})\n")
    with open(os.path.join(etc, "feat.params"), "w") as f:
        f.write(FEAT_PARAMS)
    print(f"task ready: {args.task}  ({len(clips)} clips)", flush=True)


if __name__ == "__main__":
    main()
