#!/usr/bin/env python3
"""VCM v2 -- Raspberry Pi voice command listener.

Listens on the mic, transcribes each utterance with the fine-tuned Whisper
base.en (ONNX, int8) and classifies it into one of the 31 ME2 commands
(+ REJECT), printing per-utterance metrics:

    VAD wait / mel / encoder / decoder / ASR total / classify / E2E latency,
    the recognized words (transcript), the classified command + intent +
    confidence, and a rolling session summary (avg / p50 / p95).

Two backends:
  --model full      (default)  Whisper encoder + decoder ONNX, greedy decode,
                               TF-IDF+LogReg stage-2 classifier on the text.
                               85.4% command / 86.0% intent on the 171-clip
                               held-out set. ~580 MB int8.
  --model student           distilled ~1.5M-param encoder + linear 32-way head.
                               mel -> embedding -> command, no text step.
                               ~1.5 MB int8. (Accuracy: see README.)

Usage:
    python vcm_pi.py                          # live mic, full model
    python vcm_pi.py --model student          # live mic, distilled student
    python vcm_pi.py --test                   # run the 171-clip test set
    python vcm_pi.py --file path/to/clip.wav  # transcribe a single file
"""
import argparse
import json
import os
import re
import statistics
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ME2 = os.path.dirname(os.path.dirname(_HERE))  # .../ME2 - Voice Controlled Smart Device

# --------------------------------------------------------------------------
# text normalization (inlined from vcm2.normalize / vcm.parser / vcm.numbers
# so this folder is fully self-contained)
# --------------------------------------------------------------------------
_ONES_REV = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
             11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
             15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
             19: "nineteen"}
_TENS_REV = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
             60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}


def _int_to_words(n):
    if not 0 <= n <= 120:
        raise ValueError("out of range 0-120")
    if n < 20:
        return _ONES_REV[n]
    if n < 100:
        t, r = divmod(n, 10)
        return _TENS_REV[t * 10] + (f" {_ONES_REV[r]}" if r else "")
    h, r = divmod(n, 100)
    s = f"{_ONES_REV[h]} hundred"
    if r:
        s += f" {int_to_words(r)}"
    return s


def _base_norm(text):
    t = (text or "").lower().strip()
    t = t.replace("\u2019", "'")
    t = t.replace("what's", "whats")
    t = re.sub(r"[^\w\s%]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize(text):
    """lowercase, strip punctuation, expand digits 0-120 to number-words."""
    t = _base_norm(text)
    toks = []
    for tok in t.split():
        if tok.isdigit() and 0 <= int(tok) <= 120:
            toks.extend(_int_to_words(int(tok)).split())
        else:
            toks.append(tok)
    return " ".join(toks)


# --------------------------------------------------------------------------
# command / intent metadata (same as backbone/scripts/eval_whisper_ft.py)
# --------------------------------------------------------------------------
_INTENT_OF = {
    "PLAY_MUSIC": "play_music", "WEATHER": "ask_question", "TIME": "ask_question",
    "LIGHT_ON": "lights_switch", "LIGHT_OFF": "lights_switch",
    "BRIGHTNESS_20": "lights_adjust", "BRIGHTNESS_60": "lights_adjust",
    "BRIGHTNESS_100": "lights_adjust",
    "COLOR_RED": "lights_adjust", "COLOR_GREEN": "lights_adjust",
    "COLOR_BLUE": "lights_adjust",
    "TIMER_10s": "set_timer", "TIMER_30s": "set_timer", "TIMER_1m": "set_timer",
    "ALARM_6_00AM": "set_alarm", "ALARM_8_00AM": "set_alarm",
    "ALARM_9_00PM": "set_alarm",
    "TEMPERATURE_18": "set_temperature", "TEMPERATURE_22": "set_temperature",
    "TEMPERATURE_26": "set_temperature",
    "PAUSE": "media_control", "STOP": "media_control", "NEXT": "media_control",
    "VOLUME_UP": "media_control", "VOLUME_DOWN": "media_control",
    "LIST_REMINDERS": "reminders_lists",
    "CREATE_REMINDER_DRINK_WATER": "reminders_lists",
    "CREATE_REMINDER_EXERCISE": "reminders_lists",
    "CREATE_REMINDER_STUDY": "reminders_lists",
    "CALL": "call", "MESSAGE": "call", "REJECT": "unknown",
}

_ALARM = {"6": "ALARM_6_00AM", "8": "ALARM_8_00AM", "9": "ALARM_9_00PM"}
_BRIGHT = {"20": "BRIGHTNESS_20", "60": "BRIGHTNESS_60", "100": "BRIGHTNESS_100"}
_TEMP = {"18": "TEMPERATURE_18", "22": "TEMPERATURE_22", "26": "TEMPERATURE_26"}
_TIMER = {"10": "TIMER_10s", "30": "TIMER_30s", "1": "TIMER_1m"}
_COLOR = {"red": "COLOR_RED", "green": "COLOR_GREEN", "blue": "COLOR_BLUE"}
_REMINDER = {"drink water": "CREATE_REMINDER_DRINK_WATER",
             "exercise": "CREATE_REMINDER_EXERCISE",
             "study": "CREATE_REMINDER_STUDY"}
_SIMPLE = {"CALL": "CALL", "LIGHT_OFF": "LIGHT_OFF", "LIGHT_ON": "LIGHT_ON",
           "LIST_REMINDERS": "LIST_REMINDERS", "MESSAGE": "MESSAGE",
           "NEXT": "NEXT", "PAUSE": "PAUSE", "PLAY_MUSIC": "PLAY_MUSIC",
           "STOP": "STOP", "TIME": "TIME", "VOLUME_DOWN": "VOLUME_DOWN",
           "VOLUME_UP": "VOLUME_UP", "WEATHER": "WEATHER"}


def gold_command(folder, spoken_phrase):
    folder = folder.upper()
    t = normalize(spoken_phrase)
    raw = (spoken_phrase or "").lower()
    if folder in _SIMPLE:
        return _SIMPLE[folder]
    if folder == "ALARM":
        m = re.search(r"\d+", raw)
        return _ALARM.get(m.group(0), "REJECT") if m else "REJECT"
    if folder == "BRIGHTNESS":
        m = re.search(r"\d+", raw)
        return _BRIGHT.get(m.group(0), "REJECT") if m else "REJECT"
    if folder == "TEMPERATURE":
        m = re.search(r"\d+", raw)
        return _TEMP.get(m.group(0), "REJECT") if m else "REJECT"
    if folder == "TIMER":
        m = re.search(r"\d+", raw)
        return _TIMER.get(m.group(0), "REJECT") if m else "REJECT"
    if folder == "COLOR":
        for c, cmd in _COLOR.items():
            if re.search(r"\b" + c + r"\b", t):
                return cmd
        return "REJECT"
    if folder == "CREATE_REMINDER":
        for a, cmd in _REMINDER.items():
            if a in t:
                return cmd
        return "REJECT"
    return "REJECT"


def spoken_from_filename(fname):
    base = fname[:-4] if fname.lower().endswith(".wav") else fname
    return re.sub(r" \d+$", "", base)


def wer(ref, hyp):
    ref = normalize(ref).split()
    hyp = normalize(hyp).split()[:200]
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        d[i, 0] = i
    for j in range(m + 1):
        d[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(d[i - 1, j - 1] + 1, d[i - 1, j] + 1,
                              d[i, j - 1] + 1)
    return d[n, m] / max(n, 1)


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------
class FullWhisper:
    """Whisper base.en fine-tuned: encoder.onnx + decoder.onnx (greedy) +
    TF-IDF stage-2 classifier on the transcript."""

    name = "full (whisper base.en fine-tuned, int8)"

    def __init__(self, fp32=False):
        import onnxruntime as ort
        from transformers import AutoTokenizer, WhisperFeatureExtractor
        from vcm2.classifier import load_classifier, predict

        suffix = "" if fp32 else "_int8"
        ep = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(
            os.path.join(_HERE, f"encoder{suffix}.onnx"), providers=ep)
        self.dec = ort.InferenceSession(
            os.path.join(_HERE, f"decoder{suffix}.onnx"), providers=ep)
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(_HERE, "tokenizer"))
        self.fe = WhisperFeatureExtractor.from_pretrained(
            os.path.join(_HERE, "tokenizer"))
        self.clf = load_classifier(os.path.join(
            _ME2, "vcm-v2", "archive", "ctc_v8", "artifacts", "classifier.pkl"))
        self.predict = predict
        self.TASK = 50257   # bos / task token
        self.EOS = 50256
        self.MAX_NEW = 64

    def transcribe(self, audio, sr):
        """-> (transcript, timings dict)"""
        t0 = time.perf_counter()
        feat = self.fe(audio, sampling_rate=16000, return_tensors="np",
                       padding="max_length")["input_features"][0].astype(np.float32)
        n_frames = min(int(round(len(audio) / 16000 * 50)), 3000)
        mask = np.zeros((1, 3000), dtype=np.int64)
        mask[0, :n_frames] = 1
        t_mel = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        hidden = self.enc.run(None, {"input_features": feat[None]})[0]
        t_enc = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        ids = [self.TASK]
        steps = 0
        for _ in range(self.MAX_NEW):
            din = np.array([ids], dtype=np.int64)
            dam = np.ones((1, len(ids)), dtype=np.int64)
            logits = self.dec.run(None, {"input_ids": din,
                                         "attention_mask": dam,
                                         "encoder_hidden_states": hidden})[0]
            nxt = int(logits[0, -1].argmax())
            ids.append(nxt)
            steps += 1
            if nxt == self.EOS:
                break
        t_dec = (time.perf_counter() - t0) * 1e3
        text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        return text, {"mel_ms": t_mel, "enc_ms": t_enc, "dec_ms": t_dec,
                      "dec_steps": steps}

    def classify(self, text):
        t0 = time.perf_counter()
        cmd, prob = self.predict(self.clf, text)
        return cmd, prob, (time.perf_counter() - t0) * 1e3


class Student:
    """Distilled ~1.5M-param student: mel -> 512-d embedding -> 32-way head.
    No text step, no tokenizer, no sklearn -- just two tiny ONNX graphs."""

    name = "student (distilled ~1.5M, int8)"

    def __init__(self):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor
        ep = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(os.path.join(_HERE, "student_int8.onnx"),
                                        providers=ep)
        self.head = ort.InferenceSession(os.path.join(_HERE, "head_int8.onnx"),
                                         providers=ep)
        with open(os.path.join(_HERE, "classes.json")) as f:
            self.classes = json.load(f)["classes"]
        self.fe = WhisperFeatureExtractor.from_pretrained(
            os.path.join(_HERE, "tokenizer"))

    def transcribe(self, audio, sr):
        """No transcript for the student (embedding-only); returns '' and
        timings. The command comes straight from classify()."""
        t0 = time.perf_counter()
        feat = self.fe(audio, sampling_rate=16000, return_tensors="np",
                       padding="max_length")["input_features"][0].astype(np.float32)
        n_frames = min(int(round(len(audio) / 16000 * 50)), 3000)
        mask = np.zeros((1, 3000), dtype=np.int64)
        mask[0, :n_frames] = 1
        t_mel = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        emb = self.enc.run(None, {"input_features": feat[None],
                                  "attention_mask": mask})[0]
        t_enc = (time.perf_counter() - t0) * 1e3
        return "", {"mel_ms": t_mel, "enc_ms": t_enc, "dec_ms": 0.0,
                    "dec_steps": 0, "_emb": emb}

    def classify(self, text, emb=None):
        t0 = time.perf_counter()
        logits = self.head.run(None, {"embedding": emb})[0]
        p = np.exp(logits) / np.exp(logits).sum()
        i = int(p.argmax())
        return self.classes[i], float(p[i]), (time.perf_counter() - t0) * 1e3


# --------------------------------------------------------------------------
# VAD (energy-based, adaptive noise floor)
# --------------------------------------------------------------------------
class VAD:
    FRAME_S = 0.030
    END_S = 0.60          # silence to end an utterance
    MAX_S = 12.0

    def __init__(self, threshold=0.0035):
        self.threshold = threshold
        self.noise = 0.001
        self.buf = []
        self.silence = 0.0
        self.speaking = False

    def rms(self, frame):
        return float(np.sqrt(np.mean(frame ** 2)) + 1e-9)

    def push(self, frame):
        """feed one frame; returns 'speech-start' | 'speech-end' | None."""
        r = self.rms(frame)
        if not self.speaking:
            # track noise floor
            self.noise = 0.95 * self.noise + 0.05 * r
            if r > max(self.threshold, 3.0 * self.noise):
                self.speaking = True
                self.buf = [frame]
                self.silence = 0.0
                return "speech-start"
            return None
        else:
            self.buf.append(frame)
            dur = len(self.buf) * self.FRAME_S
            if r < self.threshold:
                self.silence += self.FRAME_S
            else:
                self.silence = 0.0
            if self.silence >= self.END_S or dur >= self.MAX_S:
                self.speaking = False
                return "speech-end"
            return None

    def audio(self):
        a = np.concatenate(self.buf)
        self.buf = []
        return a


# --------------------------------------------------------------------------
# session metrics
# --------------------------------------------------------------------------
class Session:
    def __init__(self):
        self.n = 0
        self.asr = []
        self.e2e = []
        self.cmds = []

    def add(self, asr_ms, e2e_ms, cmd):
        self.n += 1
        self.asr.append(asr_ms)
        self.e2e.append(e2e_ms)
        self.cmds.append(cmd)

    def summary(self):
        if not self.n:
            return "no utterances yet"
        p = lambda xs, q: sorted(xs)[min(len(xs) - 1, int(q * len(xs)))]
        return (f"{self.n} utterances | ASR avg {np.mean(self.asr):.0f} ms "
                f"p50 {p(self.asr, .5):.0f} p95 {p(self.asr, .95):.0f} | "
                f"E2E avg {np.mean(self.e2e):.0f} ms "
                f"p50 {p(self.e2e, .5):.0f} p95 {p(self.e2e, .95):.0f} | "
                f"REJECT x{self.cmds.count('REJECT')}")


def run_file(args, backend, session):
    import soundfile as sf
    a, sr = sf.read(args.file, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    t_start = time.perf_counter()
    text, tim = backend.transcribe(a, sr)
    asr_ms = tim["mel_ms"] + tim["enc_ms"] + tim["dec_ms"]
    if isinstance(backend, Student):
        cmd, prob, cls_ms = backend.classify(text, emb=tim["_emb"])
    else:
        cmd, prob, cls_ms = backend.classify(text)
    e2e_ms = (time.perf_counter() - t_start) * 1e3
    session.add(asr_ms, e2e_ms, cmd)
    print(f"  transcript : {text!r}")
    print(f"  command    : {cmd}  (prob {prob:.3f})   intent: {_INTENT_OF.get(cmd, '?')}")
    print(f"  mel {tim['mel_ms']:.1f} ms | enc {tim['enc_ms']:.1f} ms | "
          f"dec {tim['dec_ms']:.1f} ms ({tim['dec_steps']} steps) | "
          f"classify {cls_ms:.1f} ms | ASR {asr_ms:.0f} ms | E2E {e2e_ms:.0f} ms")
    print("  " + session.summary())


def run_test(args, backend):
    import soundfile as sf
    test_dir = os.path.join(_HERE, "test_data", "additional_test_data")
    if not os.path.isdir(test_dir):
        test_dir = os.path.join(_ME2, "vcm-v2", "test_data", "additional_test_data")
    rows = []
    for folder in sorted(os.listdir(test_dir)):
        p = os.path.join(test_dir, folder)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if f.endswith(".wav"):
                rows.append((os.path.join(p, f), folder,
                             spoken_from_filename(f)))
    print(f"running {len(rows)} clips with {backend.name} ...", flush=True)
    results = []
    t0 = time.perf_counter()
    for k, (path, folder, spoken) in enumerate(rows):
        a, sr = sf.read(path, dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        t_start = time.perf_counter()
        text, tim = backend.transcribe(a, sr)
        asr_ms = tim["mel_ms"] + tim["enc_ms"] + tim["dec_ms"]
        if isinstance(backend, Student):
            cmd, prob, cls_ms = backend.classify(text, emb=tim["_emb"])
        else:
            cmd, prob, cls_ms = backend.classify(text)
        e2e_ms = (time.perf_counter() - t_start) * 1e3
        gold = gold_command(folder, spoken)
        results.append({"path": path, "folder": folder, "spoken": spoken,
                        "gold": gold, "pred": cmd, "prob": prob,
                        "transcript": text, "asr_ms": asr_ms,
                        "e2e_ms": e2e_ms,
                        "wer": wer(spoken, text) if text else None})
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(rows)} "
                  f"({(time.perf_counter() - t0) / (k + 1):.2f}s/clip)",
                  flush=True)
    n = len(results)
    cmd_acc = sum(r["pred"] == r["gold"] for r in results) / n
    intent_acc = sum(_INTENT_OF.get(r["pred"], "?") ==
                     _INTENT_OF.get(r["gold"], "?") for r in results) / n
    wers = [r["wer"] for r in results if r["wer"] is not None]
    asr = sorted(r["asr_ms"] for r in results)
    e2e = sorted(r["e2e_ms"] for r in results)
    report = {
        "backend": backend.name, "n_clips": n,
        "command_acc": round(cmd_acc, 4), "intent_acc": round(intent_acc, 4),
        "mean_wer": round(float(np.mean(wers)), 4) if wers else None,
        "median_wer": round(float(np.median(wers)), 4) if wers else None,
        "asr_ms_p50": round(asr[len(asr) // 2], 1),
        "asr_ms_p95": round(asr[int(0.95 * len(asr))], 1),
        "e2e_ms_p50": round(e2e[len(e2e) // 2], 1),
        "e2e_ms_p95": round(e2e[int(0.95 * len(e2e))], 1),
        "reject_pred": sum(r["pred"] == "REJECT" for r in results),
        "clips": results,
    }
    out = os.path.join(_HERE, "reports",
                       f"test_{args.model}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "clips"},
                     indent=2))
    print(f"report -> {out}")
    return report


def run_mic(args, backend, session):
    import sounddevice as sd
    vad = VAD()
    print(f"listening on mic (backend: {backend.name}) ... Ctrl-C to stop")
    print("say a command; ~0.6 s of silence ends the utterance.\n")
    frame = int(16000 * VAD.FRAME_S)

    def cb(indata, n, t, status):
        nonlocal vad
        if status:
            return
        ev = vad.push(indata[:, 0].astype(np.float32))
        if ev == "speech-end":
            a = vad.audio()
            if len(a) < 16000 * 0.3:      # < 300 ms: ignore
                return
            t_vad = (time.perf_counter() - t) * 1e3
            t_start = time.perf_counter()
            text, tim = backend.transcribe(a, 16000)
            asr_ms = tim["mel_ms"] + tim["enc_ms"] + tim["dec_ms"]
            if isinstance(backend, Student):
                cmd, prob, cls_ms = backend.classify(text, emb=tim["_emb"])
            else:
                cmd, prob, cls_ms = backend.classify(text)
            e2e_ms = (time.perf_counter() - t_start) * 1e3
            session.add(asr_ms, e2e_ms, cmd)
            print(f"[{time.strftime('%H:%M:%S')}] ── utterance #{session.n} "
                  f"({len(a) / 16000:.2f} s audio) ─────────────────")
            if text:
                print(f"  transcript : {text!r}")
            print(f"  command    : {cmd}  (prob {prob:.3f})   "
                  f"intent: {_INTENT_OF.get(cmd, '?')}")
            print(f"  mel {tim['mel_ms']:.1f} | enc {tim['enc_ms']:.1f} | "
                  f"dec {tim['dec_ms']:.1f} ({tim['dec_steps']} steps) | "
                  f"classify {cls_ms:.1f} | ASR {asr_ms:.0f} ms | "
                  f"E2E {e2e_ms:.0f} ms")
            print(f"  {session.summary()}\n", flush=True)

    with sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                        blocksize=frame, callback=cb):
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    print(f"\nsession summary: {session.summary()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["full", "student"], default="full")
    ap.add_argument("--fp32", action="store_true",
                    help="full model: use fp32 ONNX instead of int8")
    ap.add_argument("--test", action="store_true",
                    help="run the 171-clip held-out test set and exit")
    ap.add_argument("--file", help="transcribe a single wav file and exit")
    args = ap.parse_args()

    if args.model == "full":
        backend = FullWhisper(fp32=args.fp32)
    else:
        backend = Student()

    session = Session()
    if args.file:
        run_file(args, backend, session)
    elif args.test:
        run_test(args, backend)
    else:
        run_mic(args, backend, session)


if __name__ == "__main__":
    main()
