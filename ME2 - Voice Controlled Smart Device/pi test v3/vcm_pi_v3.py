#!/usr/bin/env python3
"""pi test v3 -- ME2 smart-device voice command listener WITH spoken responses.

The full loop the user asked for:

    say "hey rhasspy"  ->  wait for a command  ->  run the model  ->
    classify  ->  play the matching TTS wav  ->  wait for the next wake word
    ...  (until Ctrl-C)

A **wake word** ("hey rhasspy", openWakeWord) gates the whole pipeline. The
grammar-constrained decoder no longer runs continuously -- it only arms after
the wake word fires, which is what kills the ghost commands that a noisy mic
used to produce (noise can no longer decode into a command on its own).

The model is the BEST one from the vcm-v2 work: the **PocketSphinx ensemble**
(custom 1.6 MB LDA AM + stock 6.4 MB en-us AM, both decoding the same 103-phrase
JSGF command grammar, fused by agreement / stage-2-classifier confidence).
It scores **95.3% command / 97.1% intent** on the 171-clip held-out set
(`data/additional_test_data`, one new speaker) -- see
`archived/vcm-v2/backbone/reports/pocketsphinx_ensemble_cmudict.json`.

Once a command is classified, the device "answers" by playing the matching
response WAV from the TTS repo (16 kHz mono 16-bit, Piper en_US-lessac-medium).
The 31 commands map onto the 19 response phrases; the REJECT / unknown class
plays the generated "can you repeat that?" (`19_repeat.wav`).

This folder is self-contained: acoustic models, dictionary, JSGF grammar, the
stage-2 classifier, the `vcm`/`vcm2` code, the response WAVs, and the
openWakeWord "hey rhasspy" model (in `wakeword/`) all live here.

Usage
-----
    python vcm_pi_v3.py                 # live mic: wake word -> command -> speak
    python vcm_pi_v3.py --file clip.wav # classify one file, print the response
    python vcm_pi_v3.py --test          # run the held-out test set
    python vcm_pi_v3.py --no-play       # (mic) classify + print, skip playback
    python vcm_pi_v3.py --wake-threshold 0.6   # stricter wake-word gate
    python vcm_pi_v3.py --command-window 1.5   # wait 1.5 s for the command

Flow
----
    start  ->  STANDBY: wait for "hey rhasspy" (noise is ignored)
             ->  play the "yes?" cue (mic muted while it plays)
             ->  wait up to 1 s for you to START speaking
                 (0.6 s of silence then ends the utterance)
             ->  decode + classify + play the response
             ->  0.5 s cooldown  ->  back to STANDBY

The wake word is MANDATORY. If the openWakeWord model cannot be loaded (missing
file, unfetched Git-LFS pointer, or a failed self-test), the program REFUSES
to start with a loud error instead of silently falling back to always-listen
(the old behavior that produced ghost commands). Pass --no-wake to opt into
always-listen for debugging only.

Stop with Ctrl-C at any time.
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request
import wave
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from scipy.signal import resample_poly

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)          # so `import vcm` / `import vcm2` resolve locally

from vcm2.classifier import load_classifier, predict      # noqa: E402
from vcm2.ground_truth import build_ground_truth          # noqa: E402

# --------------------------------------------------------------------------
# paths (all relative to this folder)
# --------------------------------------------------------------------------
CUSTOM_HMM = os.path.join(_HERE, "am", "custom")
CUSTOM_LM = os.path.join(CUSTOM_HMM, "vcm.lm.bin")
STOCK_HMM = os.path.join(_HERE, "am", "stock")
STOCK_DICT = os.path.join(_HERE, "am", "stock_enus", "cmudict-en-us.dict")
STOCK_LM = os.path.join(_HERE, "am", "stock_enus", "en-us.lm.bin")
DICT3 = os.path.join(_HERE, "dict3")
JSGF = os.path.join(_HERE, "vcm_commands_enh3.jsgf")
CLASSIFIER = os.path.join(_HERE, "classifier.pkl")
RESP_DIR = os.path.join(_HERE, "responses")
YES_WAV = os.path.join(RESP_DIR, "00_yes.wav")   # "yes?" cue after the wake word
TEST_DATA = os.path.normpath(os.path.join(_HERE, "..", "data",
                                          "additional_test_data"))

# tuned decode params (identical to the eval that produced the 95.3% report)
CUSTOM_EXTRA = {"-silprob": "0.65", "-wip": "0.65"}
STOCK_EXTRA = {"-silprob": "0.45", "-wip": "0.60"}

# A freshly-created PocketSphinx decoder needs a few real-speech utterances to
# lock in (its feature/AGC state adapts over the first decodes). We re-decode
# the FIRST real command WARMUP_REPS times and take the converged result; this
# both fixes the first command and warms the decoder for all that follow.
WARMUP_REPS = 4

# After the device finishes speaking (or playing a canned response), it ignores
# the mic for this many seconds before accepting the next command, so the tail
# or echo of the response can't be heard as a new command. (The mic is already
# gated while the response plays; this covers the brief moment after it ends.)
COOLDOWN_S = 0.5

# --------------------------------------------------------------------------
# wake word ("hey rhasspy", openWakeWord) -- the gate that kills ghost commands
# --------------------------------------------------------------------------
# The decoder no longer runs continuously. The openWakeWord "hey rhasspy" model
# (204 KB ONNX, bundled in ./wakeword) watches the mic at all times; only once
# it fires does the VAD + PocketSphinx ensemble arm for the actual command.
# Noise / a zoom call can't produce a command on its own anymore, because the
# grammar-constrained decoder simply isn't listening until the wake word.
WAKEWORD_MODEL = os.path.join(_HERE, "wakeword", "hey_rhasspy_v0.1.onnx")
WAKE_THRESHOLD = 0.5    # openWakeWord score 0..1; "hey rhasspy" peaks ~0.8-0.9,
                        # noise and other phrases stay < 0.01 (verified)
WAKE_WINDOW = 5         # rolling 5 x 80 ms frames (0.4 s) the peak is taken over
COMMAND_WINDOW_S = 1.0  # after the 'yes?' cue, wait this long for the command
                        # to START, then go back to waiting for the wake word

# --------------------------------------------------------------------------
# command -> response WAV (from the TTS repo). 31 commands + REJECT.
# Several commands share a phrase (e.g. all brightness -> "setting brightness").
# 00_yes.wav is kept as a spare generic ack (not bound to a command).
# --------------------------------------------------------------------------
RESPONSE_WAV = {
    "PLAY_MUSIC": "01_playing_music.wav",
    "WEATHER": "02_current_weather.wav",
    "TIME": "03_current_time.wav",
    "LIGHT_ON": "04_switching_lights.wav",
    "LIGHT_OFF": "04_switching_lights.wav",
    "BRIGHTNESS_20": "16_setting_brightness.wav",
    "BRIGHTNESS_60": "16_setting_brightness.wav",
    "BRIGHTNESS_100": "16_setting_brightness.wav",
    "COLOR_RED": "17_changing_color.wav",
    "COLOR_GREEN": "17_changing_color.wav",
    "COLOR_BLUE": "17_changing_color.wav",
    "TIMER_10s": "13_setting_timer.wav",
    "TIMER_30s": "13_setting_timer.wav",
    "TIMER_1m": "13_setting_timer.wav",
    "ALARM_6_00AM": "14_setting_alarm.wav",
    "ALARM_8_00AM": "14_setting_alarm.wav",
    "ALARM_9_00PM": "14_setting_alarm.wav",
    "TEMPERATURE_18": "15_changing_temperature.wav",
    "TEMPERATURE_22": "15_changing_temperature.wav",
    "TEMPERATURE_26": "15_changing_temperature.wav",
    "PAUSE": "05_pausing.wav",
    "STOP": "06_stopping_playback.wav",
    "NEXT": "07_next_song.wav",
    "VOLUME_UP": "08_volume_up.wav",
    "VOLUME_DOWN": "09_volume_down.wav",
    "CALL": "10_calling.wav",
    "MESSAGE": "11_sending_message.wav",
    "LIST_REMINDERS": "12_reminders_list.wav",
    "CREATE_REMINDER_DRINK_WATER": "18_creating_reminder.wav",
    "CREATE_REMINDER_EXERCISE": "18_creating_reminder.wav",
    "CREATE_REMINDER_STUDY": "18_creating_reminder.wav",
    "REJECT": "19_repeat.wav",
}

# command -> intent (same map as backbone/pocketsphinx/eval_pocketsphinx.py)
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
    "CALL": "call", "MESSAGE": "reminders_lists",
    "REJECT": "reject",
}


# --------------------------------------------------------------------------
# PocketSphinx ensemble (the best model)
# --------------------------------------------------------------------------
def _make_decoder(hmm, dictionary, lm, jsgf, extra):
    from pocketsphinx import Config, Decoder
    cfg = Config()
    cfg.set_string("-hmm", hmm)
    cfg.set_string("-dict", dictionary)
    cfg.set_string("-lm", lm)
    cfg.set_string("-logfn", "/dev/null")
    cfg.set_string("-wip", "0.65")
    for k, v in (extra or {}).items():
        cfg.set_string(k, v)
    dec = Decoder(cfg)
    if jsgf:
        with open(jsgf) as f:
            dec.add_jsgf_string("vcm", f.read())
        dec.activate_search("vcm")
    return dec


def _decode_pcm(dec, pcm: bytes):
    """Feed 16 kHz mono 16-bit PCM bytes; return (transcript, utt_prob)."""
    dec.start_utt()
    chunk = 4000 * 2  # 4000 samples * 2 bytes
    for i in range(0, len(pcm), chunk):
        dec.process_raw(pcm[i:i + chunk], False, False)
    dec.end_utt()
    hyp = dec.hyp()
    text = (hyp.hypstr or "").strip() if hyp else ""
    return text, dec.get_prob()


class Ensemble:
    """custom AM + stock AM, same JSGF grammar, agree/confidence fusion."""

    def __init__(self):
        print("building ensemble decoders (custom + stock) ...", flush=True)
        self.dec_c = _make_decoder(CUSTOM_HMM, DICT3, CUSTOM_LM, JSGF, CUSTOM_EXTRA)
        self.dec_s = _make_decoder(STOCK_HMM, STOCK_DICT, STOCK_LM, JSGF, STOCK_EXTRA)
        self.clf = load_classifier(CLASSIFIER)
        print("ready.", flush=True)

    def classify_warm(self, pcm: bytes, reps: int = 1):
        """Classify; when reps>1, decode that many times and return the LAST
        result (converged). Used to warm a freshly-created decoder on the
        first real command (see WARMUP_REPS)."""
        out = None
        for _ in range(max(1, reps)):
            out = self.classify(pcm)
        return out

    def classify(self, pcm: bytes):
        """16 kHz mono 16-bit PCM -> (command, intent, transcript, clf_prob)."""
        tc, _pc = _decode_pcm(self.dec_c, pcm)
        ts, _ps = _decode_pcm(self.dec_s, pcm)
        cc, cpc = predict(self.clf, tc)
        cs, cps = predict(self.clf, ts)
        # fusion: agreement, else the more confident stage-2 classifier
        if cc == cs:
            cmd, transcript, cprob = cc, tc, cpc
        elif cpc >= cps:
            cmd, transcript, cprob = cc, tc, cpc
        else:
            cmd, transcript, cprob = cs, ts, cps
        return cmd, _INTENT_OF.get(cmd, "unknown"), transcript, cprob


# --------------------------------------------------------------------------
# audio helpers
# --------------------------------------------------------------------------
def read_pcm16(path: str) -> bytes:
    """Read a 16 kHz mono 16-bit WAV -> raw PCM bytes."""
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, f"expected 16 kHz, got {w.getframerate()}"
        assert w.getsampwidth() == 2, f"expected 16-bit, got {w.getsampwidth()}"
        return w.readframes(w.getnframes())


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    x = np.clip(audio, -1.0, 1.0) * 32767.0
    return x.astype(np.int16).tobytes()


def play_wav(path: str, enabled: bool = True) -> None:
    """Play a WAV (blocking). Degrades to a printed note if no audio device.

    Pi speakers (USB / I2S) usually only support 44.1/48 kHz, so opening the
    stream at the wav's 16 kHz fails with "Invalid sample rate". Play at the
    device's native rate, resampling in software; fall back to 16 kHz if the
    native rate is also rejected.
    """
    if not enabled:
        print(f"  >> [no-play] {os.path.basename(path)}")
        return
    try:
        import soundfile as sf
        import sounddevice as sd
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        try:
            out_sr = int(sd.query_devices(sd.default.device[1], "output")
                         ["default_samplerate"])
        except Exception:
            out_sr = sr
        last_err = None
        for target in dict.fromkeys([out_sr, 16000]):
            d = data
            if target != sr:
                g = int(np.gcd(sr, target))
                d = resample_poly(data, int(target) // g, int(sr) // g)
            try:
                sd.play(d, target)
                sd.wait()
                return
            except Exception as e:
                last_err = e
        print(f"  >> [playback failed: {type(last_err).__name__}] would play "
              f"{os.path.basename(path)}")
    except Exception as e:  # noqa: BLE001 - headless / no PortAudio
        print(f"  >> [no audio device: {type(e).__name__}] would play "
              f"{os.path.basename(path)}")


# --------------------------------------------------------------------------
# dynamic TTS (Piper -- ultra-lightweight local neural TTS)
#
# Used for responses that change every time (e.g. the current time). Piper is
# a small ONNX model (~63 MB) that runs entirely on-device; the voice is
# downloaded ONCE into ./tts/ on first run (needs internet that one time),
# after which everything is offline. The voice is en_US-lessac-low -- the
# lightest quality tier of the same voice family the pre-recorded response
# WAVs use (lessac-medium), so it sounds consistent.
#
# piper-tts 1.8.x ships a ready aarch64 wheel (cp39-abi3), so it installs
# cleanly on the Pi's Python 3.13.
# --------------------------------------------------------------------------
TTS_DIR = os.path.join(_HERE, "tts")
TTS_VOICE = "en_US-lessac-low"
TTS_VOICE_URL = ("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                 f"en/en_US/lessac/low/{TTS_VOICE}.onnx")
TTS_VOICE_JSON_URL = TTS_VOICE_URL + ".json"


def _ensure_tts_voice() -> str:
    """Download the Piper voice into ./tts/ once; return the .onnx path."""
    os.makedirs(TTS_DIR, exist_ok=True)
    onnx = os.path.join(TTS_DIR, f"{TTS_VOICE}.onnx")
    jso = os.path.join(TTS_DIR, f"{TTS_VOICE}.onnx.json")
    for dst, url in ((onnx, TTS_VOICE_URL), (jso, TTS_VOICE_JSON_URL)):
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        print(f"  downloading TTS voice -> {os.path.basename(dst)} "
              f"(one-time, ~63 MB)")
        urllib.request.urlretrieve(url, dst)
    return onnx


def _load_tts():
    """Load the Piper voice once per process (lazy). None if unavailable."""
    global _TTS_VOICE
    if _TTS_VOICE is None:
        try:
            from piper import PiperVoice
            _TTS_VOICE = PiperVoice.load(_ensure_tts_voice())
        except Exception as e:  # noqa: BLE001 - offline / not installed
            print(f"  >> [TTS unavailable: {type(e).__name__}] "
                  f"falling back to the canned response")
            _TTS_VOICE = False
    return _TTS_VOICE or None


_TTS_VOICE = None


def speak(text: str, enabled: bool = True) -> None:
    """Synthesize `text` with Piper and play it (blocking).

    Falls back to printing the text if piper-tts is missing or there is no
    audio device.
    """
    if not enabled:
        print(f"  >> [no-play] would say: {text!r}")
        return
    voice = _load_tts()
    if voice is None:
        print(f"  >> [TTS unavailable] would say: {text!r}")
        return
    import io
    import soundfile as sf
    import sounddevice as sd
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        voice.synthesize_wav(text, w)
    buf.seek(0)
    data, sr = sf.read(buf, dtype="float32")
    try:
        out_sr = int(sd.query_devices(sd.default.device[1], "output")
                     ["default_samplerate"])
    except Exception:
        out_sr = sr
    last_err = None
    for target in dict.fromkeys([out_sr, 16000]):
        d = data
        if target != sr:
            g = int(np.gcd(sr, target))
            d = resample_poly(data, int(target) // g, int(sr) // g)
        try:
            sd.play(d, target)
            sd.wait()
            return
        except Exception as e:
            last_err = e
    print(f"  >> [playback failed: {type(last_err).__name__}] would say: "
          f"{text!r}")


def time_response_text() -> str:
    """Current time in UTC+8, 12-hour format, as a spoken sentence.

    e.g. "It is 5:42 PM."  (UTC+8 = the device's timezone, Asia/Manila)
    """
    now = datetime.now(ZoneInfo("Asia/Manila"))
    return f"It is {now.strftime('%I:%M %p')}."


# --------------------------------------------------------------------------
# VAD (webrtcvad + absolute energy gate -- noise-robust endpointing)
#
# The old energy VAD ended an utterance only when the mic's RMS dropped below
# an ABSOLUTE threshold (0.0035). Any continuous background noise (e.g. a
# zoom call) kept the RMS above it, so every utterance ran to the 12 s hard
# cap and the perceived latency was ~12 s + decode. webrtcvad separates
# speech from stationary background noise, so the 0.6 s trailing-silence rule
# now fires on your actual pause even with a call playing.
#
# webrtcvad alone still let a *noisy* mic start utterances on noise bursts
# ("ghost commands"). So a frame now counts as speech only if webrtcvad says
# so AND its RMS is above an ABSOLUTE gate (GATE, -26 dBFS). The gate was
# calibrated on the 171-clip additional_test_data set: recording noise
# floors sit at RMS ~0.038-0.057 (-28.5..-25 dBFS), while webrtcvad-voiced
# speech frames sit at p5 = 0.040 / p50 = 0.068, and every clip's peak
# speech frame is >= 0.086. At GATE=0.05 all 171 clips still yield exactly
# one utterance each (0 splits, 0 misses, lengths unchanged); at 0.06 real
# speech starts getting split (2 clips). Raise --gate if ghost commands
# persist on a noisier mic; lower it if quiet commands get dropped.
# --------------------------------------------------------------------------
class WakeWord:
    """openWakeWord "hey rhasspy" detector, fed 80 ms (1280-sample) 16 kHz
    int16 frames. Returns the peak score over a rolling WAKE_WINDOW frames."""

    def __init__(self, model_path: str, threshold: float = WAKE_THRESHOLD):
        from openwakeword.model import Model
        self.threshold = threshold
        self.model = Model(wakeword_model_paths=[model_path])
        self._scores = []

    def reset(self):
        self.model.reset()
        self._scores = []

    def push(self, block16: np.ndarray) -> float:
        """Feed one 1280-sample int16 block; returns the rolling peak score."""
        s = self.model.predict(block16)
        v = float(max(s.values()))
        self._scores.append(v)
        if len(self._scores) > WAKE_WINDOW:
            self._scores.pop(0)
        return max(self._scores)

    def fired(self, peak: float) -> bool:
        return peak >= self.threshold


# The wake model is tracked in Git LFS. On a machine where `git lfs pull`
# never ran (or git-lfs isn't installed), the file on disk is a ~131-byte
# text pointer, not the 204 KB model -- and the wake word silently can't
# fire. GitHub serves the real LFS object transparently at the raw URL, so
# we can self-heal: detect the pointer and download the real model.
_WAKE_MODEL_RAW = ("https://github.com/raesoriano/Machine-Learning-Operations/"
                   "raw/main/ME2%20-%20Voice%20Controlled%20Smart%20Device/"
                   "pi%20test%20v3/wakeword/hey_rhasspy_v0.1.onnx")


def _heal_wake_model(model_path: str) -> str:
    """Return '' if the model file is usable, else a status string.

    If the file is missing or is an unfetched Git-LFS pointer (sub-KB), try
    to download the real model from GitHub's raw endpoint (which serves the
    LFS object directly). This makes the device self-sufficient on the Pi:
    no `git lfs pull` / git-lfs install required.
    """
    import urllib.parse
    if os.path.exists(model_path) and os.path.getsize(model_path) >= 1000:
        return ""
    why = ("not found" if not os.path.exists(model_path)
           else f"is a {os.path.getsize(model_path)}-byte Git-LFS pointer")
    print(f">> wake word model {why} -- downloading the real model "
          f"from GitHub ...")
    tmp = model_path + ".dl"
    try:
        req = urllib.request.Request(_WAKE_MODEL_RAW,
                                     headers={"User-Agent": "vcm-pi-v3"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        if len(data) < 10000:
            return (f"download returned only {len(data)} bytes (expected "
                    "~204081) -- not the model. Run `git lfs pull` in the "
                    "repo, then retry")
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, model_path)
        print(f">> wake word model downloaded OK ({len(data)} bytes)")
        return ""
    except Exception as e:  # noqa: BLE001
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return (f"auto-download failed: {type(e).__name__}: {e} -- run "
                "`git lfs pull` in the repo (or copy the 204 KB "
                "hey_rhasspy_v0.1.onnx into wakeword/), then retry")


def wake_selftest(model_path: str, threshold: float) -> tuple:
    """Feed the bundled 'hey rhasspy' self-test wav through the model and
    report (peak_score, passed, reason). This proves the wake model actually
    loaded and can fire -- a silent load failure (e.g. a missing onnxruntime
    / tflite backend on the Pi) would otherwise leave the device
    always-listening and produce ghost commands. `reason` names the exact
    step that failed so the Pi output is actionable."""
    import soundfile as sf
    heal = _heal_wake_model(model_path)
    if heal:
        return 0.0, False, heal
    if not os.path.exists(model_path):
        return 0.0, False, f"model file not found: {model_path}"
    size = os.path.getsize(model_path)
    if size < 1000:
        return 0.0, False, (
            f"model file is {size} bytes -- an unfetched Git-LFS pointer, "
            "not the model. Run `git lfs pull` in the repo, then retry")
    try:
        import onnxruntime  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return 0.0, False, (
            f"onnxruntime import failed: {type(e).__name__}: {e} -- run "
            "`pip install -r requirements.txt`")
    try:
        import openwakeword  # noqa: F401
        import openwakeword as _ow
        import importlib.metadata as _imd
        ver = _imd.version("openwakeword")
    except Exception as e:  # noqa: BLE001
        return 0.0, False, (
            f"openwakeword import failed: {type(e).__name__}: {e} -- run "
            "`pip install -r requirements.txt` (needs openwakeword==0.4.0)")
    fixture = os.path.join(os.path.dirname(model_path),
                           "selftest_hey_rhasspy.wav")
    if not os.path.exists(fixture):
        return 0.0, False, (
            f"self-test fixture not found: {fixture} -- run `git lfs pull` "
            "or re-clone")
    try:
        a, sr = sf.read(fixture, dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample_poly
            g = int(np.gcd(sr, 16000))
            a = resample_poly(a, 16000 // g, sr // g).astype(np.float32)
        ww = WakeWord(model_path, threshold=threshold)
    except Exception as e:  # noqa: BLE001 - load failure
        return 0.0, False, (
            f"model failed to load (openwakeword {ver}): "
            f"{type(e).__name__}: {e}")
    try:
        pcm = (np.clip(a, -1.0, 1.0) * 32767).astype(np.int16)
        peak = 0.0
        for i in range(0, len(pcm) - 1279, 1280):
            peak = max(peak, ww.push(pcm[i:i + 1280]))
        return peak, peak >= threshold, (
            f"openwakeword {ver}, model {size} bytes, "
            f"{len(pcm) // 1280} frames fed")
    except Exception as e:  # noqa: BLE001 - inference failure
        return 0.0, False, (
            f"inference failed (openwakeword {ver}): "
            f"{type(e).__name__}: {e}")


class StreamDecimate:
    """48 kHz -> 16 kHz streaming lowpass decimator (factor 3).

    The Pi mic delivers 48 kHz but the wake-word model wants 16 kHz. Resampling
    each 30 ms callback independently corrupts the signal (filter edges reset
    every callback -> the model sees noise). This keeps the FIR filter state
    across callbacks, so the stream is sample-accurate.
    """

    def __init__(self, factor: int = 3, ntaps: int = 33):
        from scipy.signal import firwin
        self.factor = factor
        self.taps = firwin(ntaps, 1.0 / factor, window="hamming")
        self.zi = np.zeros(len(self.taps) - 1)
        self.idx = 0

    def push(self, x: np.ndarray) -> np.ndarray:
        from scipy.signal import lfilter
        y, self.zi = lfilter(self.taps, 1.0, x, zi=self.zi)
        out = y[self.idx::self.factor]
        self.idx = (self.idx + len(x)) % self.factor
        return out.astype(np.float32)


class WakeGate:
    """Three-stage state machine: WAKE (listen for 'hey rhasspy') -> CUE
    (the "yes?" cue plays; mic muted) -> COMMAND (listen for the actual
    command) -> WAKE.

    The mic callback feeds it native-rate audio; it resamples to 16 kHz,
    accumulates 80 ms blocks for the wake detector, and hands 30 ms frames to
    the command VAD once armed. The command window (self.window seconds)
    starts when the "yes?" cue FINISHES (cue_done), not when the wake word
    fires, so the full window is available for the user to start speaking.
    """

    WAKE, CUE, COMMAND = "wake", "cue", "command"

    def __init__(self, sr: int, wake: WakeWord, vad, gate_rms: float = 0.0,
                 window: float = COMMAND_WINDOW_S):
        self.sr = sr
        self.wake = wake
        self.vad = vad
        self.gate_rms = gate_rms
        self.window = window
        self.state = self.WAKE
        # 48 kHz (the Pi mic) -> 16 kHz via a stateful decimator. 16 kHz passes
        # straight through. Any other rate falls back to a per-callback
        # resample (best effort; the Pi is 48 kHz, so this is rarely used).
        self._dec = StreamDecimate(3) if sr == 48000 else None
        self._passthrough = (sr == 16000)
        self._acc = np.zeros(0, np.float32)     # 16 kHz accumulator
        self._armed_at = 0.0
        self._fired = False
        self.last_audio = None                  # captured command (16 kHz)

    def push(self, frame: np.ndarray, now: float):
        """Feed one native-rate float32 frame (sr * 0.030 samples).
        Returns 'wake' | 'command' | None. On 'command', `last_audio` holds
        the captured utterance at 16 kHz (resample before decoding)."""
        if self.state == self.WAKE:
            return self._push_wake(frame)
        if self.state == self.CUE:
            # The "yes?" cue is still playing (mic muted by the main thread).
            # Nothing to do until cue_done() is called.
            return None
        # COMMAND state: the window is the time allowed for speech to START
        # (self.window s after the cue). If it expires with no speech in
        # progress, go back to standby; if speech is in progress, let it
        # finish (the VAD's 0.6 s silence rule ends the utterance).
        if now - self._armed_at > self.window and not self.vad.speaking:
            self._to_wake()
            return None
        ev = self.vad.push(frame)
        if ev == "speech-end":
            a = self.vad.audio()                # clears the VAD buffer
            if len(a) >= self.sr * 0.3:         # >= 300 ms: a real command
                self._to_wake()
                self.last_audio = a
                return "command"
            # too short (a blip / noise burst): discard and stay armed, so a
            # real command that follows within the window is still captured.
            self.vad.reset()
            self.last_audio = None
            return None
        return None

    def _push_wake(self, frame: np.ndarray):
        if self._dec is not None:
            r16 = self._dec.push(frame)
        elif self._passthrough:
            r16 = frame                      # already 16 kHz
        else:
            # other native rate: best-effort per-callback resample to 16 kHz
            from scipy.signal import resample_poly
            r16 = resample_poly(frame, 16000, self.sr).astype(np.float32)
        self._acc = np.concatenate([self._acc, r16])
        while len(self._acc) >= 1280:
            block = self._acc[:1280]
            self._acc = self._acc[1280:]
            peak = self.wake.push((np.clip(block, -1.0, 1.0) * 32767)
                                  .astype(np.int16))
            if self.wake.fired(peak):
                self._to_cue()
                return "wake"
        return None

    def _to_cue(self):
        # Wake word fired. The main thread plays the "yes?" cue (mic muted)
        # and then calls cue_done() to arm the command window.
        self.state = self.CUE
        self._fired = True
        self.last_audio = None

    def cue_done(self, now: float = None):
        """Called by the main thread once the "yes?" cue has finished.
        Arms the command window (self.window s for speech to start). The
        wake word's own tail was already dropped: the mic is muted while the
        cue plays, so no tail audio reaches the VAD."""
        self.state = self.COMMAND
        self._armed_at = time.time() if now is None else now
        self.vad.reset()
        self.last_audio = None

    def _to_wake(self):
        self.state = self.WAKE
        self._fired = False
        self.wake.reset()
        self._acc = np.zeros(0, np.float32)
        if self._dec is not None:
            self._dec = StreamDecimate(3)
        self.vad.reset()

    @property
    def armed(self):
        return self.state == self.COMMAND


class VAD:
    FRAME_S = 0.030
    END_S = 0.60          # trailing silence to end an utterance
    MAX_S = 4.0           # hard cap: worst-case wait is 4 s, not 12
    MODE = 2              # aggressiveness 0..3; 2 = strong noise rejection
    GATE = 0.05           # absolute min frame RMS to count as speech (-26 dBFS)

    def __init__(self, rate, gate=None):
        import webrtcvad
        self._vad = webrtcvad.Vad(self.MODE)
        self.rate = rate
        self.GATE = float(gate) if gate is not None else self.GATE
        self.reset()

    def reset(self):
        self.buf = []
        self.silence = 0.0
        self.speaking = False

    def push(self, frame):
        """feed one float32 frame at self.rate; 'speech-start'|'speech-end'|None."""
        pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        voiced = self._vad.is_speech(pcm, self.rate)
        if voiced and float(np.sqrt(np.mean(frame * frame))) < self.GATE:
            voiced = False          # absolute gate: too quiet to be speech
        if not self.speaking:
            if voiced:
                self.speaking = True
                self.buf = [frame]
                self.silence = 0.0
                return "speech-start"
            return None
        self.buf.append(frame)
        if voiced:
            self.silence = 0.0
        else:
            self.silence += self.FRAME_S
        if self.silence >= self.END_S or len(self.buf) * self.FRAME_S >= self.MAX_S:
            self.speaking = False
            return "speech-end"
        return None

    def audio(self):
        a = np.concatenate(self.buf)
        self.buf = []
        return a


# --------------------------------------------------------------------------
# the loop: wait -> classify -> speak -> wait ...
# --------------------------------------------------------------------------
def _abort_no_wake(reason: str, fix: str):
    """Refuse to start when the wake word is not active.

    The device's contract is: STANDBY by default, and NO command is accepted
    until "hey rhasspy" is heard. If the wake model cannot be loaded, running
    the continuous decoder (the old always-listen behavior) would violate that
    contract and is exactly what produced the ghost commands. So instead of
    silently degrading, we exit with a loud, actionable error. Pass --no-wake
    to opt into always-listen for debugging only.
    """
    print("\n" + "!" * 72)
    print("  WAKE WORD NOT ACTIVE -- refusing to start.")
    print(f"  reason : {reason}")
    print(f"  fix    : {fix}")
    print("  The device must be in STANDBY (waiting for 'hey rhasspy') before")
    print("  it accepts any command. It will NOT run in always-listen mode")
    print("  unless you explicitly pass --no-wake (debugging only).")
    print("!" * 72 + "\n")
    sys.exit(2)


def run_mic(args):
    import sounddevice as sd
    from scipy.signal import resample_poly
    model = Ensemble()
    state = {"busy": False, "audio": None, "cue": False, "muted": False}

    # Pi mics (USB / I2S) usually only support 44.1/48 kHz, so opening the
    # stream at 16 kHz fails with "Invalid sample rate". Capture at the
    # device's native rate and resample each finished utterance to 16 kHz
    # (the model's rate). VAD timing is unaffected (frames stay 30 ms).
    try:
        sr = int(sd.query_devices(sd.default.device[0], "input")
                 ["default_samplerate"])
    except Exception:
        sr = 16000
    if sr != 16000:
        print(f"mic native rate {sr} Hz -> resampling to 16000 Hz")

    # wake word detector (openWakeWord "hey rhasspy")
    wake = None
    if getattr(args, "no_wake", False):
        print("wake word : DISABLED (--no-wake) -- always listening "
              "(DEBUG ONLY)")
    else:
        # Self-heal: if the model file is missing or is an unfetched
        # Git-LFS pointer (sub-KB), download the real model from GitHub's
        # raw endpoint (serves the LFS object directly) -- no `git lfs
        # pull` / git-lfs install needed on the Pi.
        if (not os.path.exists(WAKEWORD_MODEL)
                or os.path.getsize(WAKEWORD_MODEL) < 1000):
            heal = _heal_wake_model(WAKEWORD_MODEL)
            if heal:
                _abort_no_wake(
                    f"wake word model: {heal}",
                    "run `git lfs pull` in the repo (or copy the 204 KB "
                    "hey_rhasspy_v0.1.onnx into wakeword/), then restart")
        try:
            wake = WakeWord(WAKEWORD_MODEL, threshold=args.wake_threshold)
            # PROVE the model actually loaded and can fire: feed the bundled
            # 'hey rhasspy' fixture through it. A silent load failure (missing
            # onnxruntime / tflite backend on the Pi) would otherwise leave
            # the device always-listening -> ghost commands.
            peak, ok, why = wake_selftest(WAKEWORD_MODEL, args.wake_threshold)
            if ok:
                print(f"wake word : 'hey rhasspy' (openWakeWord, "
                      f"threshold {wake.threshold:.2f}) | self-test "
                      f"score {peak:.2f} >= {args.wake_threshold:.2f} OK")
            else:
                _abort_no_wake(
                    f"wake word self-test FAILED: score {peak:.3f} "
                    f"< {args.wake_threshold:.2f} -- {why}",
                    "run `pip install -r requirements.txt` (needs "
                    "openwakeword==0.4.0 + onnxruntime), then restart")
        except Exception as e:  # noqa: BLE001 - onnxruntime / model issue
            _abort_no_wake(
                f"wake word model failed to load: "
                f"{type(e).__name__}: {e}",
                "run `pip install -r requirements.txt` (needs "
                "openwakeword==0.4.0 + onnxruntime), then restart")

    vad = VAD(sr, gate=args.gate)
    gate = (WakeGate(sr, wake, vad, gate_rms=vad.GATE,
                     window=args.command_window)
            if wake is not None else None)
    frame = int(sr * VAD.FRAME_S)
    n = 0

    # Show the live mic noise floor next to the gate so a noisy mic is
    # visible at a glance (if the floor prints ABOVE the gate, raise --gate).
    try:
        import sounddevice as _sd
        _floor = []
        def _probe(indata, nframes, t, status):
            if not status:
                _floor.append(np.sqrt(np.mean(indata[:, 0].astype(np.float32)
                                              ** 2)))
        with _sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                             blocksize=frame, callback=_probe):
            time.sleep(1.5)
        if _floor:
            fl = float(np.percentile(np.array(_floor), 90))
            print(f"mic noise floor (1.5 s): {fl:.4f} RMS "
                  f"({20 * math.log10(fl + 1e-9):.1f} dBFS) | "
                  f"speech gate: {vad.GATE:.4f} "
                  f"({20 * math.log10(vad.GATE + 1e-9):.1f} dBFS)")
    except Exception:
        pass

    def cb(indata, nframes, t, status):
        if status or state["busy"] or state["muted"]:
            return
        f = indata[:, 0].astype(np.float32)
        if gate is not None:
            ev = gate.push(f, time.time())
            if ev == "wake":
                print(f"\n[{time.strftime('%H:%M:%S')}] wake word detected -- "
                      f"'yes?' cue, then say your command "
                      f"({gate.window:.0f} s window)")
                state["cue"] = True            # main thread plays the cue
            elif ev == "command":
                a = gate.last_audio
                if a is None or len(a) < sr * 0.3:   # < 300 ms: ignore
                    return
                if sr != 16000:
                    a = resample_poly(a, 16000, sr)
                state["audio"] = a
                state["busy"] = True       # main thread takes over
        else:
            ev = vad.push(f)
            if ev == "speech-end":
                a = vad.audio()
                if len(a) < sr * 0.3:      # < 300 ms: ignore
                    return
                if sr != 16000:
                    a = resample_poly(a, 16000, sr)
                state["audio"] = a
                state["busy"] = True       # main thread takes over

    print("listening on mic ... Ctrl-C to stop")
    if gate is not None:
        print(f"STANDBY -- say 'hey rhasspy' to wake the device, then speak "
              f"your command within {gate.window:.0f} s of the 'yes?' cue. "
              f"Noise is ignored while in standby.\n")
    else:
        print("ALWAYS-LISTEN (no wake word) -- say a command; ~0.6 s of "
              "silence ends the utterance.\n")
    with sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                        blocksize=frame, callback=cb):
        try:
            while True:
                if state["cue"]:
                    state["cue"] = False
                    state["muted"] = True          # close the mic while "yes?"
                                                  # plays (echo guard)
                    print("  >> 'yes?' -- ready for your command")
                    play_wav(YES_WAV, enabled=not args.no_play)
                    state["muted"] = False
                    gate.cue_done()                # NOW start the 1 s window
                    continue
                if not state["busy"]:
                    time.sleep(0.01)
                    continue
                a = state["audio"]
                state["audio"] = None
                vad.reset()
                n += 1
                reps = WARMUP_REPS if n == 1 else 1   # warm the fresh decoder
                t0 = time.perf_counter()
                cmd, intent, transcript, cprob = model.classify_warm(
                    float32_to_pcm16(a), reps=reps)
                e2e = (time.perf_counter() - t0) * 1000.0
                print(f"[{time.strftime('%H:%M:%S')}] ── utterance #{n} "
                      f"({len(a) / 16000:.2f} s) ─────────────────────")
                print(f"  transcript : {transcript!r}")
                print(f"  command    : {cmd}   intent: {intent}")
                print(f"  E2E {e2e:.0f} ms")
                if cmd == "TIME":
                    # dynamic response: say the ACTUAL current time (UTC+8)
                    text = time_response_text()
                    print(f"  >> saying (Piper TTS): {text!r}")
                    speak(text, enabled=not args.no_play)
                else:
                    wav = RESPONSE_WAV.get(cmd, RESPONSE_WAV["REJECT"])
                    print(f"  >> playing {wav}")
                    play_wav(os.path.join(RESP_DIR, wav),
                             enabled=not args.no_play)
                time.sleep(COOLDOWN_S)    # ignore the mic briefly after the
                                          # response ends (tail / echo guard)
                state["busy"] = False     # re-arm: wait for the next command
        except KeyboardInterrupt:
            pass
    print(f"\nstopped after {n} command(s).")


def run_file(args):
    model = Ensemble()
    pcm = read_pcm16(args.file)
    t0 = time.perf_counter()
    cmd, intent, transcript, cprob = model.classify_warm(pcm, reps=WARMUP_REPS)
    e2e = (time.perf_counter() - t0) * 1000.0
    print(f"  file       : {args.file}")
    print(f"  transcript : {transcript!r}")
    print(f"  command    : {cmd}   intent: {intent}")
    print(f"  E2E {e2e:.0f} ms")
    if cmd == "TIME":
        text = time_response_text()
        print(f"  >> saying (Piper TTS): {text!r}")
        speak(text, enabled=not args.no_play)
    else:
        wav = RESPONSE_WAV.get(cmd, RESPONSE_WAV["REJECT"])
        print(f"  >> playing {wav}")
        play_wav(os.path.join(RESP_DIR, wav), enabled=not args.no_play)


def run_test(args):
    model = Ensemble()
    rows = build_ground_truth(args.data)
    print(f"{len(rows)} clips in {args.data}\n")
    correct = intent_correct = 0
    per_folder = {}
    t0 = time.perf_counter()
    for k, r in enumerate(rows):
        gold_intent = _INTENT_OF.get(r["gold"], "unknown")
        pcm = read_pcm16(r["path"])
        cmd, intent, transcript, cprob = model.classify(pcm)
        ok = cmd == r["gold"]
        iok = intent == gold_intent
        correct += ok
        intent_correct += iok
        f = per_folder.setdefault(r["folder"], {"n": 0, "cmd": 0, "intent": 0})
        f["n"] += 1
        f["cmd"] += ok
        f["intent"] += iok
        mark = "  " if ok else "!!"
        print(f"{mark} {r['folder']:16s} {r['spoken']!r:34s} -> "
              f"{cmd:28s} (gold {r['gold']:28s}) [{RESPONSE_WAV.get(cmd, '?')}]")
    n = len(rows)
    wall = time.perf_counter() - t0
    report = {
        "model": "PocketSphinx ensemble (custom + stock, agree+clf fusion)",
        "n_clips": n,
        "command_acc": round(correct / n, 4),
        "intent_acc": round(intent_correct / n, 4),
        "wall_s": round(wall, 1),
        "per_folder": {k: {"n": v["n"],
                           "cmd_acc": round(v["cmd"] / v["n"], 4),
                           "intent_acc": round(v["intent"] / v["n"], 4)}
                       for k, v in sorted(per_folder.items())},
    }
    out = os.path.join(_HERE, "test_v3_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\ncommand_acc {report['command_acc']:.4f}  "
          f"intent_acc {report['intent_acc']:.4f}  "
          f"({wall:.1f} s, {n / wall:.1f} clips/s)")
    print(f"report -> {out}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", help="classify a single wav file and exit")
    ap.add_argument("--test", action="store_true",
                    help="run the 171-clip held-out test set and exit")
    ap.add_argument("--data", default=TEST_DATA,
                    help="test-set directory (for --test)")
    ap.add_argument("--no-play", action="store_true",
                    help="print the response instead of playing it")
    ap.add_argument("--gate", type=float, default=None,
                    help="absolute min frame RMS (0..1) to count as speech; "
                         "default 0.05 (-26 dBFS), calibrated on the "
                         "171-clip test set. Raise it (e.g. 0.07) if a noisy "
                         "mic still triggers ghost commands.")
    ap.add_argument("--wake-threshold", type=float, default=WAKE_THRESHOLD,
                    help="openWakeWord score (0..1) that counts as the wake "
                         "word; default 0.5. 'hey rhasspy' peaks ~0.8-0.9, "
                         "noise stays < 0.01, so 0.5 is very safe.")
    ap.add_argument("--command-window", type=float, default=COMMAND_WINDOW_S,
                    help="seconds to wait (after the 'yes?' cue) for the "
                         "command to start; default 1.0. If no speech "
                         "starts in that time, the device returns to "
                         "standby (waiting for the wake word).")
    ap.add_argument("--no-wake", action="store_true",
                    help="disable the wake word and always listen. "
                         "DEBUGGING ONLY -- without the wake word the "
                         "device accepts commands from noise (ghost "
                         "commands). By default the program REFUSES to "
                         "start if the wake word cannot be loaded.")
    ap.add_argument("--wake-check", action="store_true",
                    help="run ONLY the wake-word self-test (load the model, "
                         "feed the bundled 'hey rhasspy' fixture, print the "
                         "score) and exit. Use this on the Pi to confirm the "
                         "wake word can actually fire before going live.")
    args = ap.parse_args()

    if args.wake_check:
        peak, ok, why = wake_selftest(WAKEWORD_MODEL, args.wake_threshold)
        print(f"self-test: score {peak:.3f} (threshold "
              f"{args.wake_threshold:.2f}) -> {'PASS' if ok else 'FAIL'}")
        print(f"  detail: {why}")
        if not ok:
            print("The wake model is not firing. The detail line above says "
                  "exactly which step failed (missing model / LFS pointer / "
                  "import / load / inference). The program already tries to "
                  "auto-download the model from GitHub when it finds an "
                  "unfetched LFS pointer. Remaining fixes: `git lfs pull` "
                  "(model files are stored in Git LFS) and "
                  "`pip install -r requirements.txt` (openwakeword==0.4.0 + "
                  "onnxruntime).")
        sys.exit(0 if ok else 1)
    elif args.file:
        run_file(args)
    elif args.test:
        run_test(args)
    else:
        run_mic(args)


if __name__ == "__main__":
    main()
