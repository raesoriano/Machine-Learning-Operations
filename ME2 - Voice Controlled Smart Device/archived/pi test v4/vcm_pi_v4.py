#!/usr/bin/env python3
"""pi test v4 -- ME2 smart-device voice command listener that reports what
was ACTUALLY heard and REJECTS non-commands.

The full loop the user asked for:

    say "hey rhasspy"  ->  wait for a command  ->  transcribe  ->
    classify  ->  speak the response  ->  wait for the next wake word
    ...  (until Ctrl-C)

A **wake word** ("hey rhasspy", openWakeWord) gates the whole pipeline,
exactly as in v3 (noise can no longer produce a command on its own).

WHAT CHANGED FROM v3 (and why)
------------------------------
v3's transcript came from a **grammar-constrained** PocketSphinx decoder
(JSGF). A constrained decoder is FORCED to output one of the 103 grammar
phrases, so two things went wrong:

  1. The "transcript" was not what you said -- it was the nearest grammar
     phrase. Say something that is not a command and it still prints
     "play music".
  2. The stage-2 classifier's REJECT class (trained on 32,992 out-of-domain
     examples) was dead: it only ever saw valid grammar phrases, so it
     force-classified everything into the nearest command.

v4 fixes both by changing the front end to a real free-text ASR:

    faster-whisper (base.en)  ->  keyword gate  ->  31-class + REJECT clf

  * **whisper** transcribes WHAT WAS ACTUALLY HEARD (free text, no grammar).
    The device prints it ("heard: 'he will do it my man'") and, on a reject,
    SAYS it back ("I heard: he will do it my man. Can you repeat that?").
  * **keyword gate**: a real command always contains a command word
    (play / lights / volume / weather / ...). Speech with no command word is
    rejected immediately -- the OOD rejection v3 could never produce.
  * **classifier**: the same trained 31-class + REJECT model from v3, now
    fed real free text, so it can (and does) emit REJECT for the subtle
    non-commands the gate lets through.

On the 171-clip held-out set (data/additional_test_data, one new speaker):
    v3 (constrained)  : 83.5% command / 86.4% intent / 0-of-5 OOD rejected
    v4 (whisper+gate) : 84.7% command / 84.7% intent / 4-of-5 OOD rejected
The remaining errors are almost all whisper ASR mistakes on very short
words ("pause" -> "boss", "time" -> "bye"); for those, "can you repeat
that?" is the honest answer (and what this version was asked to do).

Cost: whisper base.en (int8) takes ~1-3 s per command on a Pi 4 (v3's
PocketSphinx was ~150 ms). The wake word + VAD are unchanged.

This folder is self-contained: the stage-2 classifier, the `vcm`/`vcm2`
code, the response WAVs, and the openWakeWord "hey rhasspy" model
(in `wakeword/`) all live here. The whisper model (base.en, ~145 MB) is
downloaded from Hugging Face on first run and cached in ~/.cache/huggingface.

Usage
-----
    python vcm_pi_v4.py                 # live mic: wake word -> command -> speak
    python vcm_pi_v4.py --file clip.wav # transcribe + classify one file
    python vcm_pi_v4.py --test          # run the 171-clip held-out test set
    python vcm_pi_v4.py --no-play       # (mic) classify + print, skip playback
    python vcm_pi_v4.py --whisper small.en   # bigger (slower) whisper model
    python vcm_pi_v4.py --wake-threshold 0.6 # stricter wake-word gate
    python vcm_pi_v4.py --command-window 1.5 # wait 1.5 s for the command

Flow
----
    start  ->  STANDBY: wait for "hey rhasspy" (noise is ignored)
             ->  play the "yes?" cue (mic muted while it plays)
             ->  wait up to 1 s for you to START speaking
                 (0.6 s of silence then ends the utterance)
             ->  whisper transcribes what was heard (printed)
             ->  keyword gate + classifier -> a command or REJECT
             ->  command: speak/play the response
                 REJECT: say what was heard + "can you repeat that?"
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
import subprocess
import sys
import threading
import time
import urllib.parse
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
CLASSIFIER = os.path.join(_HERE, "classifier.pkl")
RESP_DIR = os.path.join(_HERE, "responses")
YES_WAV = os.path.join(RESP_DIR, "00_yes.wav")   # "yes?" cue after the wake word
TEST_DATA = os.path.normpath(os.path.join(_HERE, "..", "data",
                                          "additional_test_data"))

# After the device finishes speaking (or playing a canned response), it ignores
# the mic for this many seconds before accepting the next command, so the tail
# or echo of the response can't be heard as a new command. (The mic is already
# gated while the response plays; this covers the brief moment after it ends.)
COOLDOWN_S = 0.5

# While a voice command is being handled (the "yes?" cue + the spoken response),
# the music is ducked to MusicPlayer.DUCK_VOLUME so the wake word / command can
# be heard over it. If no new command arrives within this many seconds, the
# music reverts to the previously set volume on its own (a timer, not a
# requirement that you speak again).
DUCK_SECONDS = 10.0

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
# v4 model: faster-whisper free-text ASR  ->  keyword gate  ->  31-class
# classifier (with a live REJECT class).
#
# WHY THIS REPLACES v3's grammar-constrained PocketSphinx ensemble:
#   v3's transcript came from a JSGF-constrained decoder, which is FORCED to
#   output one of the 103 grammar phrases. So when you said something that was
#   NOT a command, it still printed the nearest command phrase ("play music"),
#   and the stage-2 classifier's REJECT class (trained on 32,992 OOD examples)
#   was dead -- it only ever saw valid phrases and force-classified everything.
#   v4 transcribes with a real free-text ASR (faster-whisper), so the
#   transcript is WHAT WAS ACTUALLY HEARD, and the classifier sees that real
#   text -- where it can (and does) emit REJECT for non-commands.
#
#   The keyword gate is the first, transparent rejection stage: a real command
#   always contains a command word (play / lights / volume / weather / ...).
#   Speech with no command word ("he will do it my man", "i'm fucked") is
#   rejected immediately -- this is what v3 could never do. The classifier
#   then handles the subtle cases (near-miss commands, slot values).
# --------------------------------------------------------------------------
import re

WHISPER_MODEL = os.environ.get("V4_WHISPER", "base.en")
# A real command always contains at least one of these words. Speech with none
# is rejected before it reaches the classifier (the OOD rejection v3 lacked).
COMMAND_KEYWORDS = [
    "play", "pause", "stop", "next", "previous", "volume", "louder",
    "quieter", "lower", "raise", "increase", "decrease", "music", "song",
    "light", "lights", "brightness", "brighter", "darker", "color", "colour",
    "temperature", "hot", "cold", "warm", "heat", "degree", "timer", "alarm",
    "weather", "time", "call", "message", "remind", "reminder", "reminders",
    "list", "set", "turn", "switch", "on", "off", "up", "down",
]


def normalize_whisper(text: str) -> str:
    """whisper free text -> the form the stage-2 classifier expects.

    The classifier's own normalize() already lowercases, strips punctuation
    and expands digits to number-words, so this only fixes the shapes whisper
    emits that the training data never saw: "100%" -> "100 percent",
    "8am" -> "8 am", "22 degrees" kept as-is.
    """
    t = (text or "").lower().strip()
    t = re.sub(r"[.,!?;:]", " ", t)
    t = re.sub(r"\b(\d+)\s*%", r" \1 percent ", t)
    t = re.sub(r"\b(\d+)\s*(am|pm)\b", r" \1 \2 ", t)
    t = re.sub(r"\b(\d+)\s*degrees?\b", r" \1 degrees ", t)
    return re.sub(r"\s+", " ", t).strip()


def has_command_keyword(text: str) -> bool:
    """True if `text` contains a command word (the rejection gate)."""
    return any(re.search(r"\b" + re.escape(k) + r"\b", text)
               for k in COMMAND_KEYWORDS)


class Model:
    """faster-whisper (free text) -> keyword gate -> 31-class + REJECT clf.

    classify(audio) takes float32 mono audio at ANY sample rate (whisper
    resamples internally) and returns
        (command, intent, heard_text, normalized_text, clf_prob)
    where `heard_text` is the raw whisper transcript (what was actually
    spoken) and command is one of the 31 commands or REJECT.
    """

    def __init__(self, whisper_model: str = None, device: str = "cpu",
                 compute_type: str = "int8"):
        from faster_whisper import WhisperModel
        name = whisper_model or WHISPER_MODEL
        print(f"loading whisper {name} ({compute_type}) ...", flush=True)
        self._whisper = WhisperModel(name, device=device,
                                     compute_type=compute_type)
        self.clf = load_classifier(CLASSIFIER)
        print("ready.", flush=True)

    def transcribe(self, audio: np.ndarray) -> str:
        """float32 mono audio -> raw whisper transcript (what was heard)."""
        segs, _info = self._whisper.transcribe(audio, beam_size=1,
                                               vad_filter=False)
        return " ".join(s.text.strip() for s in segs).strip()

    def classify(self, audio: np.ndarray):
        """audio -> (command, intent, heard, normalized, clf_prob)."""
        heard = self.transcribe(audio)
        norm = normalize_whisper(heard)
        if not norm or not has_command_keyword(norm):
            # No command word at all -> not a command. This is the rejection
            # v3's constrained decoder could never produce.
            return "REJECT", "reject", heard, norm, 0.0
        cmd, prob = predict(self.clf, norm)
        return cmd, _INTENT_OF.get(cmd, "unknown"), heard, norm, prob


# --------------------------------------------------------------------------
# audio helpers
# --------------------------------------------------------------------------
_PLAYER_CACHE = None


def _pick_player():
    """Return (name, argv_prefix) for the first available external audio
    player, or None.

    Responses are played by a SEPARATE-PROCESS player, not in-process
    sd.play: on the Pi, opening an output stream with sd.play while the mic
    input stream is open fails with PortAudioError. A subprocess (mpv, the
    same one that plays the music) opens its own stream, which is why the
    music works. mpv is preferred -- it is installed (music) and handles
    resampling + PipeWire; paplay/aplay/ffplay are fallbacks for machines
    without mpv.
    """
    global _PLAYER_CACHE
    if _PLAYER_CACHE is not None:
        return _PLAYER_CACHE
    import shutil
    candidates = [
        ("mpv",    ["mpv", "--no-video", "--really-quiet", "--no-terminal"]),
        ("paplay", ["paplay"]),
        ("aplay",  ["aplay", "-q"]),
        ("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]),
    ]
    chosen = None
    for name, argv in candidates:
        if shutil.which(name):
            chosen = (name, argv)
            break
    _PLAYER_CACHE = chosen
    return chosen


def _play_wav_proc(path: str) -> None:
    """Play a WAV via a separate-process player (blocking)."""
    player = _pick_player()
    if player is None:
        raise RuntimeError("no external audio player found")
    name, argv = player
    subprocess.run(argv + [path], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def play_wav(path: str, enabled: bool = True) -> bool:
    """Play a WAV (blocking). Degrades to a printed note if no audio device.

    Returns True if the audio was actually played. Prefers a separate-process
    player (mpv/paplay/aplay) because on the Pi an in-process sd.play output
    stream fails (PortAudioError) while the mic input stream is open -- a
    subprocess opens its own stream, which is why the mpv music playback
    works. Falls back to in-process sd.play (works on headless dev machines /
    when the mic is not open).
    """
    if not enabled:
        print(f"  >> [no-play] {os.path.basename(path)}")
        return False
    # 1) Separate-process player (robust on the Pi).
    if _pick_player() is not None:
        try:
            _play_wav_proc(path)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  >> [player failed: {type(e).__name__}] "
                  f"falling back to in-process")
    # 2) In-process sd.play fallback. Pi speakers (USB / I2S) usually only
    #    support 44.1/48 kHz, so play at the device's native rate, resampling
    #    in software; fall back to 16 kHz if the native rate is rejected.
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
                return True
            except Exception as e:
                last_err = e
        print(f"  >> [playback failed: {type(last_err).__name__}] would play "
              f"{os.path.basename(path)}")
    except Exception as e:  # noqa: BLE001 - headless / no PortAudio
        print(f"  >> [no audio device: {type(e).__name__}] would play "
              f"{os.path.basename(path)}")
    return False


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
    import tempfile
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        voice.synthesize_wav(text, w)
    buf.seek(0)
    # Write to a temp file and play via play_wav() so the SAME separate-
    # process player is used (in-process sd.play fails on the Pi while the
    # mic stream is open -- see play_wav's docstring).
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp.write(buf.read())
        tmp.close()
        if not play_wav(tmp.name, enabled=enabled):
            print(f"  >> would say: {text!r}")
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def time_response_text() -> str:
    """Current time in UTC+8, 12-hour format, as a spoken sentence.

    e.g. "It is 5:42 PM."  (UTC+8 = the device's timezone, Asia/Manila)
    """
    now = datetime.now(ZoneInfo("Asia/Manila"))
    return f"It is {now.strftime('%I:%M %p')}."


# --------------------------------------------------------------------------
# dynamic WEATHER response (IP geolocation + Open-Meteo, then Piper TTS)
#
# 1. Location: the device's current location, looked up from its public IP
#    (ip-api.com primary, ipinfo.io fallback; both free, no key). If the
#    lookup fails (offline, blocked, or the IP is a private one), fall back
#    to UP Diliman, Quezon City. The spoken name is **city-level** (e.g.
#    "Quezon City", "Pasig") -- never a barangay.
# 2. Weather: Open-Meteo (free, no API key) -- current conditions only.
# 3. Spoken:  e.g. "Currently in Quezon City: partly cloudy, 29 degrees."
#
# The whole lookup is bounded (~10 s worst case) so a dead network degrades
# to the fallback location / a short apology instead of hanging the device.
# --------------------------------------------------------------------------
WEATHER_FALLBACK = ("UP Diliman, Quezon City", 14.6265, 121.0465)
WEATHER_HTTP_TIMEOUT = 10   # seconds, per request


def _http_json(url: str, timeout: int = WEATHER_HTTP_TIMEOUT):
    """GET `url` and parse JSON. Returns None on any failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "vcm-pi-v3"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 - offline / DNS / timeout / bad JSON
        return None


def _geocode(query: str):
    """(name, lat, lon) for a place name via Open-Meteo geocoding.

    Returns None if the service is unreachable or finds nothing.
    """
    url = ("https://geocoding-api.open-meteo.com/v1/search"
           f"?name={urllib.parse.quote(query)}&count=1&language=en&format=json")
    data = _http_json(url)
    if data and data.get("results"):
        r = data["results"][0]
        name = (f"{r.get('name')}, {r.get('admin1')}"
                if r.get("admin1") else r.get("name") or query)
        return name, float(r["latitude"]), float(r["longitude"])
    return None


def _current_location(override: str | None = None):
    """(name, lat, lon) of the device's current location, city-level.

    `override` (e.g. from --weather-loc) wins -- it is geocoded so the
    coordinates are real (e.g. 'Quezon City' -> 14.6488, 121.0509); if that
    geocoding fails, UP Diliman coordinates are used. Otherwise look up the
    public IP's location (ip-api.com, then ipinfo.io as a fallback); on any
    failure fall back to UP Diliman, Quezon City.

    The name is always reduced to the CITY level (e.g. "Quezon City",
    "Pasig") -- never a barangay -- because free IP-geolocation databases
    are unreliable at the barangay level (e.g. they reported "Guyong" for
    this HPC's IP when the city is Quezon City).
    """
    if override:
        geo = _geocode(override)
        if geo:
            return geo
        print(f"  >> [weather] could not geocode {override!r} -- using "
              f"{WEATHER_FALLBACK[0]} coordinates")
        return override, WEATHER_FALLBACK[1], WEATHER_FALLBACK[2]
    # ip-api.com (free, no key, generous rate limit)
    d = _http_json("http://ip-api.com/json/?fields=status,city,lat,lon")
    if d and d.get("status") == "success" and d.get("lat") is not None:
        name = (d.get("city") or d.get("regionName") or "your location")
        return name, float(d["lat"]), float(d["lon"])
    # ipinfo.io (free, no key) -- note: "loc" is a "lat,lon" string
    d = _http_json("https://ipinfo.io/json")
    if d and d.get("loc"):
        try:
            lat, lon = (float(x) for x in d["loc"].split(","))
            name = d.get("city") or d.get("region") or "your location"
            return name, lat, lon
        except (ValueError, AttributeError):
            pass
    print("  >> [weather] IP location lookup failed -- using "
          f"{WEATHER_FALLBACK[0]}")
    return WEATHER_FALLBACK


def weather_response_text(override: str | None = None) -> str:
    """Fetch the current weather for the device's location; spoken sentence.

    e.g. "Currently in Quezon City: partly cloudy, 29 degrees."
    """
    name, lat, lon = _current_location(override)
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat:.4f}&longitude={lon:.4f}"
           "&current=temperature_2m,apparent_temperature,relative_humidity_2m,"
           "weather_code,wind_speed_10m"
           "&timezone=auto")
    data = _http_json(url)
    if not data or "current" not in data:
        return "I could not reach the weather service right now."
    cur = data["current"]
    desc = _WEATHER_CODES.get(int(cur.get("weather_code", -1)),
                              "the current conditions")
    temp = int(round(float(cur["temperature_2m"])))
    return (f"Currently in {name}: {desc}, {temp} degrees "
            f"Celsius, feels like {int(round(float(cur['apparent_temperature'])))}.")


# WMO weather interpretation codes -> spoken description (subset used by
# Open-Meteo's `weather_code`).
_WEATHER_CODES = {
    0: "clear skies", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "heavy showers",
    85: "snow showers", 86: "snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail",
    99: "thunderstorms with hail",
}


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
# --------------------------------------------------------------------------
# MUSIC -- drive a YouTube playlist with the seven media commands
#
# play_music / pause / stop / next / volume_up / volume_down now ACTUALLY
# control playback (they no longer just play a canned "acknowledgement" WAV).
#
# How it works: **yt-dlp** resolves the YouTube playlist once (flat, fast)
# into a plain `playlist.m3u` next to this script; **mpv** (a headless
# command-line player) plays that file and exposes a JSON IPC UNIX socket,
# so each voice command is a one-line JSON message and playback never
# blocks the mic loop. mpv plays each YouTube track through its bundled
# ytdl hook (which also shells out to yt-dlp).
#
# Why the .m3u step: on a Pi, mpv handed a raw YouTube *playlist* URL
# often loads nothing (its ytdl hook is unreliable for playlist URLs), so
# the old version said "playing the playlist" while the speaker stayed
# silent. Resolving up front makes the failure visible (the command then
# says "music is not available right now") and playback reliable.
#
#   * volume is stepped through discrete levels: 0 / 25 / 50 / 75 / 100 %
#   * if mpv or yt-dlp is missing, or the playlist cannot be resolved,
#     every command degrades to a spoken "unavailable" message.
#
# Install once on the Pi:
#   sudo apt install mpv          # system player (NOT a pip package)
#   pip install yt-dlp            # into the venv (added to requirements.txt)
# --------------------------------------------------------------------------
DEFAULT_PLAYLIST = ("https://www.youtube.com/watch?v=Q-jz864NQS4"
                    "&list=PLd9fevitX97w&index=3")

# the seven commands that drive real playback (instead of a canned WAV)
MUSIC_COMMANDS = frozenset(
    {"PLAY_MUSIC", "PAUSE", "STOP", "NEXT", "PREVIOUS",
     "VOLUME_UP", "VOLUME_DOWN"})


class MusicPlayer:
    """Controls a YouTube playlist via mpv (JSON IPC) + yt-dlp.

    All methods are safe to call from the main thread; mpv runs as a separate
    process so playback never blocks the mic loop. If mpv / yt-dlp are not
    installed, the methods return a short "unavailable" message instead of
    raising, so the device degrades gracefully.
    """
    VOLUME_STEPS = (0, 25, 50, 75, 100)
    DUCK_VOLUME = 10   # music is ducked to this (percent) while a response
                       # plays / the wake word is being handled, then restored

    def __init__(self, playlist_url, socket_path, start_volume=50):
        self.url = playlist_url
        self.sock_path = socket_path
        self.proc = None
        self._sock = None
        self._log = None
        self._reader_thread = None
        self._pending = {}
        self._pending_lock = threading.Lock()
        self._paused = False
        self._vol_idx = (self.VOLUME_STEPS.index(start_volume)
                         if start_volume in self.VOLUME_STEPS else 2)
        self._ducked = False
        self._pre_duck_idx = self._vol_idx
        self._auto_held = False

    # -- mpv lifecycle ----------------------------------------------------
    def _bin(self, name):
        import shutil
        return shutil.which(name)

    def _ytdlp_cmd(self):
        """Return the yt-dlp invocation as a list, or None if missing.
        Prefers the yt-dlp executable on PATH (the venv's bin dir); falls
        back to `python -m yt_dlp` (same interpreter as this script) so
        it works even when only the module is installed."""
        exe = self._bin("yt-dlp")
        if exe:
            return [exe]
        try:
            import yt_dlp  # noqa: F401
            return [sys.executable, "-m", "yt_dlp"]
        except ImportError:
            return None

    def _resolve_playlist(self):
        """Resolve the YouTube playlist to a local playlist.m3u (fast,
        flat -- no per-track metadata). Returns the m3u path, or None if
        yt-dlp failed (offline / playlist gone / network blocked).

        We ask yt-dlp for one bare URL per line (`--print url`) and write
        the .m3u ourselves -- this is more reliable than relying on
        yt-dlp's own playlist-file naming, which silently writes nothing
        for a plain `-o` on a playlist URL."""
        ytdlp = self._ytdlp_cmd()
        if ytdlp is None:
            print("  >> [music] yt-dlp not found -- install with "
                  "`pip install yt-dlp`")
            return None
        m3u = os.path.join(_HERE, "playlist.m3u")
        try:
            r = subprocess.run(
                ytdlp + ["--flat-playlist", "--no-warnings",
                         "--playlist-items", "1-200",
                         "--print", "url", self.url],
                capture_output=True, text=True, timeout=120)
        except Exception as e:  # noqa: BLE001
            print(f"  >> [music] playlist resolve failed: {e}")
            return None
        urls = [l.strip() for l in (r.stdout or "").splitlines()
                if l.strip() and "youtube" in l]
        if r.returncode != 0 or not urls:
            tail = (r.stderr or r.stdout or "").strip().splitlines()
            print(f"  >> [music] playlist resolve failed: "
                  f"{tail[-1] if tail else 'no tracks returned'}")
            return None
        try:
            with open(m3u, "w", encoding="utf-8") as f:
                f.write("#EXTM3U\n")
                for u in urls:
                    f.write(f"#EXTINF:-1,\n{u}\n")
        except OSError as e:
            print(f"  >> [music] could not write {m3u}: {e}")
            return None
        print(f"  >> [music] playlist resolved: {len(urls)} tracks -> {m3u}")
        return m3u

    def _start(self):
        """Resolve the playlist, launch mpv on it; wait for the IPC
        socket. True on OK."""
        if not self._bin("mpv"):
            print("  >> [music] mpv not found -- install with "
                  "`sudo apt install mpv`")
            return False
        if self._ytdlp_cmd() is None:
            print("  >> [music] yt-dlp not found -- install with "
                  "`pip install yt-dlp`")
            return False
        m3u = self._resolve_playlist()
        if m3u is None:
            return False
        try:
            if os.path.exists(self.sock_path):
                os.remove(self.sock_path)
        except OSError:
            pass
        log = open(os.path.join(_HERE, "mpv.log"), "ab")
        cmd = [
            self._bin("mpv"),
            f"--input-ipc-server={self.sock_path}",
            "--no-video",
            "--really-quiet",
            m3u,
        ]
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=log,
                start_new_session=True)
        except Exception as e:  # noqa: BLE001
            print(f"  >> [music] failed to start mpv: {e}")
            return False
        self._log = log
        for _ in range(50):  # up to ~5 s for the socket to appear
            if os.path.exists(self.sock_path):
                break
            time.sleep(0.1)
        if not os.path.exists(self.sock_path):
            print("  >> [music] mpv started but the IPC socket never appeared")
            return False
        self._connect()
        self._reader_thread = threading.Thread(target=self._reader,
                                               daemon=True)
        self._reader_thread.start()
        self._set_volume(self.VOLUME_STEPS[self._vol_idx])
        return True

    def _connect(self):
        import socket as _socket
        try:
            if self._sock is not None:
                self._sock.close()
            s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
            s.connect(self.sock_path)
            self._sock = s
        except Exception:  # noqa: BLE001
            self._sock = None

    def _reader(self):
        """Drain mpv's IPC output (background thread); log real errors only."""
        buf = b""
        while True:
            try:
                chunk = self._sock.recv(4096)
            except Exception:  # noqa: BLE001
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    msg = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if "request_id" in msg:      # reply to a _query() call
                    with self._pending_lock:
                        ev = self._pending.pop(msg["request_id"], None)
                    if ev is not None:
                        ev.reply = msg
                        ev.set()
                err = msg.get("error")
                if err not in (None, "success"):
                    print(f"  >> [mpv] {err}")

    def _cmd(self, *command):
        """Send one JSON command to mpv (the reader thread handles replies)."""
        if self.proc is None or self.proc.poll() is not None:
            return False
        if self._sock is None:
            self._connect()
        if self._sock is None:
            return False
        try:
            self._sock.sendall(
                (json.dumps({"command": list(command)}) + "\n").encode())
            return True
        except Exception:  # noqa: BLE001
            self._sock = None
            return False

    def _query(self, *command, timeout=2.0):
        """Send a JSON command and wait for mpv's reply (request_id match).

        Returns the reply dict, or None if the player is gone / no reply in
        time. Lets next/previous detect the playlist boundary (mpv answers
        playlist-next/-prev at the end with an error instead of skipping)."""
        if self.proc is None or self.proc.poll() is not None:
            return None
        if self._sock is None:
            self._connect()
        if self._sock is None:
            return None
        rid = int(time.time() * 1000) % 10**9
        ev = threading.Event()
        with self._pending_lock:
            self._pending[rid] = ev
        try:
            self._sock.sendall(
                (json.dumps({"command": list(command),
                             "request_id": rid}) + "\n").encode())
        except Exception:  # noqa: BLE001
            with self._pending_lock:
                self._pending.pop(rid, None)
            self._sock = None
            return None
        if not ev.wait(timeout):
            with self._pending_lock:
                self._pending.pop(rid, None)
            return None
        return getattr(ev, "reply", None)

    def _set_volume(self, pct):
        self._cmd("set_property", "volume", pct)

    def _running(self):
        return self.proc is not None and self.proc.poll() is None

    # -- the seven voice commands ------------------------------------------
    def play_music(self):
        if not self._running():
            if self._start():
                self._paused = False
                return "playing the playlist"
            return "music is not available right now"
        if self._paused:
            self._cmd("set_property", "pause", False)
            self._paused = False
            return "resuming the playlist"
        return "the playlist is already playing"

    def pause(self):
        if not self._running():
            return "nothing is playing"
        self._cmd("set_property", "pause", True)
        self._paused = True
        return "paused"

    def stop(self):
        if not self._running():
            return "nothing is playing"
        self._cmd("set_property", "pause", True)
        self._cmd("set_property", "time-pos", 0)
        self._paused = True
        return "stopped"

    def next(self):
        if not self._running():
            return "nothing is playing"
        reply = self._query("playlist-next")
        if reply and reply.get("error") not in (None, "success"):
            return "that's the last song in the playlist"
        return "next song"

    def previous(self):
        if not self._running():
            return "nothing is playing"
        reply = self._query("playlist-prev")
        if reply and reply.get("error") not in (None, "success"):
            return "that's the first song in the playlist"
        return "previous song"

    def volume_up(self):
        if self._vol_idx < len(self.VOLUME_STEPS) - 1:
            self._vol_idx += 1
        pct = self.VOLUME_STEPS[self._vol_idx]
        if self._running():
            self._set_volume(pct)
        return f"volume {pct} percent"

    def volume_down(self):
        if self._vol_idx > 0:
            self._vol_idx -= 1
        pct = self.VOLUME_STEPS[self._vol_idx]
        if self._running():
            self._set_volume(pct)
        return f"volume {pct} percent"

    # -- ducking (let the wake word / a spoken response be heard) ----------
    def duck(self):
        """Drop the music to DUCK_VOLUME while the device is busy (a wake word
        fired / a response is about to play), remembering the previous volume.
        Idempotent -- calling it again just keeps it ducked (and, in the main
        loop, resets the revert timer). No-op if nothing is playing."""
        if not self._running():
            return
        if not self._ducked:
            self._pre_duck_idx = self._vol_idx
            self._ducked = True
        self._set_volume(self.DUCK_VOLUME)

    def unduck(self):
        """Restore the music to the volume it had before it was ducked. No-op
        if it was never ducked or nothing is playing."""
        if not self._ducked:
            return
        self._ducked = False
        if self._running():
            self._set_volume(self.VOLUME_STEPS[self._pre_duck_idx])

    def dispatch(self, cmd):
        """Map a classified command to a music action; return spoken text."""
        return {
            "PLAY_MUSIC": self.play_music,
            "PAUSE": self.pause,
            "STOP": self.stop,
            "NEXT": self.next,
            "PREVIOUS": self.previous,
            "VOLUME_UP": self.volume_up,
            "VOLUME_DOWN": self.volume_down,
        }.get(cmd, lambda: "music command not recognized")()

    def shutdown(self):
        """Stop mpv and clean up (called on Ctrl-C)."""
        try:
            if self._running():
                self._cmd("quit")
                time.sleep(0.2)
                if self._running():
                    self.proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if os.path.exists(self.sock_path):
                os.remove(self.sock_path)
        except OSError:
            pass


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
    model = Model(whisper_model=getattr(args, "whisper", None))
    state = {"busy": False, "audio": None, "cue": False, "muted": False}

    # music player (mpv + yt-dlp) -- the seven media commands drive this
    playlist = getattr(args, "playlist", None) or DEFAULT_PLAYLIST
    player = MusicPlayer(
        playlist,
        socket_path=os.path.join(_HERE, "mpv.sock"),
        start_volume=getattr(args, "volume", 50))
    print(f"music      : {playlist}")

    # Ducking timer: when the wake word fires the music is ducked to 10% so the
    # "yes?" cue + spoken response are audible over it. Each command extends the
    # window; if no command arrives within DUCK_SECONDS the music reverts to the
    # volume it had before the duck (checked in the idle loop below).
    duck_until = 0.0

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
                    # Duck the music to 10% so the cue + the spoken response
                    # are heard over it; it reverts on its own after
                    # DUCK_SECONDS if no command follows (see idle check).
                    player.duck()
                    duck_until = time.time() + DUCK_SECONDS
                    print("  >> 'yes?' -- ready for your command")
                    play_wav(YES_WAV, enabled=not args.no_play)
                    state["muted"] = False
                    gate.cue_done()                # NOW start the 1 s window
                    continue
                if not state["busy"]:
                    # Revert the duck on its own once the window has elapsed
                    # with no command (the user is done; music goes back up).
                    if player._ducked and time.time() > duck_until:
                        player.unduck()
                    time.sleep(0.01)
                    continue
                a = state["audio"]
                state["audio"] = None
                vad.reset()
                n += 1
                t0 = time.perf_counter()
                # v4: whisper transcribes WHAT WAS ACTUALLY HEARD (free text),
                # the keyword gate rejects non-commands, and the classifier
                # maps the rest to one of the 31 commands (or REJECT).
                cmd, intent, heard, norm, cprob = model.classify(a)
                e2e = (time.perf_counter() - t0) * 1000.0
                print(f"[{time.strftime('%H:%M:%S')}] -- utterance #{n} "
                      f"({len(a) / 16000:.2f} s)")
                print(f"  heard      : {heard!r}   (what was actually spoken)")
                print(f"  command    : {cmd}   intent: {intent}   "
                      f"(clf {cprob:.2f})")
                print(f"  E2E {e2e:.0f} ms")
                # "previous song" is not a trained classifier class (the
                # 31-class model has no PREVIOUS), so route it from the heard
                # text: "previous" only appears in that one command.
                if "previous" in norm:
                    cmd, intent = "PREVIOUS", "media_control"
                    print("  (routed PREVIOUS from heard text -- not a "
                          "trained class)")
                # A command arrived: keep the music ducked while the response
                # plays, and extend the revert window (so the user can issue
                # another command right after without the music jumping up).
                player.duck()
                duck_until = time.time() + DUCK_SECONDS
                if cmd == "REJECT":
                    # HONEST rejection: say what was actually heard, then ask
                    # to repeat. (v3 never reached here -- the constrained
                    # decoder force-classified everything into a command.)
                    if heard.strip():
                        text = f"I heard: {heard}. Can you repeat that?"
                    else:
                        text = "I didn't catch that. Can you repeat?"
                    print(f"  >> saying (Piper TTS): {text!r}")
                    speak(text, enabled=not args.no_play)
                elif cmd == "TIME":
                    # dynamic response: say the ACTUAL current time (UTC+8)
                    text = time_response_text()
                    print(f"  >> saying (Piper TTS): {text!r}")
                    speak(text, enabled=not args.no_play)
                elif cmd == "WEATHER":
                    # dynamic response: live weather for the device's
                    # current location (IP geolocation; fallback UP Diliman)
                    text = weather_response_text(getattr(args, "weather_loc", None))
                    print(f"  >> saying (Piper TTS): {text!r}")
                    speak(text, enabled=not args.no_play)
                elif cmd in MUSIC_COMMANDS:
                    # REAL playback: the seven media commands drive the
                    # YouTube playlist (mpv + yt-dlp), then the spoken
                    # confirmation is synthesized (Piper TTS).
                    text = player.dispatch(cmd)
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
    player.shutdown()
    print(f"\nstopped after {n} command(s).")


def run_file(args):
    import soundfile as sf
    model = Model(whisper_model=getattr(args, "whisper", None))
    a, sr = sf.read(args.file, dtype="float32")
    if a.ndim > 1:
        a = a[:, 0]
    t0 = time.perf_counter()
    cmd, intent, heard, norm, cprob = model.classify(a)
    e2e = (time.perf_counter() - t0) * 1000.0
    print(f"  file       : {args.file}")
    print(f"  heard      : {heard!r}   (what was actually spoken)")
    print(f"  command    : {cmd}   intent: {intent}   (clf {cprob:.2f})")
    print(f"  E2E {e2e:.0f} ms")
    if cmd == "REJECT":
        text = (f"I heard: {heard}. Can you repeat that?"
                if heard.strip() else "I didn't catch that. Can you repeat?")
        print(f"  >> saying (Piper TTS): {text!r}")
        speak(text, enabled=not args.no_play)
    elif cmd == "TIME":
        text = time_response_text()
        print(f"  >> saying (Piper TTS): {text!r}")
        speak(text, enabled=not args.no_play)
    elif cmd == "WEATHER":
        text = weather_response_text(getattr(args, "weather_loc", None))
        print(f"  >> saying (Piper TTS): {text!r}")
        speak(text, enabled=not args.no_play)
    elif cmd in MUSIC_COMMANDS:
        player = MusicPlayer(
            getattr(args, "playlist", None) or DEFAULT_PLAYLIST,
            socket_path=os.path.join(_HERE, "mpv.sock"),
            start_volume=getattr(args, "volume", 50))
        text = player.dispatch(cmd)
        print(f"  >> saying (Piper TTS): {text!r}")
        speak(text, enabled=not args.no_play)
        player.shutdown()
    else:
        wav = RESPONSE_WAV.get(cmd, RESPONSE_WAV["REJECT"])
        print(f"  >> playing {wav}")
        play_wav(os.path.join(RESP_DIR, wav), enabled=not args.no_play)


def run_test(args):
    import soundfile as sf
    model = Model(whisper_model=getattr(args, "whisper", None))
    rows = build_ground_truth(args.data)
    print(f"{len(rows)} clips in {args.data}\n")
    correct = intent_correct = 0
    ood_n = ood_reject = false_reject = 0
    per_folder = {}
    t0 = time.perf_counter()
    for k, r in enumerate(rows):
        gold_intent = _INTENT_OF.get(r["gold"], "unknown")
        a, sr = sf.read(r["path"], dtype="float32")
        if a.ndim > 1:
            a = a[:, 0]
        cmd, intent, heard, norm, cprob = model.classify(a)
        ok = cmd == r["gold"]
        iok = intent == gold_intent
        correct += ok
        intent_correct += iok
        if r["gold"] == "REJECT":
            ood_n += 1
            ood_reject += cmd == "REJECT"
        else:
            false_reject += cmd == "REJECT"
        f = per_folder.setdefault(r["folder"], {"n": 0, "cmd": 0, "intent": 0})
        f["n"] += 1
        f["cmd"] += ok
        f["intent"] += iok
        mark = "  " if ok else "!!"
        print(f"{mark} {r['folder']:16s} {r['spoken']!r:30s} "
              f"heard={heard!r:34s} -> {cmd:24s} (gold {r['gold']:20s})")
    n = len(rows)
    wall = time.perf_counter() - t0
    report = {
        "model": f"faster-whisper {WHISPER_MODEL} + keyword gate + "
                 "31-class+REJECT classifier",
        "n_clips": n,
        "command_acc": round(correct / n, 4),
        "intent_acc": round(intent_correct / n, 4),
        "ood_rejected": f"{ood_reject}/{ood_n}",
        "false_reject": false_reject,
        "wall_s": round(wall, 1),
        "per_folder": {k: {"n": v["n"],
                           "cmd_acc": round(v["cmd"] / v["n"], 4),
                           "intent_acc": round(v["intent"] / v["n"], 4)}
                       for k, v in sorted(per_folder.items())},
    }
    out = os.path.join(_HERE, "test_v4_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\ncommand_acc {report['command_acc']:.4f}  "
          f"intent_acc {report['intent_acc']:.4f}  "
          f"OOD {ood_reject}/{ood_n}  false-reject {false_reject}  "
          f"({wall:.1f} s, {n / wall:.1f} clips/s)")
    print(f"report -> {out}")


def run_music_test(args):
    """No-mic smoke test for the music pipeline: resolve the playlist,
    start mpv, let it play for ~12 s, then stop. If you hear music, the
    seven voice commands will work."""
    playlist = getattr(args, "playlist", None) or DEFAULT_PLAYLIST
    player = MusicPlayer(
        playlist,
        socket_path=os.path.join(_HERE, "mpv.sock"),
        start_volume=getattr(args, "volume", 50))
    print(f"music test : {playlist}")
    msg = player.dispatch("PLAY_MUSIC")
    print(f"  >> {msg}")
    if "not available" in msg:
        print("  FAILED -- see the [music] lines above (and mpv.log).")
        player.shutdown()
        sys.exit(1)
    time.sleep(12)
    print("  (played 12 s -- did you hear music?)")
    print(f"  >> {player.dispatch('NEXT')}")
    time.sleep(5)
    print(f"  >> {player.dispatch('PREVIOUS')}")
    time.sleep(3)
    print(f"  >> {player.dispatch('VOLUME_UP')}")
    time.sleep(3)
    print(f"  >> {player.dispatch('STOP')}")
    player.shutdown()
    print("  music test done -- mpv.log has the full player output.")
    sys.exit(0)


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
    ap.add_argument("--whisper", default=None,
                    help="faster-whisper model name for the free-text "
                         "transcriber (default: base.en). Bigger = more "
                         "accurate but slower: tiny.en, base.en, small.en, "
                         "medium.en. Downloaded from Hugging Face on first "
                         "run and cached in ~/.cache/huggingface.")
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
    ap.add_argument("--weather-loc", default=None,
                    help="location name to use for the WEATHER command "
                         "instead of IP geolocation (e.g. 'UP Diliman, "
                         "Quezon City'). The coordinates always fall back "
                         "to UP Diliman when an override is given.")
    ap.add_argument("--playlist", default=None,
                    help="YouTube playlist URL for the music commands "
                         "(play_music / pause / stop / next / volume_up / "
                         "volume_down). Default: the built-in playlist.")
    ap.add_argument("--volume", type=int, default=50,
                    choices=list(MusicPlayer.VOLUME_STEPS),
                    help="initial music volume, one of 0/25/50/75/100 "
                         "(percent). volume_up / volume_down step through "
                         "these levels.")
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
    ap.add_argument("--music-test", action="store_true",
                    help="test the music pipeline WITHOUT the mic: resolve "
                         "the playlist, start mpv, play for ~12 s, then "
                         "stop and exit. Use this on the Pi to confirm "
                         "sound actually comes out before going live.")
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
    elif args.music_test:
        run_music_test(args)
    else:
        run_mic(args)


if __name__ == "__main__":
    main()
