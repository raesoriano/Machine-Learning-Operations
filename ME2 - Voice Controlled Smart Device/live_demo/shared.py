"""live_demo -- shared layer for the reusable live voice-command demo.

Every recognizer (v3 / v6 / v7 / v8) is wrapped by an adapter that exposes the
SAME interface, so one live loop (vcm_live.py) can drive any of them:

    model.classify_warm(pcm: bytes, reps: int = 1)
        -> (command, intent, transcript, prob)

    * `command`    -- the COARSE 19-command schema label (or "REJECT").
                      v3/v7 (PocketSphinx, 31 fine classes) and v6 (HMM/GMM,
                      93/31 fine classes) are rolled up to this coarse schema;
                      v8 (Conformer+CTC) already emits it.
    * `intent`     -- the coarse intent (see COARSE_INTENT).
    * `transcript` -- the free / unconstrained decode (what was actually heard).
    * `prob`       -- a confidence in [0, 1] (model-specific scale).

The live loop and the response map only ever see the coarse 19 classes, which
is what lets a single `vcm_live.py --model {v3,v6,v7,v8}` drive all four.
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ME2 = os.path.dirname(_HERE)

# ---------------------------------------------------------------------------
# coarse 19-command schema -> intent (the same roll-up the v8 eval uses)
# ---------------------------------------------------------------------------
COARSE_INTENT = {
    "PLAY_MUSIC": "play_music",
    "NEXT": "media_control", "PAUSE": "media_control", "STOP": "media_control",
    "VOLUME_UP": "media_control", "VOLUME_DOWN": "media_control",
    "LIGHT_ON": "lights_switch", "LIGHT_OFF": "lights_switch",
    "BRIGHTNESS": "lights_adjust", "COLOR": "lights_adjust",
    "TEMPERATURE": "set_temperature",
    "TIMER": "set_timer", "ALARM": "set_alarm",
    "CREATE_REMINDER": "reminders_lists", "LIST_REMINDERS": "reminders_lists",
    "TIME": "ask_question", "WEATHER": "ask_question",
    "CALL": "call", "MESSAGE": "call",
    "REJECT": "unknown",
}

# ---------------------------------------------------------------------------
# coarse 19-command schema -> response WAV (from the TTS repo, in responses/).
# Several coarse classes share a phrase (all brightness -> "setting
# brightness", all colors -> "changing color", ...). 00_yes.wav is the "yes?"
# cue (not bound to a command); 19_repeat.wav is the REJECT / "can you repeat
# that?" response.
# ---------------------------------------------------------------------------
RESPONSE_WAV = {
    "PLAY_MUSIC": "01_playing_music.wav",
    "WEATHER": "02_current_weather.wav",
    "TIME": "03_current_time.wav",
    "LIGHT_ON": "04_switching_lights.wav",
    "LIGHT_OFF": "04_switching_lights.wav",
    "BRIGHTNESS": "16_setting_brightness.wav",
    "COLOR": "17_changing_color.wav",
    "TIMER": "13_setting_timer.wav",
    "ALARM": "14_setting_alarm.wav",
    "TEMPERATURE": "15_changing_temperature.wav",
    "PAUSE": "05_pausing.wav",
    "STOP": "06_stopping_playback.wav",
    "NEXT": "07_next_song.wav",
    "VOLUME_UP": "08_volume_up.wav",
    "VOLUME_DOWN": "09_volume_down.wav",
    "CALL": "10_calling.wav",
    "MESSAGE": "11_sending_message.wav",
    "LIST_REMINDERS": "12_reminders_list.wav",
    "CREATE_REMINDER": "18_creating_reminder.wav",
    "REJECT": "19_repeat.wav",
}

# PREVIOUS is a music command that is NOT in the coarse 19 schema (the v3/v7
# 31-class model has no PREVIOUS class, and v8's grammar has no "previous
# song"). It is routed from the decoded transcript in the live loop.
MUSIC_COMMANDS = frozenset(
    {"PLAY_MUSIC", "PAUSE", "STOP", "NEXT", "PREVIOUS",
     "VOLUME_UP", "VOLUME_DOWN"})


def coarse_intent(command: str) -> str:
    return COARSE_INTENT.get(command, "unknown")


def response_wav(command: str) -> str:
    return RESPONSE_WAV.get(command, RESPONSE_WAV["REJECT"])


# ---------------------------------------------------------------------------
# model factory -- the ONLY place that knows how to build each recognizer.
# Each adapter adds its own `pi test vN` folder to sys.path and imports the
# production recognizer, so the live loop runs the EXACT same code the Pi
# runs in each folder (no re-implementation, no retraining).
# ---------------------------------------------------------------------------
def build_model(name: str, **kw):
    """Build the recognizer named `name` (v3 / v6 / v7 / v8).

    Extra kwargs are forwarded to the adapter (e.g. v8's `reject_empty`).
    """
    from adapters import pocketsphinx, v6, v8
    name = name.lower()
    if name in ("v3", "v7"):
        return pocketsphinx.build(name, **kw)
    if name == "v6":
        return v6.build(**kw)
    if name == "v8":
        return v8.build(**kw)
    raise SystemExit(f"unknown --model {name!r} (choose v3, v6, v7, v8)")
