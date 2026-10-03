"""Command ontology: phrase -> command class (the 32-class v3/v4/v5 ontology)
and class -> intent (the 10-intent set used for intent_acc).

The mapping is rule-based (keyword match) so it works for any grammar phrase,
independent of the exact transcript wording.
"""
from __future__ import annotations

# phrase -> class, evaluated in order; first match wins.
_RULES = [
    ("ALARM_6_00AM", lambda p: "alarm" in p and "six" in p),
    ("ALARM_8_00AM", lambda p: "alarm" in p and "eight" in p),
    ("ALARM_9_00PM", lambda p: "alarm" in p and "nine" in p),
    ("ALARM", lambda p: "alarm" in p or "wake me up" in p),
    ("BRIGHTNESS_100", lambda p: "brightness" in p and "hundred" in p),
    ("BRIGHTNESS_60", lambda p: "brightness" in p and "sixty" in p),
    ("BRIGHTNESS_20", lambda p: "brightness" in p and "twenty" in p),
    ("BRIGHTNESS", lambda p: "brightness" in p),
    ("COLOR_RED", lambda p: ("color" in p and "red" in p) or ("lights to red" in p) or ("set the lights to red" in p)),
    ("COLOR_GREEN", lambda p: ("color" in p and "green" in p) or ("lights to green" in p) or ("set the lights to green" in p)),
    ("COLOR_BLUE", lambda p: ("color" in p and "blue" in p) or ("lights to blue" in p) or ("set the lights to blue" in p)),
    ("COLOR", lambda p: "color" in p),
    ("TEMPERATURE_18", lambda p: "temperature" in p and "eighteen" in p),
    ("TEMPERATURE_22", lambda p: "temperature" in p and "twenty two" in p),
    ("TEMPERATURE_26", lambda p: "temperature" in p and "twenty six" in p),
    ("TEMPERATURE", lambda p: "temperature" in p),
    ("CREATE_REMINDER_DRINK_WATER", lambda p: "drink water" in p),
    ("CREATE_REMINDER_EXERCISE", lambda p: "exercise" in p),
    ("CREATE_REMINDER_STUDY", lambda p: "study" in p),
    ("CREATE_REMINDER", lambda p: ("create" in p and "reminder" in p) or "remind me" in p),
    ("LIST_REMINDERS", lambda p: "list my reminders" in p or "show my reminders" in p or p.strip() == "reminders"),
    ("TIMER_1m", lambda p: ("timer" in p or "countdown" in p) and "minute" in p),
    ("TIMER_30s", lambda p: ("timer" in p or "countdown" in p) and "thirty" in p),
    ("TIMER_10s", lambda p: ("timer" in p or "countdown" in p) and "ten" in p),
    ("TIMER", lambda p: "timer" in p or "countdown" in p),
    ("VOLUME_UP", lambda p: "volume up" in p or "increase the volume" in p or "turn the volume up" in p),
    ("VOLUME_DOWN", lambda p: "volume down" in p or "lower the volume" in p or "turn the volume down" in p),
    ("LIGHT_OFF", lambda p: "lights off" in p or "kill the lights" in p or "lights out" in p
                       or "shut off the lights" in p or "turn off the lights" in p),
    ("LIGHT_ON", lambda p: "lights on" in p or "turn on the lights" in p or "power on the lights" in p),
    ("PLAY_MUSIC", lambda p: "play music" in p or "play some music" in p or "start music" in p),
    ("NEXT", lambda p: "next song" in p or "play next song" in p or "skip song" in p),
    ("PAUSE", lambda p: "pause" in p),
    ("STOP", lambda p: "stop" in p or "end playback" in p),
    ("TIME", lambda p: "time" in p),
    ("WEATHER", lambda p: "weather" in p),
    ("CALL", lambda p: "call" in p),
    ("MESSAGE", lambda p: "message" in p),
]

# class -> intent (10-intent ontology, matches v3 eval)
CLASS_TO_INTENT = {
    "ALARM_6_00AM": "set_alarm", "ALARM_8_00AM": "set_alarm", "ALARM_9_00PM": "set_alarm",
    "ALARM": "set_alarm",
    "BRIGHTNESS_100": "lights_adjust", "BRIGHTNESS_60": "lights_adjust",
    "BRIGHTNESS_20": "lights_adjust", "BRIGHTNESS": "lights_adjust",
    "COLOR_RED": "lights_adjust", "COLOR_GREEN": "lights_adjust",
    "COLOR_BLUE": "lights_adjust", "COLOR": "lights_adjust",
    "TEMPERATURE_18": "set_temperature", "TEMPERATURE_22": "set_temperature",
    "TEMPERATURE_26": "set_temperature", "TEMPERATURE": "set_temperature",
    "CREATE_REMINDER_DRINK_WATER": "reminders_lists",
    "CREATE_REMINDER_EXERCISE": "reminders_lists",
    "CREATE_REMINDER_STUDY": "reminders_lists",
    "CREATE_REMINDER": "reminders_lists",
    "LIST_REMINDERS": "reminders_lists",
    "TIMER_1m": "set_timer", "TIMER_30s": "set_timer", "TIMER_10s": "set_timer",
    "TIMER": "set_timer",
    "VOLUME_UP": "media_control", "VOLUME_DOWN": "media_control",
    "LIGHT_OFF": "lights_switch", "LIGHT_ON": "lights_switch",
    "PLAY_MUSIC": "play_music", "NEXT": "media_control", "PAUSE": "media_control",
    "STOP": "media_control", "TIME": "ask_question", "WEATHER": "ask_question",
    "CALL": "call", "MESSAGE": "call",
    "REJECT": "unknown",
}


def phrase_to_class(phrase: str) -> str:
    p = " ".join(phrase.lower().split())
    for cls, rule in _RULES:
        if rule(p):
            return cls
    return "REJECT"


def class_to_intent(cls: str) -> str:
    return CLASS_TO_INTENT.get(cls, "unknown")


# fine slot class -> coarse 19-command schema label (for eval + device loop).
# The recognizer emits the fine class (it knows the slot value); the device
# only needs the coarse command to act.
_FINE_TO_COARSE = {
    "ALARM_6_00AM": "ALARM", "ALARM_8_00AM": "ALARM", "ALARM_9_00PM": "ALARM",
    "ALARM": "ALARM",
    "BRIGHTNESS_100": "BRIGHTNESS", "BRIGHTNESS_60": "BRIGHTNESS",
    "BRIGHTNESS_20": "BRIGHTNESS", "BRIGHTNESS": "BRIGHTNESS",
    "COLOR_RED": "COLOR", "COLOR_GREEN": "COLOR", "COLOR_BLUE": "COLOR",
    "COLOR": "COLOR",
    "TEMPERATURE_18": "TEMPERATURE", "TEMPERATURE_22": "TEMPERATURE",
    "TEMPERATURE_26": "TEMPERATURE", "TEMPERATURE": "TEMPERATURE",
    "CREATE_REMINDER_DRINK_WATER": "CREATE_REMINDER",
    "CREATE_REMINDER_EXERCISE": "CREATE_REMINDER",
    "CREATE_REMINDER_STUDY": "CREATE_REMINDER",
    "CREATE_REMINDER": "CREATE_REMINDER",
    "LIST_REMINDERS": "LIST_REMINDERS",
    "TIMER_1m": "TIMER", "TIMER_30s": "TIMER", "TIMER_10s": "TIMER",
    "TIMER": "TIMER",
    "VOLUME_UP": "VOLUME_UP", "VOLUME_DOWN": "VOLUME_DOWN",
    "LIGHT_OFF": "LIGHT_OFF", "LIGHT_ON": "LIGHT_ON",
    "PLAY_MUSIC": "PLAY_MUSIC", "NEXT": "NEXT", "PAUSE": "PAUSE",
    "STOP": "STOP", "TIME": "TIME", "WEATHER": "WEATHER",
    "CALL": "CALL", "MESSAGE": "MESSAGE",
    "REJECT": "REJECT",
}


def coarse_class(fine: str) -> str:
    """Map a fine slot class to its coarse 19-command schema label."""
    return _FINE_TO_COARSE.get(fine, "REJECT")
