"""Mock smart-home backend — local, zero-dependency (stdlib http.server).

Receives {intent, slots} commands from the VCM service and maintains device
state (lights, thermostat, timers, ...). This is what the RPi service talks
to in the demo; in a real install it is replaced by MQTT to a real hub —
the VCM side never changes (see deploy/rpi_service/backend.py).

Run:
    python -m demo.mock_home.server --port 5000
Then open http://localhost:5000/ for the dashboard, or send commands:
    curl -X POST localhost:5000/command -d '{"intent":"lights_switch","slots":{"state":"on","location":"kitchen"}}'
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DASHBOARD = Path(__file__).parent / "dashboard" / "index.html"


class HomeState:
    """In-memory devices. Thread-safe enough for a demo."""

    def __init__(self):
        self.lights = {}          # location -> {on: bool, percent: int, color: str}
        self.temperature = {"value": 22, "unit": "C"}
        self.media = {"state": "stopped", "volume": 50, "muted": False}
        self.timers = []          # [{label, set_at, duration}]
        self.alarms = []          # [{time, set_at}]
        self.reminders = []       # [{action, set_at}]
        self.history = []         # last 50 (ts, command, response)

    def _light(self, loc="default"):
        return self.lights.setdefault(
            loc, {"on": False, "percent": 100, "color": "white"})

    def handle(self, cmd):
        intent, slots = cmd.get("intent"), cmd.get("slots", {})
        loc = slots.get("location", "default")
        resp = {"ok": False, "message": f"unknown intent: {intent}"}

        if intent == "lights_switch":
            L = self._light(loc)
            L["on"] = slots.get("state") == "on"
            resp = {"ok": True,
                    "message": f"lights {loc}: {'on' if L['on'] else 'off'}"}
        elif intent == "lights_adjust":
            L = self._light(loc)
            L["on"] = True
            if "percent" in slots:
                L["percent"] = int(slots["percent"])
            if "color" in slots:
                L["color"] = slots["color"]
            resp = {"ok": True,
                    "message": f"lights {loc}: {L['percent']}% {L['color']}"}
        elif intent == "set_temperature":
            self.temperature = {"value": slots.get("value", self.temperature["value"]),
                                "unit": slots.get("unit", self.temperature["unit"])}
            resp = {"ok": True,
                    "message": f"thermostat set to {self.temperature['value']}"
                               f"{self.temperature['unit']}"}
        elif intent == "set_timer":
            self.timers.append({"duration": slots.get("duration"),
                                "set_at": time.time()})
            resp = {"ok": True, "message": f"timer set: {slots.get('duration')}"}
        elif intent == "set_alarm":
            self.alarms.append({"time": slots.get("time"), "set_at": time.time()})
            resp = {"ok": True, "message": f"alarm set: {slots.get('time')}"}
        elif intent == "reminders_lists":
            if slots.get("action") == "list":
                resp = {"ok": True, "message": f"reminders: {self.reminders}"}
            else:
                self.reminders.append({"action": slots.get("action"),
                                       "set_at": time.time()})
                resp = {"ok": True, "message": f"reminder set: {slots.get('action')}"}
        elif intent == "play_music":
            self.media["state"] = "playing"
            resp = {"ok": True,
                    "message": f"playing: {slots.get('query') or 'music'}"}
        elif intent == "media_control":
            a = slots.get("action")
            if a == "pause":
                self.media["state"] = "paused"
            elif a == "stop":
                self.media["state"] = "stopped"
            elif a == "next":
                self.media["state"] = "next track"
            elif a == "previous":
                self.media["state"] = "previous track"
            elif a == "volume_up":
                self.media["volume"] = min(100, self.media["volume"] + 10)
            elif a == "volume_down":
                self.media["volume"] = max(0, self.media["volume"] - 10)
            elif a == "mute":
                self.media["muted"] = not self.media["muted"]
            resp = {"ok": True, "message": f"media: {a}"}
        elif intent == "call":
            resp = {"ok": True,
                    "message": f"calling {slots.get('contact')}... (mock)"}
        elif intent == "ask_question":
            resp = {"ok": True,
                    "message": f"(mock) answer for: {slots.get('query')}"}
        elif intent == "unknown":
            resp = {"ok": False, "message": "I didn't catch a command."}

        self.history.append({"ts": time.time(), "command": cmd, "response": resp})
        self.history = self.history[-50:]
        return resp

    def snapshot(self):
        return {
            "lights": self.lights,
            "temperature": self.temperature,
            "media": self.media,
            "timers": self.timers,
            "alarms": self.alarms,
            "reminders": self.reminders,
            "history": self.history[-20:],
        }


STATE = HomeState()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/state":
            self._json(STATE.snapshot())
        elif self.path in ("/", "/index.html"):
            body = DASHBOARD.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/command":
            n = int(self.headers.get("Content-Length", 0))
            try:
                cmd = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._json({"ok": False, "message": "bad json"}, 400)
                return
            self._json(STATE.handle(cmd))
        else:
            self._json({"error": "not found"}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    print(f"mock home on http://{args.host}:{args.port}/  (dashboard at /)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
