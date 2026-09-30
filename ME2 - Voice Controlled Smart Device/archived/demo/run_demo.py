"""End-to-end demo driver — works TODAY, no audio, no trained model.

Simulates the full on-device loop:
    command text -> (mock ASR) -> vcm.parser -> mock_home backend -> dashboard

It is the rehearsal script for the real demo: on the RPi, the mock ASR step
is replaced by the ONNX VCM (deploy/rpi_service/server.py --file), and the
text list below is replaced by the live mic. Everything downstream is
identical, which is the point of the shared parser + backend contract.

Usage:
    python -m demo.run_demo            # prints the loop, no server needed
    # with the mock home running (python -m demo.mock_home.server --port 5000):
    python -m demo.run_demo --send --port 5000
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vcm.parser import parse  # noqa: E402

# The 10-command demo script (PLAN.md §6). One utterance per intent,
# plus one rejection at the end.
DEMO_SCRIPT = [
    "play lofi music",
    "what's the weather",
    "turn on the living room lights",
    "dim the kitchen lights to 40 percent",
    "set a timer for 10 minutes",
    "set an alarm for 7 am",
    "set the temperature to 24 degrees celsius",
    "volume up",
    "remind me to buy milk",
    "call mom",
    "hello, are you there",   # -> must be rejected (unknown)
]


def send(port, command):
    url = f"http://127.0.0.1:{port}/command"
    req = urllib.request.Request(
        url, data=json.dumps(command).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=2) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--send", action="store_true",
                    help="POST commands to a running mock_home server")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    print(f"{'utterance':34} {'intent':18} slots -> response")
    print("-" * 90)
    for text in DEMO_SCRIPT:
        cmd = parse(text)
        d = cmd.to_dict()
        resp = None
        if args.send:
            try:
                resp = send(args.port, d)
            except Exception as e:  # server down
                resp = {"ok": False, "message": f"(not sent: {e})"}
        line = f"{text:34} {d['intent']:18} {json.dumps(d['slots'])}"
        if resp:
            line += f" -> {resp['message']}"
        print(line)
    print("-" * 90)
    print("Open the dashboard (http://localhost:%d/) to watch state change."
          % args.port if args.send else
          "Run `python -m demo.mock_home.server` and add --send to see the dashboard update.")


if __name__ == "__main__":
    main()
