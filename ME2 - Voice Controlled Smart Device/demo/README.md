# Real-World Demo (individual task 5)

A live, fully on-device demo: talk to a Raspberry Pi, it runs the VCM, and it
controls a (mock) smart home — no cloud, no LLM.

## What it shows

1. **10-command script** — one utterance per intent (PLAN.md §6), ending with
   a rejection ("hello, are you there" → "I didn't catch a command").
2. **Live dashboard** — `demo/mock_home/dashboard/index.html` shows lights,
   thermostat, media, timers, reminders, and a command log updating in real
   time.
3. **Latency readout** — every command prints its end-to-end ms (target p50
   ≤ 500 ms on RPi4).

## Bill of materials

| Item | Purpose | Notes |
|------|---------|-------|
| Raspberry Pi 4 (4 GB) or 5 | VCM host | shared between students OK |
| ReSpeaker 2-Mic HAT (or any USB mic) | far-field-ish capture | 16 kHz mono is enough |
| LED strip + relay (optional) | physical "lights" | mock_home can drive GPIO |
| (optional) small speaker | TTS feedback | offline only |

## Setup (on the RPi)

```bash
sudo apt install mosquitto mosquitto-clients   # only if using the MQTT backend
python3 -m venv vcm && source vcm/bin/activate
pip install -r deploy/rpi_service/requirements.txt
```

Copy the int8 model (built on the dev machine via `deploy/export_onnx.py` +
`deploy/quantize.py`) to the RPi, e.g. `~/vcm/vcm_v1_int8.onnx`.

## Run

```bash
# 1. mock home + dashboard (terminal A)
python -m demo.mock_home.server --port 5000

# 2. VCM service (terminal B) — live mic
python -m deploy.rpi_service.server \
    --onnx ~/vcm/vcm_v1_int8.onnx \
    --backend http --backend-url http://127.0.0.1:5000/command

# 3. open the dashboard
#    http://<rpi-ip>:5000/
```

No mic handy? Rehearse the exact same downstream path with a WAV file:

```bash
python -m deploy.rpi_service.server --onnx vcm_v1_int8.onnx --file cmd.wav
# or, without any model at all (parser + backend only):
python -m demo.run_demo --send --port 5000
```

## Recording the demo video

1. Show the dashboard, then say each of the 10 commands; capture the state
   change + latency line for each.
2. Say the rejection; show it is *not* executed.
3. `ping` the RPi to a public site during the run to prove no cloud egress
   (or run with Wi-Fi in airplane mode — it still works).

## Swapping the backend

The VCM service only depends on the `{intent, slots}` JSON contract. To go
from the mock to a real hub, change one flag:

```bash
--backend mqtt --mqtt-host <hub-ip>
```

No model or parser changes (see `deploy/rpi_service/backend.py`).
