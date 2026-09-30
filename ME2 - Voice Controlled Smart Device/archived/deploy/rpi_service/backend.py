"""Local smart-home backend client for the RPi service.

The VCM emits {intent, slots}; this client turns that into a device action.
Two transports, both LOCAL (no cloud):

  * mqtt  — paho-mqtt to a broker on the LAN (e.g. Mosquitto on the RPi or a
            home hub). Topic scheme:  home/<intent>/<location-or-default>
  * http  — POST JSON to a local Flask endpoint (demo/mock_home).

Payload (both):
    {"intent": "lights_switch", "slots": {"state": "on", "location": "kitchen"},
     "ts": 1726070000.123}
"""
import json
import time
import urllib.request


class MqttBackend:
    def __init__(self, host="localhost", port=1883, client_id="vcm-rpi"):
        import paho.mqtt.client as mqtt  # local import: optional dep
        self.client = mqtt.Client(client_id=client_id)
        self.client.connect(host, port, keepalive=60)
        self.client.loop_start()

    def send(self, command):
        loc = command["slots"].get("location", "default")
        topic = f"home/{command['intent']}/{loc}"
        payload = json.dumps({**command, "ts": time.time()})
        self.client.publish(topic, payload, qos=1)


class HttpBackend:
    def __init__(self, url="http://127.0.0.1:5000/command"):
        self.url = url

    def send(self, command):
        data = json.dumps({**command, "ts": time.time()}).encode()
        req = urllib.request.Request(
            self.url, data=data,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=2) as r:
            r.read()


def make_backend(kind="http", **kw):
    kind = kind.lower()
    if kind == "mqtt":
        return MqttBackend(**kw)
    if kind == "http":
        return HttpBackend(**kw)
    raise ValueError(f"unknown backend: {kind} (have: mqtt, http)")
