"""RPi runtime service: VAD -> VCM (ONNX) -> parser -> local backend.

Everything on-device, no cloud, no LLM. Two modes:

  1. Live mic mode (on the RPi):
       python -m deploy.rpi_service.server --onnx model/checkpoints/vcm_v1_int8.onnx \
           --backend http --backend-url http://127.0.0.1:5000/command

  2. File mode (anywhere, for dev/testing without a mic):
       python -m deploy.rpi_service.server --onnx vcm_v1_int8.onnx --file cmd.wav
       -> prints the transcript, the parsed command, and the action it would send.

Pipeline per utterance:
    mic frames -> VAD (webrtcvad) -> 16 kHz float32 utterance
      -> log-mel [T,40]  (16 kHz native; no resampling)
      -> ONNX VCM -> CTC logits -> greedy decode (constrained vocab)
      -> vcm.parser.parse() -> {intent, slots}
      -> backend.send()  (local MQTT/HTTP)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmark.models.feats import log_mel  # noqa: E402
from model.decode import decode_to_text, BLANK  # noqa: E402
from vcm.parser import parse  # noqa: E402


class VCMPipeline:
    """The on-device model front-end (no I/O, easy to unit test).

    ``reject_min_conf`` is the OOD-reject gate: an utterance whose mean
    non-blank-frame log-prob is below the threshold decodes to "" and the
    parser returns {intent: "unknown"}, so the device stays silent instead
    of acting on a misheard command. Tune with
    scripts/tune_reject_threshold.py (default None = no gate, legacy
    behaviour).
    """

    def __init__(self, onnx_path, reject_min_conf=None):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"])
        self.in_name = self.session.get_inputs()[0].name
        self.reject_min_conf = reject_min_conf

    def transcribe(self, audio, sr=16000):
        """16 kHz float32 mono -> transcript string ('' if rejected)."""
        mel = log_mel(audio, sr=sr)                 # 16 kHz native
        logits = self.session.run(None, {self.in_name: mel[None]})[0]
        if self.reject_min_conf is not None:
            from model.decode import decode_with_confidence
            text, _ = decode_with_confidence(logits[0],
                                             min_conf=self.reject_min_conf)
            return text
        return decode_to_text(logits[0], blank=BLANK)

    def command(self, audio, sr=16000):
        """16 kHz float32 mono -> (transcript, Command)."""
        transcript = self.transcribe(audio, sr=sr)
        return transcript, parse(transcript)


def run_file_mode(pipeline, wav_path, backend=None):
    from scipy.io import wavfile
    sr, x = wavfile.read(wav_path)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    t0 = time.time()
    transcript, cmd = pipeline.command(x, sr)
    dt = time.time() - t0
    print(f"transcript : {transcript!r}")
    print(f"command    : {cmd.to_dict()}")
    print(f"latency    : {dt * 1000:.0f} ms  "
          f"(RTF {dt / max(len(x) / sr, 1e-6):.3f})")
    if backend is not None and cmd.intent != "unknown":
        backend.send(cmd.to_dict())
        print("action     : sent to backend")
    return cmd


def run_live_mode(pipeline, backend, sr=16000, frame_ms=30):
    import sounddevice as sd
    from .vad import VAD, collect_utterance

    vad = VAD(sr=sr, frame_ms=frame_ms, aggressiveness=2)
    flen = vad.frame_bytes_len()

    def frames():
        with sd.RawInputStream(samplerate=sr, blocksize=flen // 2,
                               dtype="int16", channels=1):
            while True:
                data, _ = sd.read(flen // 2)
                yield data.tobytes()

    it = frames()
    print("listening... (Ctrl-C to stop)")
    while True:
        audio, sr = collect_utterance(vad, it, sr)
        if audio is None:
            continue
        t0 = time.time()
        transcript, cmd = pipeline.command(audio, sr)
        dt = time.time() - t0
        print(f"[{dt * 1000:5.0f} ms] {transcript!r} -> {cmd.to_dict()}")
        if cmd.intent != "unknown":
            backend.send(cmd.to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--file", default=None, help="process one WAV and exit")
    ap.add_argument("--backend", default=None, choices=["mqtt", "http"],
                    help="omit to only print (no action sent)")
    ap.add_argument("--backend-url", default="http://127.0.0.1:5000/command")
    ap.add_argument("--mqtt-host", default="localhost")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--reject-min-conf", type=float, default=None,
                    help="OOD-reject gate: mean non-blank log-prob below "
                         "this -> no action (tune with "
                         "scripts/tune_reject_threshold.py)")
    args = ap.parse_args()

    pipeline = VCMPipeline(args.onnx, reject_min_conf=args.reject_min_conf)
    backend = None
    if args.backend:
        from .backend import make_backend
        backend = make_backend(args.backend,
                               host=args.mqtt_host,
                               url=args.backend_url)
    if args.file:
        run_file_mode(pipeline, args.file, backend)
    else:
        run_live_mode(pipeline, backend, sr=args.sr)


if __name__ == "__main__":
    main()
