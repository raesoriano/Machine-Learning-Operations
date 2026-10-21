import numpy as np
import torch
import pyaudio
import time
from typing import Callable

class VADManager:
    def __init__(self, sample_rate=16000, chunk_size=512, on_speech_start: Callable = None, on_speech_end: Callable = None):
        """
        VAD Manager using Silero VAD for high-accuracy voice detection on Raspberry Pi.
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        
        # Load Silero VAD model
        self.model, utils = torch.hub.load(repo_or_dir='snyprow/silero-vad', model='silero_vad', force_reload=False)
        (self.get_speech_timestamps, _, self.read_audio, _, _) = utils
        
        self.is_speaking = False
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

    def listen(self):
        print("VAD Listening... (Press Ctrl+C to stop)")
        try:
            while True:
                # Read audio chunk
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_int16 = np.frombuffer(data, dtype=np.int16)
                # Normalize to float32 for Silero
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                tensor_audio = torch.from_numpy(audio_float32)

                # Get probability of speech
                speech_prob = self.model(tensor_audio, self.sample_rate).item()

                if speech_prob > 0.5 and not self.is_speaking:
                    self.is_speaking = True
                    print("[VAD] Speech Started")
                    if self.on_speech_start:
                        self.on_speech_start()

                elif speech_prob < 0.3 and self.is_speaking:
                    self.is_speaking = False
                    print("[VAD] Speech Ended")
                    if self.on_speech_end:
                        self.on_speech_end()

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

# Example usage for testing
if __name__ == "__main__":
    def handle_start():
        print(">>> Triggering ASR / Interrupting TTS...")

    def handle_end():
        print(">>> Finalizing Transcription...")

    vad = VADManager(on_speech_start=handle_start, on_speech_end=handle_end)
    vad.listen()
