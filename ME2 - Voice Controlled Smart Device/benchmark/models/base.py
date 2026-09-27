"""Model interface — the swap point for the benchmark.

Any VCM implementation must expose:

    class MyModel:
        def load(self, path: str) -> None: ...
        def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
            \"\"\"Return the transcript string (words, no punctuation).\"\"\"

The benchmark then runs ``vcm.parser.parse(transcript)`` and scores it.
This is exactly what the RPi runtime does, so benchmark numbers transfer.
"""
import abc


class Model(abc.ABC):
    name = "abstract"

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """Load weights/config from path. No-op for stateless models."""

    @abc.abstractmethod
    def transcribe(self, audio, sr=16000) -> str:
        """audio: float32 mono numpy array. Returns transcript string."""
