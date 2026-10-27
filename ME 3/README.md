# 🎙️ MCP Voice System (ME 3) - Raspberry Pi Installation Guide

This project implements a modular voice system using the Model Context Protocol (MCP), consisting of Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Voice Activity Detection (VAD).

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your Raspberry Pi. It is highly recommended to use a virtual environment to avoid system package conflicts.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate
```

### 2. Installation
Install the necessary dependencies. The MCP servers use the `fastmcp` package; the orchestrator uses the plain helper modules (`asr_server/transcribe.py`, `tts_server/synthesize.py`) directly and does **not** require MCP.

```bash
pip install fastmcp openai numpy
# Install additional VAD requirements if using the VAD system
pip install -r "mlo_repo/ME 3/vad_system/requirements.txt"
```

### 3. Environment Configuration
The ASR server requires an OpenAI API key for Whisper transcription.

```bash
export OPENAI_API_KEY="your_openai_key_here"
```

**Note:** The TTS server uses **local Piper TTS** — no ElevenLabs API key is required. Piper runs entirely on-device.

### 4. Piper TTS Setup (Local, No API Key)
The TTS server relies on the Piper binary and a voice model. Download Piper and a model, then place them in `/home/raesoriano/piper/` (the default) **or** override the locations with environment variables:

```bash
# Defaults (overridable):
export PIPER_BINARY="/home/raesoriano/piper/piper"
export PIPER_MODEL="/home/raesoriano/piper/en_US-lessac-medium.onnx"
export PIPER_CONFIG="/home/raesoriano/piper/en_US-lessac-medium.onnx.json"
export PIPER_LD_LIBRARY_PATH="/home/raesoriano/piper"
```

```bash
mkdir -p /home/raesoriano/piper
# Download piper binary (Linux aarch64 for Pi)
# Download a voice model, e.g. en_US-lessac-medium.onnx + .json
```

---

## 🛠️ Running the Components

The system is modular. You can run the servers independently or via the orchestrator.

### A. TTS Server (Text-to-Speech)
Converts text to audio using local Piper TTS (no API key needed).
```bash
python "mlo_repo/ME 3/tts_server/server.py"
```

### B. ASR Server (Speech-to-Text)
Transcribes audio files using OpenAI Whisper (requires OPENAI_API_KEY).
```bash
python "mlo_repo/ME 3/asr_server/server.py"
```

### C. VAD System (Voice Activity Detection)
Manages audio streaming and detects when a user is speaking.
```bash
python "mlo_repo/ME 3/vad_system/vad_manager.py"
```

### D. Agent Orchestrator
The main entry point that coordinates the ASR, TTS, and VAD components to create a full voice loop. It calls the plain helper modules directly (no MCP), so it only needs `openai`, `numpy`, `torch`, and `pyaudio`.
```bash
python "mlo_repo/ME 3/agent_orchestrator.py"
```

### E. Dashboard (Live Telemetry)
A Streamlit monitor that reads the orchestrator's real telemetry log. It shows live
latency metrics, the interaction stream, and trend charts. It auto-refreshes every
5 seconds and falls back to clearly-labelled simulated data only when no real
telemetry exists yet (i.e. before the agent completes a turn).
```bash
pip install streamlit pandas
streamlit run "mlo_repo/ME 3/dashboard.py"
```

### F. Latency Benchmark (Real Data)
Reports mean / P95 latency per category from the orchestrator's telemetry log.
```bash
python "mlo_repo/ME 3/latency_benchmark.py"            # real telemetry
python "mlo_repo/ME 3/latency_benchmark.py" --simulate # simulated demo only
```

---

## 📊 Telemetry
The orchestrator instruments every turn and writes two kinds of records to a
JSONL log (`telemetry.jsonl` next to the code, override with `TELEMETRY_FILE`):

- **latency** — one sample per category per turn: `vad_trigger`,
  `asr_turnaround`, `llm_ttfb`, `tts_synthesis`, `interruption`.
- **interaction** — each User / OnIt line, with its associated latency.

The dashboard and the latency benchmark both read this log (cross-process), so
they show **real** numbers once the agent has processed at least one turn.
`telemetry.py` is the shared module that defines the format and the read/write
API.

---

## ⚠️ Troubleshooting (Pi Specific)

**1. "ModuleNotFoundError: No module named 'fastmcp'"**
The MCP servers use the standalone `fastmcp` package. Install it:
`pip install fastmcp`
(The orchestrator does not need this — it uses the plain helper modules.)

**2. Permission Denied (Audio Files)**
If the server cannot save `.wav` or `.mp3` files, ensure the folder has the correct permissions:
`chmod -R 755 /home/raesoriano/sandbox`

**3. Path Issues**
Always run the commands from the root of the sandbox directory (`/home/raesoriano/sandbox`) to ensure relative imports and file paths work correctly.

**4. Virtual Environment**
If you are not using an active venv, call the python binary directly:
`/home/raesoriano/onit/.venv/bin/python "mlo_repo/ME 3/tts_server/server.py"`

**5. Piper Not Found**
If TTS fails with a synthesis error, verify the `PIPER_BINARY`, `PIPER_MODEL`, and `PIPER_CONFIG` environment variables (or their defaults in `tts_server/synthesize.py`) point to valid files.

## 📊 Project Structure
- `asr_server/`: Whisper-based transcription (OpenAI API).
- `tts_server/`: Piper-based synthesis (local, no API key).
- `vad_system/`: Real-time voice activity detection.
- `agent_orchestrator.py`: The "brain" connecting all components.
- `telemetry.py`: Shared telemetry sink (JSONL log + in-memory buffer).
- `dashboard.py`: Live monitoring interface (reads real telemetry).
- `latency_benchmark.py`: Reports real mean / P95 latency per category.
