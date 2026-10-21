# MCP Voice Servers
# Phase 1: ASR and TTS Implementation

## ASR Server
- **Model**: OpenAI Whisper (via API)
- **Function**: `transcribe_audio(audio_path)`
- **Implementation**: Uses `FastMCP` to expose a tool that takes an audio file path and returns the transcription.

## TTS Server
- **Model**: ElevenLabs
- **Function**: `synthesize_speech(text, output_path)`
- **Implementation**: Uses `FastMCP` to expose a tool that takes text and saves the resulting audio to a specified path.

## Setup
1. Install dependencies: `pip install mcp openai requests`
2. Set environment variables: `OPENAI_API_KEY`, `ELEVENLABS_API_KEY`
3. Run servers: `python server.py`
