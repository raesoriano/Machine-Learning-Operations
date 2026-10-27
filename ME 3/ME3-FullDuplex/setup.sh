#!/usr/bin/env bash
#
# setup.sh — environment setup for ME3-FullDuplex on a Raspberry Pi
#
# Usage:
#   bash ME3-FullDuplex/setup.sh
#
# ME3-FullDuplex adds NO new big dependencies:
#   - ASR stays on the OpenAI Whisper API (no local Whisper weights)
#   - Echo handling is a zero-dependency gate (no SpeexDSP)
#   - TTS reuses the existing Piper install
#
# So this script just runs the parent ME 3 setup, which installs
# portaudio, the venv, and all Python deps (torch, pyaudio, silero-vad,
# fastmcp, openai, numpy, streamlit). Safe to re-run.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "${HERE}/.." && pwd)"

echo "=============================================="
echo " ME3-FullDuplex — Raspberry Pi setup"
echo " (delegates to the parent ME 3 setup)"
echo "=============================================="

bash "${PARENT}/setup.sh"

echo ""
echo "ME3-FullDuplex is ready. To run it:"
echo ""
echo "  cd ${PARENT}"
echo "  source .venv/bin/activate"
echo "  export OPENAI_API_KEY=sk-..."
echo "  python ME3-FullDuplex/agent_orchestrator.py"
echo ""
echo "Tip: check your Piper version for --output-raw (Piper >= 1.2):"
echo "  \${PIPER_BINARY:-\$HOME/piper/piper} --help | grep output-raw"
echo "  If it's missing, ME3-FullDuplex falls back to WAV-per-sentence streaming."
