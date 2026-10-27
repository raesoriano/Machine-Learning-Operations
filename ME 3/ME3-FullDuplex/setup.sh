#!/usr/bin/env bash
#
# setup.sh — environment setup for ME3-FullDuplex on a Raspberry Pi
#
# Usage:
#   bash ME3-FullDuplex/setup.sh
#
# ME3-FullDuplex adds NO new big dependencies:
#   - ASR stays on the OpenAI Whisper API (no local Whisper weights)
#   - Echo cancellation uses SpeexDSP if present, else a pure-NumPy NLMS
#     fallback (no hard dependency). We *recommend* installing libspeexdsp
#     (small, ~10-20 MB) for better echo suppression — see below.
#   - TTS reuses the existing Piper install
#
# This script runs the parent ME 3 setup (portaudio, venv, Python deps),
# then optionally installs libspeexdsp for the AEC. Safe to re-run.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "${HERE}/.." && pwd)"

echo "=============================================="
echo " ME3-FullDuplex — Raspberry Pi setup"
echo " (delegates to the parent ME 3 setup)"
echo "=============================================="

bash "${PARENT}/setup.sh"

# ---------------------------------------------------------------------- #
# Optional: SpeexDSP for the AEC (recommended)
#
# The AEC works without it (pure-NumPy NLMS fallback), but SpeexDSP gives
# noticeably better echo suppression and is small. This is best-effort: if
# sudo/apt is unavailable the agent still runs on the fallback backend.
# ---------------------------------------------------------------------- #
if command -v apt-get >/dev/null 2>&1; then
    echo ""
    echo "Installing libspeexdsp (recommended for AEC)..."
    if command -v sudo >/dev/null 2>&1; then
        sudo apt-get install -y libspeexdsp-dev || \
            echo "  (could not install libspeexdsp — AEC will use the NumPy fallback)"
    else
        apt-get install -y libspeexdsp-dev || \
            echo "  (could not install libspeexdsp — AEC will use the NumPy fallback)"
    fi
else
    echo ""
    echo "(apt-get not found — AEC will use the pure-NumPy NLMS fallback)"
fi

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
