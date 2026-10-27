#!/usr/bin/env bash
#
# setup.sh — one-shot environment setup for ME 3 on a Raspberry Pi
#
# Usage:
#   bash setup.sh
#
# What it does:
#   1. Installs system packages (portaudio, python3-venv)
#   2. Creates a virtualenv in ./.venv
#   3. Installs all Python dependencies
#   4. Creates the Piper TTS directory if it doesn't exist
#   5. Prints the export lines you need in your shell
#
# It is idempotent — safe to re-run.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_DIR}/.venv"
PIPER_DIR="${PIPER_DIR:-$HOME/piper}"

echo "=============================================="
echo " ME 3 — Raspberry Pi setup"
echo " Repo: ${REPO_DIR}"
echo "=============================================="

# ------------------------------------------------------------------
# 1. System packages
# ------------------------------------------------------------------
echo ""
echo "[1/4] System packages..."
if command -v apt-get &>/dev/null; then
    if [ "$(id -u)" -ne 0 ]; then
        echo "  Installing portaudio19-dev and python3-venv (needs sudo)..."
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3-venv portaudio19-dev
    else
        apt-get update -qq
        apt-get install -y -qq python3-venv portaudio19-dev
    fi
else
    echo "  WARNING: apt-get not found. Skipping system packages."
    echo "  Make sure portaudio is installed before running pyaudio."
fi

# ------------------------------------------------------------------
# 2. Virtualenv
# ------------------------------------------------------------------
echo ""
echo "[2/4] Virtualenv..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "  Created ${VENV_DIR}"
else
    echo "  Already exists: ${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip -q
echo "  Python: $(python --version)"
echo "  Pip:    $(pip --version | awk '{print $2}')"

# ------------------------------------------------------------------
# 3. Python dependencies
# ------------------------------------------------------------------
echo ""
echo "[3/4] Python dependencies (this can take a while on a Pi)..."
pip install -q fastmcp openai numpy streamlit
pip install -q -r "${REPO_DIR}/vad_system/requirements.txt"
echo "  Installed:"
pip list --format=columns 2>/dev/null | grep -iE "torch|openai|fastmcp|pyaudio|numpy|streamlit|silero" || true

# ------------------------------------------------------------------
# 4. Piper TTS directory
# ------------------------------------------------------------------
echo ""
echo "[4/4] Piper TTS directory..."
if [ ! -d "${PIPER_DIR}" ]; then
    mkdir -p "${PIPER_DIR}"
    echo "  Created ${PIPER_DIR}"
    echo ""
    echo "  Download the Piper aarch64 binary and a voice model into that folder:"
    echo "    https://github.com/rhasspy/piper/releases"
    echo "    https://huggingface.co/rhasspy/piper-voices"
    echo ""
    echo "  Example (en_US-lessac-medium):"
    echo "    curl -L -o ${PIPER_DIR}/en_US-lessac-medium.onnx \\"
    echo "      https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    echo "    curl -L -o ${PIPER_DIR}/en_US-lessac-medium.onnx.json \\"
    echo "      https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
else
    echo "  Already exists: ${PIPER_DIR}"
fi

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "=============================================="
echo " Setup complete."
echo "=============================================="
echo ""
echo " Next steps — add these to your ~/.bashrc or run them in your shell:"
echo ""
cat <<EOF
source "${VENV_DIR}/bin/activate"
export OPENAI_API_KEY="your_key_here"
export PIPER_BINARY="${PIPER_DIR}/piper"
export PIPER_MODEL="${PIPER_DIR}/en_US-lessac-medium.onnx"
export PIPER_CONFIG="${PIPER_DIR}/en_US-lessac-medium.onnx.json"
export PIPER_LD_LIBRARY_PATH="${PIPER_DIR}"
EOF
echo ""
echo " Then run the agent:"
echo "   python agent_orchestrator.py"
echo ""
