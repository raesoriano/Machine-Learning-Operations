#!/usr/bin/env bash
# AI 231 — MLOps environment setup
# Creates a conda env on the shared filesystem (visible from all DGX cluster nodes)
# and installs requirements from this repo.
set -euo pipefail

ENV_DIR="${CONDA_BASE:-/opt/miniconda3}/envs/ai231"
# On this cluster the scheduler may land jobs on different nodes, and /opt is
# node-local, so the env lives on the shared filesystem by default:
ENV_DIR="${AI231_ENV_DIR:-/mnt/jfs_hpc/home/ron.andrei.soriano/sandbox/ai231-env}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

/opt/miniconda3/bin/conda create -p "$ENV_DIR" python=3.11 pip -y
"$ENV_DIR/bin/pip" install -r "$REPO_DIR/requirements.txt"

echo
echo "Done. Activate with:"
echo "  conda activate $ENV_DIR"
