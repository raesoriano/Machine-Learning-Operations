#!/usr/bin/env bash
# MLOps environment setup
# Creates a conda environment and installs requirements from this repo.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default: the environment lives inside this repo. Override with MLOPS_ENV_DIR,
# e.g. to place it on a shared filesystem in a cluster setup:
#   MLOPS_ENV_DIR=/shared/path/env bash setup_env.sh
ENV_DIR="${MLOPS_ENV_DIR:-$REPO_DIR/env}"

CONDA_BIN="${CONDA_BIN:-conda}"

"$CONDA_BIN" create -p "$ENV_DIR" python=3.11 pip -y
"$ENV_DIR/bin/pip" install -r "$REPO_DIR/requirements.txt"

echo
echo "Done. Activate with:"
echo "  conda activate $ENV_DIR"
