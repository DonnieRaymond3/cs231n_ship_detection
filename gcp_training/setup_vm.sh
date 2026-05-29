#!/usr/bin/env bash
# Run this ON the GCP VM after SSHing in. Installs Python deps into a venv.
# Assumes an NVIDIA Deep Learning VM image (CUDA + driver + PyTorch preinstalled).
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cs231n_ship_detection}"

echo "==> Verifying GPU is visible"
nvidia-smi || { echo "ERROR: nvidia-smi failed. GPU driver not ready yet."; exit 1; }

cd "$REPO_DIR/gcp_training"

if ! python3 -m venv --help >/dev/null 2>&1; then
  echo "==> Installing python3-venv (missing on base image)"
  sudo apt-get update -qq
  sudo apt-get install -y python3-venv
fi

echo "==> Creating virtualenv (.venv)"
python3 -m venv "$HOME/.venv"
# shellcheck disable=SC1091
source "$HOME/.venv/bin/activate"
pip install --upgrade pip

echo "==> Installing requirements"
pip install -r requirements.txt

echo "==> Sanity check"
python -c "import torch, ultralytics; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); ultralytics.checks()"

echo
echo "Setup done. Next:"
echo "  source ~/.venv/bin/activate"
echo "  cd $REPO_DIR/gcp_training"
echo "  python prepare_hrsid.py"
echo "  wandb login        # paste your API key"
echo "  python train_rtdetr.py --epochs 50"
