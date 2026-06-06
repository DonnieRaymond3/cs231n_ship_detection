#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cs231n_ship_detection}"

echo "==> Verifying GPU is visible"
nvidia-smi || { echo "ERROR: nvidia-smi failed. GPU driver not ready yet."; exit 1; }

cd "$REPO_DIR/gcp_training"

echo "==> Installing system deps (python3-venv, OpenCV runtime libs)"
sudo apt-get update -qq
sudo apt-get install -y python3-venv libgl1 libglib2.0-0

echo "==> Creating virtualenv (.venv)"
python3 -m venv "$HOME/.venv"
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
