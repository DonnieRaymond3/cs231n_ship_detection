#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/cs231n_ship_detection}"
VENV="${VENV:-$HOME/.venv-rfdetr}"

echo "==> Verifying GPU"
nvidia-smi || { echo "ERROR: GPU not ready"; exit 1; }

echo "==> System deps (OpenCV runtime libs, venv)"
sudo apt-get update -qq
sudo apt-get install -y python3-venv libgl1 libglib2.0-0

echo "==> venv -> $VENV"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install "rfdetr[train,loggers]" wandb

echo "==> Building HRSID dataset for RF-DETR"
python "$REPO_DIR/gcp_training/rfdetr/prep_rfdetr_data.py"

echo "==> Sanity check"
python -c "import torch, rfdetr; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo
echo "Setup done. Next:"
echo "  source $VENV/bin/activate"
echo "  cd $REPO_DIR/gcp_training/rfdetr"
echo "  wandb login"
echo "  python train_rfdetr.py --model medium --epochs 50"
