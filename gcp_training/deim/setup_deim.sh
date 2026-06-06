#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-s}"
DEIM_DIR="${DEIM_DIR:-$HOME/DEIM}"
VENV="${VENV:-$HOME/.venv-deim}"

declare -A CKPT_ID=(
  [n]=1ZPEhiU9nhW4M5jLnYOFwTSLQC1Ugf62e
  [s]=1tB8gVJNrfb6dhFvoHJECKOF5VpkthhfC
  [m]=18Lj2a6UN6k_n_UzqnJyiaiLGpDzQQit8
  [l]=1PIRf02XkrA2xAD3wEiKE2FaamZgSGTAr
  [x]=1dPtbgtGgq1Oa7k_LgH1GXPelg1IVeu0j
)

echo "==> Verifying GPU"
nvidia-smi || { echo "ERROR: GPU not ready"; exit 1; }

echo "==> System deps"
sudo apt-get update -qq
sudo apt-get install -y python3-venv libgl1 libglib2.0-0 git

echo "==> Cloning DEIM -> $DEIM_DIR"
[ -d "$DEIM_DIR" ] || git clone https://github.com/ShihuaHuang95/DEIM.git "$DEIM_DIR"

echo "==> venv -> $VENV"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r "$DEIM_DIR/requirements.txt"
pip install "torch==2.1.2" "torchvision==0.16.2" --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2"
pip install wandb gdown

echo "==> Downloading COCO-pretrained DEIM-${MODEL} checkpoint"
mkdir -p "$DEIM_DIR/ckpts"
gdown "${CKPT_ID[$MODEL]}" -O "$DEIM_DIR/ckpts/deim_dfine_${MODEL}_coco.pth"

echo "==> Sanity check"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo
echo "Setup done. Next:"
echo "  source $VENV/bin/activate"
echo "  cd ~/cs231n_ship_detection/gcp_training/deim"
echo "  wandb login"
echo "  MODEL=$MODEL bash run_deim.sh"
