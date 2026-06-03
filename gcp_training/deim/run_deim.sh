#!/usr/bin/env bash
# Generate the HRSID config and launch DEIM fine-tuning with live W&B logging.
# Run ON the VM after setup_deim.sh.
#
#   MODEL=s EPOCHS=50 BATCH=32 bash run_deim.sh
#   NO_WANDB=1 bash run_deim.sh        # disable W&B
set -euo pipefail

# Raise FD limit to avoid DataLoader "received 0 items of ancdata".
ulimit -n 65535 2>/dev/null || true

MODEL="${MODEL:-s}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-32}"
IMGSZ="${IMGSZ:-640}"
REPO="${REPO:-$HOME/cs231n_ship_detection}"
DEIM_DIR="${DEIM_DIR:-$HOME/DEIM}"
VENV="${VENV:-$HOME/.venv-deim}"
WANDB_PROJECT="${WANDB_PROJECT:-deim-hrsid}"

HERE="$(cd "$(dirname "$0")" && pwd)"
HRSID="$REPO/HRSID/HRSID_JPG"
IMG="$HRSID/JPEGImages"
TRAIN_JSON="$HRSID/annotations/train2017.json"
VAL_JSON="$HRSID/annotations/test2017.json"
CKPT="$DEIM_DIR/ckpts/deim_dfine_${MODEL}_coco.pth"

for p in "$IMG" "$TRAIN_JSON" "$VAL_JSON" "$CKPT"; do
  [ -e "$p" ] || { echo "ERROR: missing $p (run setup_deim.sh first)"; exit 1; }
done

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "==> Generating config"
python "$HERE/gen_config.py" \
  --deim-root "$DEIM_DIR" --size "$MODEL" \
  --img-dir "$IMG" --train-json "$TRAIN_JSON" --val-json "$VAL_JSON" \
  --epochs "$EPOCHS" --batch "$BATCH" --imgsz "$IMGSZ"

REL_CONFIG="configs/deim_dfine/deim_hrsid_${MODEL}.yml"
OUT="outputs/deim_hrsid_${MODEL}_${IMGSZ}"

echo "==> Training DEIM-${MODEL} for ${EPOCHS} epochs"
python "$HERE/train_deim_wandb.py" \
  --deim-root "$DEIM_DIR" \
  --config "$REL_CONFIG" \
  --tuning "$CKPT" \
  --summary-dir "$OUT/summary" \
  --output-dir "$OUT" \
  --wandb-project "$WANDB_PROJECT" \
  --name "deim_hrsid_${MODEL}_${IMGSZ}_${EPOCHS}ep" \
  ${NO_WANDB:+--no-wandb}

echo
echo "Done. Outputs in $DEIM_DIR/$OUT"
echo "  log.txt (per-epoch metrics), checkpoints (best_stg*.pth, last.pth), summary/ (TensorBoard)"
