#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65535 2>/dev/null || true

MODEL="${MODEL:-s}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-32}"
REPO="${REPO:-$HOME/cs231n_ship_detection}"
DEIM_DIR="${DEIM_DIR:-$HOME/DEIM}"
VENV="${VENV:-$HOME/.venv-deim}"

HERE="$(cd "$(dirname "$0")" && pwd)"
HRSID="$REPO/HRSID/HRSID_JPG"
IMG="${IMG:-$HRSID/JPEGImages}"
TRAIN_JSON="${TRAIN_JSON:-$HRSID/annotations/train2017.json}"
VAL_JSON="${VAL_JSON:-$HRSID/annotations/test2017.json}"
OUT="$DEIM_DIR/outputs/deim_hrsid_${MODEL}_${IMGSZ}"

source "$VENV/bin/activate"

if [ -z "${CKPT:-}" ]; then
  for c in best_stg2.pth best_stg1.pth last.pth; do
    [ -f "$OUT/$c" ] && { CKPT="$OUT/$c"; break; }
  done
fi
[ -n "${CKPT:-}" ] && [ -f "$CKPT" ] || {
  echo "No checkpoint found in $OUT. Pass CKPT=/path/to/x.pth"; ls "$OUT" 2>/dev/null; exit 1;
}

echo "==> Syncing config to ${IMGSZ}px"
python "$HERE/gen_config.py" \
  --deim-root "$DEIM_DIR" --size "$MODEL" \
  --img-dir "$IMG" --train-json "$TRAIN_JSON" --val-json "$VAL_JSON" \
  --epochs 50 --batch "$BATCH" --imgsz "$IMGSZ" >/dev/null

echo "==> Evaluating $CKPT at ${IMGSZ}px on HRSID test split"
cd "$DEIM_DIR"
python train.py -c "configs/deim_dfine/deim_hrsid_${MODEL}.yml" --test-only -r "$CKPT"
