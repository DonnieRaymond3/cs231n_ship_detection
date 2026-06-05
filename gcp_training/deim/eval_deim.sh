#!/usr/bin/env bash
# Run DEIM's full COCO evaluation (--test-only) on a trained checkpoint.
# Regenerates the config first so the eval input-size matches how the model was
# trained (DEIM's base configs are mutated per run, so this keeps them in sync).
#
#   IMGSZ=640 bash eval_deim.sh                 # eval the 640 model
#   IMGSZ=800 MODEL=s bash eval_deim.sh         # eval the 800 model
#   CKPT=/path/to/best_stg2.pth bash eval_deim.sh   # eval a specific checkpoint
set -euo pipefail

# DataLoader workers pass tensors via file descriptors; raise the limit to avoid
# "received 0 items of ancdata" (eval calls train.py directly, bypassing the
# train wrapper's file_system sharing strategy).
ulimit -n 65535 2>/dev/null || true

MODEL="${MODEL:-s}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-32}"
REPO="${REPO:-$HOME/cs231n_ship_detection}"
DEIM_DIR="${DEIM_DIR:-$HOME/DEIM}"
VENV="${VENV:-$HOME/.venv-deim}"

HERE="$(cd "$(dirname "$0")" && pwd)"
HRSID="$REPO/HRSID/HRSID_JPG"
IMG="$HRSID/JPEGImages"
TRAIN_JSON="$HRSID/annotations/train2017.json"
VAL_JSON="$HRSID/annotations/test2017.json"
OUT="$DEIM_DIR/outputs/deim_hrsid_${MODEL}_${IMGSZ}"

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Pick checkpoint: explicit CKPT, else best_stg2 -> best_stg1 -> last.
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
