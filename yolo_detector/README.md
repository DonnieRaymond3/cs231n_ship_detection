# YOLOv11l Detector on HRSID -> SSDD

Train an **Ultralytics YOLO 11 large** detector on
[HRSID](../HRSID/HRSID_JPG/) and evaluate cross-domain on the full
[SSDD](../SSDD/) dataset, matching the Modal and W&B workflow used by
`dino_detector` and `dinov3_detr`.

## Why This Baseline

YOLOv11l is a strong one-stage detector with fast training and inference. This
folder keeps the experiment comparable with the other survey models:

- Train on HRSID `train2017.json`.
- Monitor in-domain HRSID `test2017.json`.
- Evaluate final cross-domain transfer on SSDD `ssdd_all.json`.
- Log comparable COCO metrics as `val/*` and `ssdd/*` in one W&B run.

HRSID stores ship annotations with `category_id == 1`; SSDD uses `0`. YOLO
trains a single internal class `0`, then evaluation exports predictions with the
category id expected by each dataset.

## Files

| File | Purpose |
|------|---------|
| `config.py` | Default paths, hyperparameters, W&B project, category ids |
| `data.py` | COCO-to-YOLO conversion for generated train/val datasets |
| `train.py` | `run_training()` + CLI around Ultralytics YOLO training |
| `evaluate.py` | `run_eval()` + CLI; COCO mAP via `pycocotools` |
| `metrics.py` | Shared COCO metric names and W&B logging helper |
| `modal_app.py` | Upload, train, smoke, and eval on Modal GPUs |

## Data Splits

| Role | Source | Images |
|------|--------|--------|
| Train | HRSID `train2017.json` | 3,642 |
| Val (in-domain monitor) | HRSID `test2017.json` | 1,962 |
| Test (cross-domain) | SSDD full (`ssdd_all.json`) | 1,160 |

Build the merged SSDD test file once if it is missing:

```bash
python dinov3_detr/make_ssdd_coco.py
```

## Weights & Biases

Training initializes/resumes a W&B run using the supplied run name. Ultralytics
logs native training curves, and this wrapper logs COCOeval metrics with the
same naming convention as the other baselines:

- `val/AP`, `val/AP50`, `val/AP75`, `val/AR_100`, ...
- `ssdd/AP`, `ssdd/AP50`, `ssdd/AP75`, `ssdd/AR_100`, ...

Use `--no-wandb` for local runs without tracking.

## Local Usage

Install dependencies in a CUDA-capable Python environment:

```bash
pip install ultralytics pycocotools wandb
wandb login
```

Train on HRSID:

```bash
python yolo_detector/train.py --wandb-run-name yolo11l-hrsid
```

Evaluate the trained checkpoint on SSDD:

```bash
python yolo_detector/evaluate.py --wandb-run-name yolo11l-hrsid
```

Run a tiny local smoke test:

```bash
python yolo_detector/train.py --epochs 1 --batch-size 1 --limit 8 --map-limit 4
python yolo_detector/evaluate.py --limit 4 --batch-size 1 --metric-prefix val \
  --images HRSID/HRSID_JPG/JPEGImages \
  --ann HRSID/HRSID_JPG/annotations/test2017.json \
  --label-to-cat 1
```

## Modal Usage

Create the W&B secret once:

```bash
modal secret create wandb WANDB_API_KEY=xxx
```

Upload HRSID and SSDD to the shared `ship-data` volume:

```bash
modal run yolo_detector/modal_app.py::upload
```

Run a small Modal smoke test:

```bash
modal run yolo_detector/modal_app.py::smoke
```

Train YOLOv11l on HRSID and then evaluate on SSDD in the same W&B run:

```bash
modal run yolo_detector/modal_app.py::main --epochs 12 --batch-size 8
```

Override the GPU type with `MODAL_GPU`, for example:

```bash
MODAL_GPU=A10G modal run yolo_detector/modal_app.py::main --epochs 12 --batch-size 8
```

Checkpoints and prediction JSON files are stored in the `yolo-detector-outputs`
Modal volume under `/outputs/yolo_detector`.
