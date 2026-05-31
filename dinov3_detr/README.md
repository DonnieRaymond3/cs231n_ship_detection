# DINOv3 + DETR baseline (HRSID -> SSDD)

A minimal object-detection baseline that pairs a **frozen DINOv3 backbone**
with a **DETR** head. It is **trained on
[HRSID](../HRSID/HRSID_JPG/)** and tested **cross-domain on the full
[SSDD](../SSDD/) dataset** (1,160 images), measuring how well DINO features
transfer across SAR sources.

The DETR transformer/query/bbox-regression layers are initialized from
`facebook/detr-resnet-50`; the ResNet backbone, input projection, and COCO class
classifier are skipped because DINOv3 and the single-class ship task require
different shapes.

The code uses the Hugging Face Backbone API, so the backbone can be swapped
from DINOv3 to DINOv2 with only a checkpoint/config change.

## Why this design

- DINOv3 is pretrained on optical imagery; SAR has a large domain gap, so the
  backbone is **frozen** by default and only the DETR head is trained. Add
  `--no-freeze-backbone` (or LoRA later) to experiment with adaptation.
- Both datasets are COCO with a single `ship` class, but the ids differ:
  **HRSID uses `category_id == 1`, SSDD uses `0`**. Both map to DETR label `0`;
  predictions are written back with each set's own id at eval time.
- DETR is NMS-free and easy to wire up; note it is comparatively weak on very
  small objects (SAR ships are tiny), so treat mAP here as a baseline.

## Data splits

| Role | Source | Images |
|------|--------|--------|
| Train | HRSID `train2017.json` | 3,642 |
| Val (in-domain monitor) | HRSID `test2017.json` | 1,962 |
| Test (cross-domain) | SSDD full (`ssdd_all.json`) | 1,160 |

Build the merged SSDD test file once:

```bash
cd dinov3_detr
python make_ssdd_coco.py    # writes ../SSDD/ssdd_all.json
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | Default paths (resolve to repo root) and hyperparameters |
| `dataset.py` | `HRSIDDetectionDataset` + DETR collate function |
| `model.py` | `build_model()` — DINO backbone + DETR head, frozen |
| `train.py` | `run_training()` + CLI via the HF `Trainer` (device-agnostic) |
| `evaluate.py` | `run_eval()` + CLI; COCO mAP via `pycocotools` (SSDD by default) |
| `make_ssdd_coco.py` | Merge SSDD train+test into one full-set COCO file |
| `modal_app.py` | Train + evaluate on a Modal GPU |

## Experiment tracking (Weights & Biases)

Training logs loss/lr curves through the HF `Trainer`, and evaluation logs the
SSDD COCO metrics (`ssdd/AP`, `ssdd/AP50`, ...). Passing the same run name to
both steps keeps everything in a single W&B run.

During training, the script also runs full HRSID validation COCO evaluation once
per epoch and logs `val/AP`, `val/AP50`, `val/AP75`, `val/AR_100`, etc. Use
`--no-epoch-map` to skip it or `--map-limit 200` for a faster approximate curve.

```bash
pip install wandb && wandb login        # local runs

python train.py --wandb-run-name my-run            # logs to project dinov3-detr-ship
python evaluate.py --wandb-run-name my-run         # attaches SSDD metrics to the same run
python train.py --no-wandb                         # disable tracking
```

Project name defaults to `dinov3-detr-ship` (`--wandb-project` to change).

## Setup

DINOv3 weights are gated on the Hugging Face Hub. Make sure your Hugging Face
token is approved for `facebook/dinov3-convnext-large-pretrain-lvd1689m`. Use a
fresh environment (Python >= 3.10, ideally a CUDA GPU).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login
```

## Train

```bash
cd dinov3_detr
python train.py --epochs 12 --batch-size 4

# quick smoke test on a tiny subset (1 epoch)
python train.py --limit 50 --epochs 1
```

Defaults: backbone `facebook/dinov3-convnext-large-pretrain-lvd1689m`, trained
on `HRSID/HRSID_JPG/annotations/train2017.json`, evaluated each epoch on
`test2017.json`. Compatible DETR weights are initialized from
`facebook/detr-resnet-50`. Checkpoints + image processor are saved to `outputs/`.

## Evaluate

```bash
cd dinov3_detr
# Cross-domain test on the full SSDD set (default; category_id 0)
python evaluate.py --model-dir outputs --threshold 0.0

# Optional in-domain HRSID test (category_id 1)
python evaluate.py --model-dir outputs \
    --ann ../HRSID/HRSID_JPG/annotations/test2017.json \
    --images ../HRSID/HRSID_JPG/JPEGImages --label-to-cat 1
```

Prints the standard COCO `bbox` summary (AP, AP50, AP75, AP for small/medium/
large objects).

## Visualize predictions

Generate a PNG with **green ground-truth boxes** and **red predicted boxes**.

On Modal, using the trained checkpoint in `/outputs/dinov3_detr`:

```bash
modal run dinov3_detr/modal_app.py::visualize_remote \
  --dataset ssdd \
  --index 0 \
  --threshold 0.3 \
  --output-name ssdd_000.png

modal volume get dinov3-detr-outputs /visualizations/ssdd_000.png ./ssdd_000.png
```

For HRSID validation:

```bash
modal run dinov3_detr/modal_app.py::visualize_remote \
  --dataset hrsid-val \
  --index 0 \
  --threshold 0.3 \
  --output-name hrsid_val_000.png
```

After pulling a checkpoint locally, you can also run:

```bash
python visualize.py --model-dir outputs --dataset ssdd --index 0 --threshold 0.3
```

## Train on Modal (GPU)

[Modal](https://modal.com) runs training on a cloud GPU. The app uploads both
datasets to a persistent Volume, trains on HRSID, and evaluates on SSDD.

```bash
pip install modal && modal setup

# 1. Store Hugging Face + W&B API keys
modal secret create huggingface HF_TOKEN=hf_xxx
modal secret create wandb WANDB_API_KEY=xxx

# 2. Build the SSDD test file, then upload data to a Modal Volume (once)
cd dinov3_detr
python make_ssdd_coco.py
modal run modal_app.py::upload

# 3. Train on HRSID + evaluate on SSDD (one shared W&B run)
modal run modal_app.py::main --epochs 12 --batch-size 8

# Faster experimental run: approximate per-epoch mAP on 200 val images
modal run modal_app.py::main --epochs 12 --batch-size 8 --map-limit 200

# Or run the steps separately (pass the same name to share a W&B run)
modal run modal_app.py::train_remote --epochs 12 --batch-size 8 --wandb-run-name my-run
modal run modal_app.py::evaluate_remote --wandb-run-name my-run
```

- **GPU:** defaults to `A100` for the heavier ConvNeXt-Large backbone. Override
  with `MODAL_GPU=H100 modal run ...` for maximum headroom, or `MODAL_GPU=A10G`
  for a cheaper run if memory allows.
- **Outputs:** checkpoints persist in the `dinov3-detr-outputs` Volume at
  `/outputs/dinov3_detr`; pull them with `modal volume get dinov3-detr-outputs /dinov3_detr ./outputs`.
- **OOM:** lower `--batch-size` (try 4) or use a smaller input size / backbone.

## Notes & knobs

- **Backbone size:** default is the heavy practical option,
  `facebook/dinov3-convnext-large-pretrain-lvd1689m`. Use `convnext-base` or
  `convnext-small` if runtime or memory is too high.
- **DETR init:** swap `--pretrained-detr facebook/detr-resnet-50` or pass an
  empty value to disable pretrained DETR initialization.
- **DINOv2 fallback:** use `--backbone facebook/dinov2-small` if gated DINOv3
  access is unavailable.
- **Device:** CUDA uses fp16 automatically; CPU/MPS run fp32 (slow — prefer a GPU).
- **Small objects:** consider higher input resolution or a Deformable/RT-DETR
  head as a follow-up; the backbone integration is identical.
- **Backbone injection:** if your `transformers` version errors in `forward()`
  after `model.model.backbone = backbone`, switch to
  `model.model.backbone.conv_encoder.model = backbone` (see `model.py`).
