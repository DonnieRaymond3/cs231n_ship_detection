# DINO Detector on HRSID — Modal Training

Fine-tune **DINO: DETR with Improved DeNoising Anchor Boxes** from
[IDEA-Research/DINO](https://github.com/IDEA-Research/DINO) on HRSID, then
evaluate cross-domain on SSDD with the same COCO metric style used by
`dinov3_detr`.

This is Zhang et al.'s detector architecture, not the Meta DINOv2/DINOv3
feature-backbone baseline in `dinov3_detr`.

## What This Adds

| File | Purpose |
|------|---------|
| `modal_app.py` | Modal image, dataset upload, checkpoint download, train/eval entry points |
| `train.py` | Wrapper around official DINO `main.py` for HRSID fine-tuning |
| `evaluate.py` | DINO inference plus `pycocotools.COCOeval` reporting |
| `gen_config.py` | Generates the HRSID R50 4-scale DINO config |
| `data.py` | Creates COCO `train2017` / `val2017` symlink layouts |
| `metrics.py` | Maps official DINO logs to W&B keys like `val/AP` |

## Metrics

Training uses official DINO validation on HRSID `test2017.json`. The wrapper
replays `log.txt` into W&B using the same key style as `dinov3_detr`:

- `val/AP`, `val/AP50`, `val/AP75`
- `val/AP_small`, `val/AP_medium`, `val/AP_large`
- `val/AR_1`, `val/AR_10`, `val/AR_100`, and size-stratified AR

Final SSDD evaluation uses `evaluate.py`, which converts DINO predictions to
COCO detection JSON and runs `pycocotools.COCOeval(iouType="bbox")`. HRSID
predictions are exported with `category_id=1`; SSDD predictions are exported
with `category_id=0`.

## Setup

Create the W&B secret once:

```bash
modal secret create wandb WANDB_API_KEY=xxx
```

Build the SSDD merged COCO file if needed:

```bash
python dinov3_detr/make_ssdd_coco.py
```

Upload HRSID and SSDD to the shared Modal volume:

```bash
modal run dino_detector/modal_app.py::upload
```

The first Modal run may still build the training image because the app registers
train/eval functions, but DINO's CUDA ops are compiled lazily inside GPU
containers, not during upload.

## Pretrained Checkpoint

Use the official **DINO R50 4-scale 12-epoch COCO checkpoint**
(`checkpoint0011_4scale.pth`). The official project hosts checkpoints in a
Google Drive model zoo, so the wrapper keeps the path configurable rather than
hard-coding a fragile file URL.

If you have a direct/share URL for the checkpoint:

```bash
modal run dino_detector/modal_app.py::download_pretrained --url "https://drive.google.com/..."
```

By default this stores the checkpoint at:

```text
/outputs/dino_detector/pretrained/checkpoint0011_4scale.pth
```

You can also pass any existing Modal path with `--pretrain-model-path` /
`--pretrain_model_path`.

## Train And Evaluate

Run the default HRSID fine-tune, then SSDD cross-domain evaluation:

```bash
modal run dino_detector/modal_app.py::main --epochs 12 --batch-size 2
```

Use a cheaper/smaller smoke run first:

```bash
modal run dino_detector/modal_app.py::smoke
```

Run pieces separately:

```bash
modal run dino_detector/modal_app.py::train_remote \
  --epochs 12 \
  --batch-size 2 \
  --wandb-run-name dino-r50-hrsid

modal run dino_detector/modal_app.py::evaluate_remote \
  --dataset ssdd \
  --wandb-run-name dino-r50-hrsid
```

Override GPU from your shell:

```bash
MODAL_GPU=A100 modal run dino_detector/modal_app.py::main
MODAL_GPU=H100 modal run dino_detector/modal_app.py::main
```

## Local Use

For local runs, clone and build the official DINO repo yourself, then point
`--dino-repo` at it:

```bash
git clone https://github.com/IDEA-Research/DINO.git /tmp/DINO
cd /tmp/DINO/models/dino/ops && python setup.py build install

python dino_detector/train.py \
  --dino-repo /tmp/DINO \
  --pretrain-model-path dino_detector/pretrained/checkpoint0011_4scale.pth
```

## Notes

- Internally, the generated config uses `num_classes=2` because HRSID stores the
  single `ship` category as raw `category_id == 1`; class index 0 is unused.
- Official DINO's custom CUDA ops cannot be compiled during Modal image build
  because no GPU is visible there. The train/eval image uses NVIDIA's CUDA devel
  base image and `modal_app.py` builds the ops when `train_remote` or
  `evaluate_remote` starts on a GPU.
- Official DINO evaluates HRSID at every epoch and writes `log.txt`. The wrapper
  streams live train metrics from DINO's progress output to W&B during training
  and logs epoch-level HRSID COCO metrics from `log.txt` at the end of training.
- The SSDD evaluation wrapper intentionally remaps predictions to
  `category_id=0` so metrics match `SSDD/ssdd_all.json`.
