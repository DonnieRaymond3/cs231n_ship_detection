"""Run YOLOv11l HRSID -> SSDD training on Modal with a GPU.

Workflow:
    1. Create a W&B secret:
         modal secret create wandb WANDB_API_KEY=xxx
    2. Build the SSDD test file locally if needed:
         python dinov3_detr/make_ssdd_coco.py
    3. Upload datasets:
         modal run yolo_detector/modal_app.py::upload
    4. Train on HRSID, then evaluate on the full SSDD set:
         modal run yolo_detector/modal_app.py::main --epochs 12 --batch-size 8
"""

from __future__ import annotations

import os

import modal

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))

GPU = os.environ.get("MODAL_GPU", "A100")

HRSID_IMAGES = "/data/hrsid/JPEGImages"
HRSID_TRAIN_ANN = "/data/hrsid/annotations/train2017.json"
HRSID_VAL_ANN = "/data/hrsid/annotations/test2017.json"
SSDD_IMAGES = "/data/ssdd/JPEGImages"
SSDD_ANN = "/data/ssdd/ssdd_all.json"

OUTPUT_DIR = "/outputs/yolo_detector"
WORK_DIR = "/outputs/yolo_detector_work"
WANDB_PROJECT = "yolo-detector-ship"
DEFAULT_MODEL = "yolo11l.pt"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch>=2.2",
        "torchvision>=0.17",
        "ultralytics>=8.3.0",
        "pycocotools>=2.0.7",
        "pillow>=9.0",
        "numpy>=1.24",
        "opencv-python-headless>=4.8",
        "wandb>=0.16",
    )
    .add_local_dir(APP_DIR, remote_path="/root/app", ignore=["outputs", "work", "__pycache__", "*.pyc"])
)

app = modal.App("yolo-detector-hrsid-ssdd")

data_vol = modal.Volume.from_name("ship-data", create_if_missing=True)
out_vol = modal.Volume.from_name("yolo-detector-outputs", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb")


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/data": data_vol, "/outputs": out_vol},
    secrets=[wandb_secret],
    timeout=12 * 60 * 60,
)
def train_remote(
    epochs: int = 12,
    batch_size: int = 8,
    imgsz: int = 640,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    workers: int = 4,
    limit: int = 0,
    model: str = DEFAULT_MODEL,
    wandb_run_name: str = None,
    compute_epoch_map: bool = True,
    map_limit: int = 0,
):
    import sys

    sys.path.insert(0, "/root/app")
    from train import run_training

    result = run_training(
        model_name=model,
        train_images=HRSID_IMAGES,
        train_ann=HRSID_TRAIN_ANN,
        val_images=HRSID_IMAGES,
        val_ann=HRSID_VAL_ANN,
        output_dir=OUTPUT_DIR,
        work_dir=WORK_DIR,
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz,
        lr=lr,
        weight_decay=weight_decay,
        workers=workers,
        limit=limit,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=wandb_run_name,
        compute_epoch_map=compute_epoch_map,
        map_limit=map_limit,
    )
    out_vol.commit()
    return result


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/data": data_vol, "/outputs": out_vol},
    secrets=[wandb_secret],
    timeout=3 * 60 * 60,
)
def evaluate_remote(
    dataset: str = "ssdd",
    threshold: float = 0.0,
    limit: int = 0,
    batch_size: int = 8,
    imgsz: int = 640,
    checkpoint: str = None,
    wandb_run_name: str = None,
):
    import sys

    sys.path.insert(0, "/root/app")
    from evaluate import run_eval

    if dataset == "ssdd":
        images, ann, label_to_cat, prefix = SSDD_IMAGES, SSDD_ANN, 0, "ssdd"
    elif dataset == "hrsid-val":
        images, ann, label_to_cat, prefix = HRSID_IMAGES, HRSID_VAL_ANN, 1, "val"
    else:
        raise ValueError("dataset must be 'ssdd' or 'hrsid-val'")

    stats = run_eval(
        model_dir=OUTPUT_DIR,
        images=images,
        ann=ann,
        threshold=threshold,
        label_to_cat=label_to_cat,
        limit=limit,
        batch_size=batch_size,
        imgsz=imgsz,
        checkpoint=checkpoint,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=wandb_run_name,
        metric_prefix=prefix,
    )
    out_vol.commit()
    return stats


@app.local_entrypoint()
def upload():
    """Push HRSID + SSDD from the local repo into the shared `ship-data` volume."""

    hrsid = os.path.join(REPO_ROOT, "HRSID", "HRSID_JPG")
    ssdd_images = os.path.join(REPO_ROOT, "SSDD", "BBox_SSDD", "voc_style", "JPEGImages")
    ssdd_ann = os.path.join(REPO_ROOT, "SSDD", "ssdd_all.json")

    assert os.path.isfile(ssdd_ann), "Run `python dinov3_detr/make_ssdd_coco.py` first."

    with data_vol.batch_upload(force=True) as b:
        b.put_directory(os.path.join(hrsid, "JPEGImages"), "/hrsid/JPEGImages")
        b.put_file(os.path.join(hrsid, "annotations", "train2017.json"), "/hrsid/annotations/train2017.json")
        b.put_file(os.path.join(hrsid, "annotations", "test2017.json"), "/hrsid/annotations/test2017.json")
        b.put_directory(ssdd_images, "/ssdd/JPEGImages")
        b.put_file(ssdd_ann, "/ssdd/ssdd_all.json")
    print("Upload complete.")


@app.local_entrypoint()
def smoke():
    """Run a tiny job to verify imports, data conversion, training, and metrics."""

    run_name = "yolo-detector-smoke"
    train_remote.remote(
        epochs=1,
        batch_size=1,
        workers=0,
        limit=8,
        model=DEFAULT_MODEL,
        wandb_run_name=run_name,
        compute_epoch_map=True,
        map_limit=4,
    )
    stats = evaluate_remote.remote(dataset="hrsid-val", limit=4, batch_size=1, wandb_run_name=run_name)
    if stats:
        print(
            "Smoke HRSID "
            f"AP@[.50:.95] = {stats[0]:.4f}  AP50 = {stats[1]:.4f}  "
            f"APs/APm/APl = {stats[3]:.4f}/{stats[4]:.4f}/{stats[5]:.4f}"
        )


@app.local_entrypoint()
def main(
    epochs: int = 12,
    batch_size: int = 8,
    imgsz: int = 640,
    run_name: str = None,
    model: str = DEFAULT_MODEL,
    limit: int = 0,
    compute_epoch_map: bool = True,
    map_limit: int = 0,
):
    """Train on HRSID, then evaluate on full SSDD with a shared W&B run."""

    import time

    run_name = run_name or f"yolo-detector-{int(time.time())}"
    print(f"W&B run: {run_name} (project {WANDB_PROJECT})")
    train_remote.remote(
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz,
        limit=limit,
        model=model,
        wandb_run_name=run_name,
        compute_epoch_map=compute_epoch_map,
        map_limit=map_limit,
    )
    stats = evaluate_remote.remote(dataset="ssdd", batch_size=batch_size, imgsz=imgsz, wandb_run_name=run_name)
    if stats:
        print(
            "SSDD cross-domain "
            f"AP@[.50:.95] = {stats[0]:.4f}  AP50 = {stats[1]:.4f}  "
            f"APs/APm/APl = {stats[3]:.4f}/{stats[4]:.4f}/{stats[5]:.4f}"
        )
