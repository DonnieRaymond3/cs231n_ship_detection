"""Default paths and hyperparameters for YOLOv11l HRSID -> SSDD runs."""

from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HRSID_JPG = os.path.join(REPO_ROOT, "HRSID", "HRSID_JPG")

DEFAULTS = {
    # Ultralytics YOLO 11 large detector checkpoint.
    "model": "yolo11l.pt",

    # Train on HRSID and monitor on its test split, matching the other baselines.
    "train_images": os.path.join(HRSID_JPG, "JPEGImages"),
    "train_ann": os.path.join(HRSID_JPG, "annotations", "train2017.json"),
    "val_images": os.path.join(HRSID_JPG, "JPEGImages"),
    "val_ann": os.path.join(HRSID_JPG, "annotations", "test2017.json"),

    # Final cross-domain test: full SSDD merged COCO file.
    "test_images": os.path.join(REPO_ROOT, "SSDD", "BBox_SSDD", "voc_style", "JPEGImages"),
    "test_ann": os.path.join(REPO_ROOT, "SSDD", "ssdd_all.json"),

    "output_dir": os.path.join(REPO_ROOT, "yolo_detector", "outputs"),
    "work_dir": os.path.join(REPO_ROOT, "yolo_detector", "work"),

    "wandb_project": "yolo-detector-ship",

    # YOLO trains a single zero-indexed class internally. Predictions are
    # exported with each dataset's original COCO category id for COCOeval.
    "num_classes": 1,
    "class_names": ["ship"],
    "coco_category_id": 1,  # HRSID
    "test_category_id": 0,  # SSDD

    "epochs": 12,
    "batch_size": 8,
    "imgsz": 640,
    "lr": 1e-3,
    "weight_decay": 5e-4,
    "patience": 0,
    "workers": 4,
    "seed": 42,
}
