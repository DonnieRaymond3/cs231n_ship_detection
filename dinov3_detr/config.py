"""Default paths and hyperparameters for the DINOv3 + DETR HRSID baseline.

Paths are resolved relative to the repository root so the scripts work
regardless of the current working directory.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HRSID_JPG = os.path.join(REPO_ROOT, "HRSID", "HRSID_JPG")

DEFAULTS = {
    # Gated DINOv3 checkpoint. Make sure your Modal "huggingface" secret has a
    # token approved for this model.
    "backbone": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
    # Initialize DETR's transformer/query/bbox pieces from a pretrained detector.
    # Backbone/input projection/classifier are intentionally task-specific.
    "pretrained_detr": "facebook/detr-resnet-50",
    # DETR image processor (resize + normalize + COCO annotation formatting).
    "image_processor": "facebook/detr-resnet-50",

    # Train on HRSID (in-domain). HRSID has no official val split, so its
    # test2017 split is used for in-domain monitoring / early stopping.
    "train_images": os.path.join(HRSID_JPG, "JPEGImages"),
    "train_ann": os.path.join(HRSID_JPG, "annotations", "train2017.json"),
    "val_images": os.path.join(HRSID_JPG, "JPEGImages"),
    "val_ann": os.path.join(HRSID_JPG, "annotations", "test2017.json"),

    # Final cross-domain test: the entire SSDD dataset (1,160 images).
    # Build SSDD/ssdd_all.json once with: python make_ssdd_coco.py
    "test_images": os.path.join(REPO_ROOT, "SSDD", "BBox_SSDD", "voc_style", "JPEGImages"),
    "test_ann": os.path.join(REPO_ROOT, "SSDD", "ssdd_all.json"),

    "output_dir": os.path.join(REPO_ROOT, "dinov3_detr", "outputs"),

    # Weights & Biases experiment tracking.
    "wandb_project": "dinov3-detr-ship",

    # Single class: ship.
    "num_labels": 1,
    "id2label": {0: "ship"},
    "label2id": {"ship": 0},
    # HRSID stores ship under category_id == 1; SSDD uses 0. Both map to DETR
    # label 0, and predictions are written back with the test set's own id.
    "coco_category_id": 1,   # HRSID
    "test_category_id": 0,   # SSDD
    "epochs": 12,
    "batch_size": 4,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "warmup_steps": 300,
}
