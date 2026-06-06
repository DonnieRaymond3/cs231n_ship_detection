"""Default paths and hyperparameters for Zhang et al. DINO detector.

The official DINO repo expects COCO's ``train2017`` / ``val2017`` directory
names. The wrappers in this package create lightweight symlink layouts around
the HRSID and SSDD files already stored in this repository or in Modal volumes.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HRSID_JPG = os.path.join(REPO_ROOT, "HRSID", "HRSID_JPG")

DEFAULTS = {
    "dino_repo": os.environ.get("DINO_REPO", "/opt/DINO"),
    "config_file": os.path.join(REPO_ROOT, "dino_detector", "generated", "DINO_4scale_hrsid.py"),
    "work_dir": os.path.join(REPO_ROOT, "dino_detector", "work"),
    "output_dir": os.path.join(REPO_ROOT, "dino_detector", "outputs"),
    "pretrain_model_path": os.path.join(
        REPO_ROOT,
        "dino_detector",
        "pretrained",
        "checkpoint0011_4scale.pth",
    ),

    # HRSID train and in-domain validation.
    "train_images": os.path.join(HRSID_JPG, "JPEGImages"),
    "train_ann": os.path.join(HRSID_JPG, "annotations", "train2017.json"),
    "val_images": os.path.join(HRSID_JPG, "JPEGImages"),
    "val_ann": os.path.join(HRSID_JPG, "annotations", "test2017.json"),

    # Final cross-domain test: the entire SSDD dataset. Build this once with
    # ``python dinov3_detr/make_ssdd_coco.py`` if it does not exist yet.
    "test_images": os.path.join(REPO_ROOT, "SSDD", "BBox_SSDD", "voc_style", "JPEGImages"),
    "test_ann": os.path.join(REPO_ROOT, "SSDD", "ssdd_all.json"),

    # HRSID stores ship as category_id 1; SSDD stores ship as category_id 0.
    # Internally DINO needs num_classes=2 so raw HRSID label 1 is in range.
    "num_classes": 2,
    "dn_labelbook_size": 2,
    "train_category_id": 1,
    "test_category_id": 0,

    "epochs": 12,
    "batch_size": 2,
    "lr": 1e-4,
    "lr_backbone": 1e-5,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "seed": 42,
    "amp": True,

    "wandb_project": "dino-detector-ship",
}
