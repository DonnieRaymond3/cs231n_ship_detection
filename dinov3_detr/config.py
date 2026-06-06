import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HRSID_JPG = os.path.join(REPO_ROOT, "HRSID", "HRSID_JPG")

DEFAULTS = {

    "backbone": "facebook/dinov3-convnext-large-pretrain-lvd1689m",

    "pretrained_detr": "facebook/detr-resnet-50",

    "image_processor": "facebook/detr-resnet-50",

    "train_images": os.path.join(HRSID_JPG, "JPEGImages"),
    "train_ann": os.path.join(HRSID_JPG, "annotations", "train2017.json"),
    "val_images": os.path.join(HRSID_JPG, "JPEGImages"),
    "val_ann": os.path.join(HRSID_JPG, "annotations", "test2017.json"),

    "test_images": os.path.join(REPO_ROOT, "SSDD", "BBox_SSDD", "voc_style", "JPEGImages"),
    "test_ann": os.path.join(REPO_ROOT, "SSDD", "ssdd_all.json"),

    "output_dir": os.path.join(REPO_ROOT, "dinov3_detr", "outputs"),

    "wandb_project": "dinov3-detr-ship",

    "num_labels": 1,
    "id2label": {0: "ship"},
    "label2id": {"ship": 0},

    "coco_category_id": 1,          
    "test_category_id": 0,         
    "epochs": 12,
    "batch_size": 4,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "warmup_steps": 300,
}
