import os

import modal

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))

GPU = os.environ.get("MODAL_GPU", "A100")                             
DEFAULT_BACKBONE = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
DEFAULT_PRETRAINED_DETR = "facebook/detr-resnet-50"

HRSID_IMAGES = "/data/hrsid/JPEGImages"
HRSID_TRAIN_ANN = "/data/hrsid/annotations/train2017.json"
HRSID_VAL_ANN = "/data/hrsid/annotations/test2017.json"
SSDD_IMAGES = "/data/ssdd/JPEGImages"
SSDD_ANN = "/data/ssdd/ssdd_all.json"
OUTPUT_DIR = "/outputs/dinov3_detr"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "torchvision>=0.17",
        "transformers>=4.56.0",
        "accelerate>=0.30",
        "pycocotools>=2.0.7",
        "pillow>=9.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "timm>=1.0",
        "tqdm>=4.65",
        "wandb>=0.16",
    )

    .add_local_dir(APP_DIR, remote_path="/root/app", ignore=["outputs", "__pycache__", "*.pyc"])
)

app = modal.App("dinov3-detr-hrsid-ssdd")

data_vol = modal.Volume.from_name("ship-data", create_if_missing=True)
out_vol = modal.Volume.from_name("dinov3-detr-outputs", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")                                                      
wandb_secret = modal.Secret.from_name("wandb")                              

WANDB_PROJECT = "dinov3-detr-ship"

@app.function(
    image=image,
    gpu=GPU,
    volumes={"/data": data_vol, "/outputs": out_vol},
    secrets=[hf_secret, wandb_secret],
    timeout=6 * 60 * 60,
)
def train_remote(epochs: int = 12, batch_size: int = 8, lr: float = 1e-4,
                 backbone: str = DEFAULT_BACKBONE, limit: int = 0,
                 wandb_run_name: str = None, compute_epoch_map: bool = True,
                 map_limit: int = 0, pretrained_detr: str = DEFAULT_PRETRAINED_DETR):
    import sys

    sys.path.insert(0, "/root/app")
    from train import run_training

    run_training(
        backbone=backbone,
        pretrained_detr=pretrained_detr or None,
        image_processor="facebook/detr-resnet-50",
        train_images=HRSID_IMAGES,
        train_ann=HRSID_TRAIN_ANN,
        val_images=HRSID_IMAGES,
        val_ann=HRSID_VAL_ANN,
        output_dir=OUTPUT_DIR,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        warmup_steps=300,
        freeze_backbone=True,
        limit=limit,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=wandb_run_name,
        compute_epoch_map=compute_epoch_map,
        map_limit=map_limit,
    )
    out_vol.commit()

@app.function(
    image=image,
    gpu=GPU,
    volumes={"/data": data_vol, "/outputs": out_vol},
    secrets=[hf_secret, wandb_secret],
    timeout=2 * 60 * 60,
)
def evaluate_remote(threshold: float = 0.0, limit: int = 0, wandb_run_name: str = None,
                    backbone: str = DEFAULT_BACKBONE,
                    pretrained_detr: str = DEFAULT_PRETRAINED_DETR):
    import sys

    sys.path.insert(0, "/root/app")
    from evaluate import run_eval

    return run_eval(
        model_dir=OUTPUT_DIR,
        images=SSDD_IMAGES,
        ann=SSDD_ANN,
        threshold=threshold,
        label_to_cat=0,
        limit=limit,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=wandb_run_name,
        metric_prefix="ssdd",
            backbone=backbone,
            pretrained_detr=pretrained_detr or None,
    )

@app.function(
    image=image,
    gpu=GPU,
    volumes={"/data": data_vol, "/outputs": out_vol},
    secrets=[hf_secret, wandb_secret],
    timeout=30 * 60,
)
def visualize_remote(
    dataset: str = "ssdd",
    index: int = 0,
    image_id: int = None,
    file_name: str = None,
    threshold: float = 0.3,
    max_preds: int = 50,
    output_name: str = "prediction_vs_gt.png",
):
    import os
    import sys

    sys.path.insert(0, "/root/app")
    from visualize import visualize_prediction

    if dataset == "ssdd":
        images, ann = SSDD_IMAGES, SSDD_ANN
    elif dataset == "hrsid-val":
        images, ann = HRSID_IMAGES, HRSID_VAL_ANN
    else:
        raise ValueError("dataset must be 'ssdd' or 'hrsid-val'")

    output = os.path.join("/outputs", "visualizations", output_name)
    result = visualize_prediction(
        model_dir=OUTPUT_DIR,
        images_dir=images,
        ann_file=ann,
        output=output,
        image_id=image_id,
        file_name=file_name,
        index=index,
        threshold=threshold,
        max_preds=max_preds,
        backbone=DEFAULT_BACKBONE,
        pretrained_detr=DEFAULT_PRETRAINED_DETR,
    )
    out_vol.commit()
    return result

@app.local_entrypoint()
def upload():
    hrsid = os.path.join(REPO_ROOT, "HRSID", "HRSID_JPG")
    ssdd_images = os.path.join(REPO_ROOT, "SSDD", "BBox_SSDD", "voc_style", "JPEGImages")
    ssdd_ann = os.path.join(REPO_ROOT, "SSDD", "ssdd_all.json")

    assert os.path.isfile(ssdd_ann), "Run `python make_ssdd_coco.py` first."

    with data_vol.batch_upload(force=True) as b:
        b.put_directory(os.path.join(hrsid, "JPEGImages"), "/hrsid/JPEGImages")
        b.put_file(os.path.join(hrsid, "annotations", "train2017.json"),
                   "/hrsid/annotations/train2017.json")
        b.put_file(os.path.join(hrsid, "annotations", "test2017.json"),
                   "/hrsid/annotations/test2017.json")
        b.put_directory(ssdd_images, "/ssdd/JPEGImages")
        b.put_file(ssdd_ann, "/ssdd/ssdd_all.json")
    print("Upload complete.")

@app.local_entrypoint()
def main(epochs: int = 12, batch_size: int = 8, run_name: str = None,
         compute_epoch_map: bool = True, map_limit: int = 0,
         backbone: str = DEFAULT_BACKBONE,
         pretrained_detr: str = DEFAULT_PRETRAINED_DETR):
    import time

    run_name = run_name or f"dinov3-detr-{int(time.time())}"
    print(f"W&B run: {run_name} (project {WANDB_PROJECT})")

    train_remote.remote(
        epochs=epochs,
        batch_size=batch_size,
            backbone=backbone,
            pretrained_detr=pretrained_detr,
        wandb_run_name=run_name,
        compute_epoch_map=compute_epoch_map,
        map_limit=map_limit,
    )
    stats = evaluate_remote.remote(
        wandb_run_name=run_name,
        backbone=backbone,
        pretrained_detr=pretrained_detr,
    )
    if stats:
        print(f"SSDD cross-domain AP@[.50:.95] = {stats[0]:.4f}  AP50 = {stats[1]:.4f}")
