"""Run official DINO detector fine-tuning on Modal.

Workflow:
    1. Create W&B secret:
         modal secret create wandb WANDB_API_KEY=xxx
    2. Build the SSDD test file locally if needed:
         python dinov3_detr/make_ssdd_coco.py
    3. Upload datasets:
         modal run dino_detector/modal_app.py::upload
    4. Put the official R50 4-scale checkpoint in the output volume, or pass a
       checkpoint path to train_remote/main.
    5. Train on HRSID and evaluate on SSDD:
         modal run dino_detector/modal_app.py::main --epochs 12 --batch-size 2
"""

from __future__ import annotations

import os

import modal

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))

GPU = os.environ.get("MODAL_GPU", "A100")
DINO_REPO = "/opt/DINO"

HRSID_IMAGES = "/data/hrsid/JPEGImages"
HRSID_TRAIN_ANN = "/data/hrsid/annotations/train2017.json"
HRSID_VAL_ANN = "/data/hrsid/annotations/test2017.json"
SSDD_IMAGES = "/data/ssdd/JPEGImages"
SSDD_ANN = "/data/ssdd/ssdd_all.json"

OUTPUT_DIR = "/outputs/dino_detector"
WORK_DIR = "/outputs/dino_detector_work"
CONFIG_FILE = "/outputs/dino_detector/config/DINO_4scale_hrsid.py"
PRETRAIN_PATH = "/outputs/dino_detector/pretrained/checkpoint0011_4scale.pth"
WANDB_PROJECT = "dino-detector-ship"

upload_image = modal.Image.debian_slim(python_version="3.10")
download_image = upload_image.pip_install("gdown>=5.0")

train_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .entrypoint([])
    .apt_install("git", "build-essential", "gcc", "g++", "ninja-build")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "cython",
        "packaging",
        "wheel",
        "setuptools",
        "pycocotools>=2.0.7",
        "submitit",
        "scipy",
        "termcolor",
        "addict",
        "yapf==0.40.1",
        "timm",
        "numpy",
        "pillow",
        "wandb>=0.16",
        "gdown>=5.0",
        "git+https://github.com/cocodataset/panopticapi.git#egg=panopticapi",
    )
    .run_commands(
        f"git clone --depth 1 https://github.com/IDEA-Research/DINO.git {DINO_REPO}",
    )
    .add_local_dir(APP_DIR, remote_path="/root/app", ignore=["outputs", "work", "__pycache__", "*.pyc"])
)

app = modal.App("dino-detector-hrsid-ssdd")

data_vol = modal.Volume.from_name("ship-data", create_if_missing=True)
out_vol = modal.Volume.from_name("dino-detector-outputs", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb")


def ensure_dino_ops():
    """Build official DINO CUDA ops inside a GPU container.

    The upstream setup.py checks ``torch.cuda.is_available()``, which is false
    during Modal image builds. Compile lazily after the GPU function starts.
    """

    import importlib
    import os
    import subprocess
    import sys

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "DINO CUDA ops require a Modal GPU runtime with torch.cuda available. "
            "Check that the function has a GPU assigned and the CUDA/PyTorch image built correctly."
        )

    try:
        importlib.import_module("MultiScaleDeformableAttention")
        return
    except ModuleNotFoundError:
        pass

    ops_dir = os.path.join(DINO_REPO, "models", "dino", "ops")
    print(f"Compiling DINO CUDA ops on {torch.cuda.get_device_name(0)}", flush=True)
    env = os.environ.copy()
    env.update({"CC": "gcc", "CXX": "g++", "CUDAHOSTCXX": "g++"})
    subprocess.run(
        [sys.executable, "-m", "pip", "install", ".", "--no-build-isolation"],
        cwd=ops_dir,
        env=env,
        check=True,
    )
    importlib.import_module("MultiScaleDeformableAttention")
    with open(os.path.join(ops_dir, ".modal_ops_built"), "w") as f:
        f.write("ok\n")


@app.function(
    image=download_image,
    volumes={"/outputs": out_vol},
    timeout=30 * 60,
)
def download_pretrained_remote(url: str, output_path: str = PRETRAIN_PATH):
    """Download a user-provided official DINO checkpoint URL into the output volume."""

    import os
    import re
    import gdown

    match = re.search(r"/d/([A-Za-z0-9_-]+)", url)
    if match:
        url = f"https://drive.google.com/uc?id={match.group(1)}"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdown.download(url=url, output=output_path, quiet=False)
    if not os.path.isfile(output_path):
        raise FileNotFoundError(f"Download did not produce {output_path}")
    out_vol.commit()
    return output_path


@app.function(
    image=train_image,
    gpu=GPU,
    volumes={"/data": data_vol, "/outputs": out_vol},
    secrets=[wandb_secret],
    timeout=12 * 60 * 60,
)
def train_remote(
    epochs: int = 12,
    batch_size: int = 2,
    lr: float = 1e-4,
    lr_backbone: float = 1e-5,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    limit: int = 0,
    pretrain_model_path: str = PRETRAIN_PATH,
    wandb_run_name: str = None,
    no_pretrain: bool = False,
    output_dir: str = OUTPUT_DIR,
    work_dir: str = WORK_DIR,
    config_file: str = CONFIG_FILE,
):
    import sys

    ensure_dino_ops()
    sys.path.insert(0, "/root/app")
    from train import run_training

    result = run_training(
        dino_repo=DINO_REPO,
        train_images=HRSID_IMAGES,
        train_ann=HRSID_TRAIN_ANN,
        val_images=HRSID_IMAGES,
        val_ann=HRSID_VAL_ANN,
        output_dir=output_dir,
        work_dir=work_dir,
        config_file=config_file,
        pretrain_model_path=None if no_pretrain else pretrain_model_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lr_backbone=lr_backbone,
        weight_decay=weight_decay,
        num_workers=num_workers,
        limit=limit,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=wandb_run_name,
    )
    out_vol.commit()
    return result


@app.function(
    image=train_image,
    gpu=GPU,
    volumes={"/data": data_vol, "/outputs": out_vol},
    secrets=[wandb_secret],
    timeout=3 * 60 * 60,
)
def evaluate_remote(
    dataset: str = "ssdd",
    threshold: float = 0.0,
    limit: int = 0,
    batch_size: int = 4,
    checkpoint: str = None,
    wandb_run_name: str = None,
    model_dir: str = OUTPUT_DIR,
    work_dir: str = WORK_DIR,
    config_file: str = CONFIG_FILE,
):
    import sys

    ensure_dino_ops()
    sys.path.insert(0, "/root/app")
    from evaluate import run_eval

    if dataset == "ssdd":
        images, ann, label_to_cat, prefix = SSDD_IMAGES, SSDD_ANN, 0, "ssdd"
    elif dataset == "hrsid-val":
        images, ann, label_to_cat, prefix = HRSID_IMAGES, HRSID_VAL_ANN, 1, "val"
    else:
        raise ValueError("dataset must be 'ssdd' or 'hrsid-val'")

    stats = run_eval(
        dino_repo=DINO_REPO,
        model_dir=model_dir,
        images=images,
        ann=ann,
        work_dir=work_dir,
        config_file=config_file,
        checkpoint=checkpoint,
        label_to_cat=label_to_cat,
        threshold=threshold,
        limit=limit,
        batch_size=batch_size,
        metric_prefix=prefix,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=wandb_run_name,
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
def download_pretrained(url: str, output_path: str = PRETRAIN_PATH):
    path = download_pretrained_remote.remote(url=url, output_path=output_path)
    print(f"Downloaded checkpoint to {path}")


@app.local_entrypoint()
def smoke():
    """Run a tiny no-pretrain job to verify imports, CUDA ops, data, and metrics."""

    import time

    run_name = f"dino-detector-smoke-{int(time.time())}"
    output_dir = f"{OUTPUT_DIR}/runs/{run_name}"
    work_dir = f"{WORK_DIR}/runs/{run_name}"
    config_file = f"{output_dir}/config/DINO_4scale_hrsid.py"
    train_remote.remote(
        epochs=1,
        batch_size=1,
        num_workers=0,
        limit=8,
        no_pretrain=True,
        wandb_run_name=run_name,
        output_dir=output_dir,
        work_dir=work_dir,
        config_file=config_file,
    )
    stats = evaluate_remote.remote(
        dataset="hrsid-val",
        limit=4,
        batch_size=1,
        wandb_run_name=run_name,
        model_dir=output_dir,
        work_dir=work_dir,
        config_file=config_file,
    )
    if stats:
        print(f"Smoke HRSID AP@[.50:.95] = {stats[0]:.4f}  AP50 = {stats[1]:.4f}")


@app.local_entrypoint()
def main(
    epochs: int = 12,
    batch_size: int = 2,
    run_name: str = None,
    pretrain_model_path: str = PRETRAIN_PATH,
    limit: int = 0,
):
    """Train on HRSID, then evaluate on full SSDD with a shared W&B run."""

    import time

    run_name = run_name or f"dino-detector-{int(time.time())}"
    output_dir = f"{OUTPUT_DIR}/runs/{run_name}"
    work_dir = f"{WORK_DIR}/runs/{run_name}"
    config_file = f"{output_dir}/config/DINO_4scale_hrsid.py"
    print(f"W&B run: {run_name} (project {WANDB_PROJECT})")
    print(f"Output dir: {output_dir}")
    train_remote.remote(
        epochs=epochs,
        batch_size=batch_size,
        limit=limit,
        pretrain_model_path=pretrain_model_path,
        wandb_run_name=run_name,
        output_dir=output_dir,
        work_dir=work_dir,
        config_file=config_file,
    )
    stats = evaluate_remote.remote(
        dataset="ssdd",
        wandb_run_name=run_name,
        model_dir=output_dir,
        work_dir=work_dir,
        config_file=config_file,
    )
    if stats:
        print(f"SSDD cross-domain AP@[.50:.95] = {stats[0]:.4f}  AP50 = {stats[1]:.4f}")
