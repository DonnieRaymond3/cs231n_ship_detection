"""Fine-tune official IDEA-Research DINO on HRSID."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

from config import DEFAULTS
from data import prepare_coco_layout, require_file
from gen_config import write_config
from metrics import log_official_training_log, parse_official_train_progress


def require_dino_repo(path: str) -> str:
    main_py = os.path.join(path, "main.py")
    if not os.path.isfile(main_py):
        raise FileNotFoundError(
            f"Official DINO repo not found at {path}. Expected {main_py}. "
            "In Modal this is cloned into /opt/DINO."
        )
    return path


def count_coco_images(ann_path: str) -> int:
    with open(ann_path) as f:
        return len(json.load(f).get("images", []))


def init_wandb_run(
    wandb_project: str | None,
    wandb_run_name: str | None,
    config: dict,
):
    if not wandb_project:
        return None

    import wandb

    if wandb.run is None:
        wandb.init(project=wandb_project, id=wandb_run_name, name=wandb_run_name, resume="allow")
    wandb.define_metric("train/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")
    wandb.define_metric("train_epoch/global_step")
    wandb.define_metric("train_epoch/*", step_metric="train_epoch/global_step")
    wandb.define_metric("val/global_step")
    wandb.define_metric("val/*", step_metric="val/global_step")
    wandb.config.update(config, allow_val_change=True)
    return wandb


def run_streaming_subprocess(
    cmd: list[str],
    cwd: str,
    env: dict[str, str],
    steps_per_epoch: int,
    wandb_module=None,
) -> None:
    """Run official DINO while streaming stdout and live W&B train metrics."""

    coco_summary: dict[str, float] = {}
    test_progress_re = re.compile(r"Test:\s+\[\s*(?P<index>\d+)/(?P<total>\d+)\]")
    scalar_re = re.compile(r"(?<!\S)(?P<key>[A-Za-z_][\w./-]*):\s*(?P<value>-?(?:\d+\.\d+|\d+)(?:e[+-]?\d+)?)")

    def compact_progress_line(line: str, step: int | None, metrics: dict[str, float]) -> str:
        if not metrics:
            return line

        epoch = int(metrics["train/epoch"])
        step_in_epoch = int(metrics["train/step_in_epoch"])
        parts = [f"Epoch: [{epoch}] step {step_in_epoch}"]
        if step is not None:
            parts.append(f"global_step={step}")
        for key in ("loss", "loss_ce", "loss_bbox", "loss_giou", "class_error", "grad_norm", "lr"):
            value = metrics.get(f"train/{key}")
            if value is None:
                continue
            if key == "lr":
                parts.append(f"{key}={value:.2e}")
            else:
                parts.append(f"{key}={value:.4f}")
        return "  ".join(parts) + "\n"

    def compact_eval_progress_line(line: str) -> str | None:
        stripped = line.strip()
        if not (stripped.startswith("Test:") or stripped.startswith("Averaged stats:")):
            return None

        values = {match.group("key"): float(match.group("value")) for match in scalar_re.finditer(stripped)}
        parts: list[str] = []
        match = test_progress_re.search(stripped)
        if match:
            parts.append(f"Test: [{int(match.group('index'))}/{int(match.group('total'))}]")
        else:
            parts.append("Test averaged:")

        for key in ("loss", "loss_ce", "loss_bbox", "loss_giou", "class_error"):
            value = values.get(key)
            if value is not None:
                parts.append(f"{key}={value:.4f}")

        return "  ".join(parts) + "\n"

    def compact_coco_summary_line(line: str) -> str | None:
        stripped = line.strip()
        if not (stripped.startswith("Average Precision") or stripped.startswith("Average Recall")):
            return None

        try:
            value = float(stripped.rsplit("=", 1)[1].strip())
        except (IndexError, ValueError):
            return ""

        key = None
        if stripped.startswith("Average Precision"):
            if "IoU=0.50:0.95" in stripped and "area=   all" in stripped and "maxDets=100" in stripped:
                key = "AP"
            elif "IoU=0.50" in stripped and "area=   all" in stripped and "maxDets=100" in stripped:
                key = "AP50"
            elif "IoU=0.75" in stripped and "area=   all" in stripped and "maxDets=100" in stripped:
                key = "AP75"
        elif "IoU=0.50:0.95" in stripped and "area=   all" in stripped and "maxDets=100" in stripped:
            key = "AR100"

        if key:
            coco_summary[key] = value
        if key == "AR100":
            parts = [f"{name}={coco_summary[name]:.4f}" for name in ("AP", "AP50", "AP75", "AR100") if name in coco_summary]
            coco_summary.clear()
            return "COCOeval: " + "  ".join(parts) + "\n"
        return ""

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        step, metrics = parse_official_train_progress(line, steps_per_epoch=steps_per_epoch)
        coco_line = compact_coco_summary_line(line)
        if coco_line is not None:
            if coco_line:
                print(coco_line, end="", flush=True)
            eval_line = None
        else:
            eval_line = compact_eval_progress_line(line)

        if coco_line is None and eval_line is not None:
            print(eval_line, end="", flush=True)
        elif coco_line is None and eval_line is None:
            print(compact_progress_line(line, step, metrics), end="", flush=True)
        if wandb_module is None:
            continue
        if metrics:
            if step is not None:
                metrics["train/global_step"] = float(step)
            wandb_module.log(metrics)

    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, cmd)


def run_training(
    dino_repo: str,
    train_images: str,
    train_ann: str,
    val_images: str,
    val_ann: str,
    output_dir: str,
    work_dir: str,
    config_file: str,
    pretrain_model_path: str | None,
    epochs: int = DEFAULTS["epochs"],
    batch_size: int = DEFAULTS["batch_size"],
    lr: float = DEFAULTS["lr"],
    lr_backbone: float = DEFAULTS["lr_backbone"],
    weight_decay: float = DEFAULTS["weight_decay"],
    num_workers: int = DEFAULTS["num_workers"],
    seed: int = DEFAULTS["seed"],
    amp: bool = DEFAULTS["amp"],
    limit: int = 0,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> str:
    dino_repo = require_dino_repo(dino_repo)
    require_file(train_ann, "HRSID train annotations")
    require_file(val_ann, "HRSID validation annotations")
    if pretrain_model_path:
        require_file(pretrain_model_path, "DINO COCO pretrained checkpoint")

    os.makedirs(output_dir, exist_ok=True)
    coco_root = prepare_coco_layout(
        root=os.path.join(work_dir, "hrsid_coco"),
        train_images=train_images,
        train_ann=train_ann,
        val_images=val_images,
        val_ann=val_ann,
        limit=limit,
    )
    train_ann_for_count = os.path.join(coco_root, "annotations", "instances_train2017.json")
    train_image_count = count_coco_images(train_ann_for_count)
    steps_per_epoch = max(1, (train_image_count + batch_size - 1) // batch_size)
    config_file = write_config(
        output=config_file,
        dino_repo=dino_repo,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lr_backbone=lr_backbone,
        weight_decay=weight_decay,
    )

    cmd = [
        sys.executable,
        "-u",
        os.path.join(dino_repo, "main.py"),
        "--output_dir", output_dir,
        "--config_file", config_file,
        "--coco_path", coco_root,
        "--dataset_file", "coco",
        "--num_workers", str(num_workers),
        "--seed", str(seed),
        "--finetune_ignore", "label_enc.weight", "class_embed",
        "--options",
        "dn_scalar=100",
        "embed_init_tgt=True",
        "dn_label_coef=1.0",
        "dn_bbox_coef=1.0",
        "use_ema=False",
        "dn_box_noise_scale=1.0",
    ]
    if pretrain_model_path:
        cmd.extend(["--pretrain_model_path", pretrain_model_path])
    if amp:
        cmd.append("--amp")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if wandb_project:
        env["WANDB_PROJECT"] = wandb_project
        if wandb_run_name:
            env["WANDB_RUN_ID"] = wandb_run_name
            env["WANDB_RESUME"] = "allow"

    wandb_module = init_wandb_run(
        wandb_project,
        wandb_run_name,
        {
            "model": "IDEA-Research/DINO",
            "backbone": "resnet50",
            "config_file": config_file,
            "pretrain_model_path": pretrain_model_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "lr_backbone": lr_backbone,
            "weight_decay": weight_decay,
            "num_workers": num_workers,
            "seed": seed,
            "amp": amp,
            "train_images": train_image_count,
            "steps_per_epoch": steps_per_epoch,
        },
    )

    print("Running official DINO training:")
    print(" ".join(cmd), flush=True)
    run_streaming_subprocess(
        cmd,
        cwd=dino_repo,
        env=env,
        steps_per_epoch=steps_per_epoch,
        wandb_module=wandb_module,
    )

    log_official_training_log(
        os.path.join(output_dir, "log.txt"),
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        steps_per_epoch=steps_per_epoch,
    )
    return output_dir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dino-repo", default=DEFAULTS["dino_repo"])
    p.add_argument("--train-images", default=DEFAULTS["train_images"])
    p.add_argument("--train-ann", default=DEFAULTS["train_ann"])
    p.add_argument("--val-images", default=DEFAULTS["val_images"])
    p.add_argument("--val-ann", default=DEFAULTS["val_ann"])
    p.add_argument("--output-dir", default=DEFAULTS["output_dir"])
    p.add_argument("--work-dir", default=DEFAULTS["work_dir"])
    p.add_argument("--config-file", default=DEFAULTS["config_file"])
    p.add_argument("--pretrain-model-path", default=DEFAULTS["pretrain_model_path"])
    p.add_argument("--no-pretrain", action="store_true", help="train without COCO pretrained DINO weights")
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--lr-backbone", type=float, default=DEFAULTS["lr_backbone"])
    p.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="limit images for smoke tests")
    p.add_argument("--wandb-project", default=DEFAULTS["wandb_project"])
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run_training(
        dino_repo=args.dino_repo,
        train_images=args.train_images,
        train_ann=args.train_ann,
        val_images=args.val_images,
        val_ann=args.val_ann,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        config_file=args.config_file,
        pretrain_model_path=None if args.no_pretrain else args.pretrain_model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=not args.no_amp,
        limit=args.limit,
        wandb_project=None if args.no_wandb else args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()
