"""Train YOLOv11l on HRSID and log comparable COCO metrics.

Examples:
    python train.py --epochs 12 --batch-size 8
    python train.py --limit 50 --epochs 1   # quick smoke test

``run_training`` is importable so the Modal app can call it on a GPU.
"""

from __future__ import annotations

import argparse
import os


def pick_device():
    import torch

    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_training(
    model_name: str,
    train_images: str,
    train_ann: str,
    val_images: str,
    val_ann: str,
    output_dir: str,
    work_dir: str,
    epochs: int = 12,
    batch_size: int = 8,
    imgsz: int = 640,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    patience: int = 0,
    workers: int = 4,
    seed: int = 42,
    limit: int = 0,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    compute_epoch_map: bool = True,
    map_limit: int = 0,
):
    # Heavy deps are imported lazily so this module imports cleanly without a
    # full ML stack on the Modal client side.
    from ultralytics import YOLO, settings

    from config import DEFAULTS
    from data import prepare_yolo_dataset
    from evaluate import evaluate_model_on_coco, pick_checkpoint

    os.makedirs(output_dir, exist_ok=True)
    dataset_yaml = prepare_yolo_dataset(
        root=os.path.join(work_dir, "hrsid_yolo"),
        train_images=train_images,
        train_ann=train_ann,
        val_images=val_images,
        val_ann=val_ann,
        limit=limit,
        class_names=DEFAULTS["class_names"],
    )

    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
        if wandb_run_name:
            os.environ.setdefault("WANDB_RUN_ID", wandb_run_name)
            os.environ.setdefault("WANDB_RESUME", "allow")
        try:
            import wandb

            if wandb.run is None:
                wandb.init(project=wandb_project, id=wandb_run_name, name=wandb_run_name, resume="allow")
            wandb.config.update(
                {
                    "model": model_name,
                    "dataset": "HRSID",
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "imgsz": imgsz,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "limit": limit,
                },
                allow_val_change=True,
            )
            settings.update({"wandb": True})
        except Exception as exc:
            print(f"W&B setup failed; continuing without explicit W&B init: {exc}", flush=True)

    device = pick_device()
    print(f"Model: {model_name}", flush=True)
    print(f"Device: {device}", flush=True)
    model = YOLO(model_name)
    results = model.train(
        data=dataset_yaml,
        project=output_dir,
        name="train",
        exist_ok=True,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        lr0=lr,
        weight_decay=weight_decay,
        patience=patience,
        workers=workers,
        seed=seed,
        device=device,
        val=True,
        plots=True,
    )

    save_dir = getattr(results, "save_dir", None) or os.path.join(output_dir, "train")
    with open(os.path.join(output_dir, "latest_train_dir.txt"), "w") as f:
        f.write(str(save_dir))

    if compute_epoch_map:
        ckpt = pick_checkpoint(output_dir)
        print(f"Running HRSID COCO mAP with checkpoint: {ckpt}", flush=True)
        eval_model = YOLO(ckpt)
        evaluate_model_on_coco(
            model=eval_model,
            images=val_images,
            ann=val_ann,
            output_dir=output_dir,
            threshold=0.0,
            label_to_cat=DEFAULTS["coco_category_id"],
            limit=map_limit,
            batch_size=batch_size,
            imgsz=imgsz,
            device=device,
            metric_prefix="val",
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
        )

    print(f"Saved YOLO run under {save_dir}", flush=True)
    return output_dir


def parse_args():
    from config import DEFAULTS

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULTS["model"])
    p.add_argument("--train-images", default=DEFAULTS["train_images"])
    p.add_argument("--train-ann", default=DEFAULTS["train_ann"])
    p.add_argument("--val-images", default=DEFAULTS["val_images"])
    p.add_argument("--val-ann", default=DEFAULTS["val_ann"])
    p.add_argument("--output-dir", default=DEFAULTS["output_dir"])
    p.add_argument("--work-dir", default=DEFAULTS["work_dir"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    p.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--limit", type=int, default=0, help="cap dataset size for debugging")
    p.add_argument("--wandb-project", default=DEFAULTS["wandb_project"])
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-wandb", action="store_true", help="disable W&B logging")
    p.add_argument("--no-epoch-map", action="store_true", help="disable HRSID COCO mAP after training")
    p.add_argument("--map-limit", type=int, default=0, help="cap val images for COCO mAP debugging")
    return p.parse_args()


def main():
    args = parse_args()
    run_training(
        model_name=args.model,
        train_images=args.train_images,
        train_ann=args.train_ann,
        val_images=args.val_images,
        val_ann=args.val_ann,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        limit=args.limit,
        wandb_project=None if args.no_wandb else args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        compute_epoch_map=not args.no_epoch_map,
        map_limit=args.map_limit,
    )


if __name__ == "__main__":
    main()
