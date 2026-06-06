"""Evaluate a trained YOLOv11l ship detector with COCO mAP.

By default this evaluates on the full SSDD dataset (cross-domain test) using
SSDD's category_id 0. Override --ann / --images / --label-to-cat for HRSID.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def pick_device():
    import torch

    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_checkpoint(model_dir: str, checkpoint: str | None = None) -> str:
    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    path = Path(model_dir)
    if path.is_file():
        return str(path)

    preferred = [
        path / "train" / "weights" / "best.pt",
        path / "train" / "weights" / "last.pt",
        path / "weights" / "best.pt",
        path / "weights" / "last.pt",
        path / "best.pt",
        path / "last.pt",
    ]
    for candidate in preferred:
        if candidate.is_file():
            return str(candidate)

    matches = list(path.glob("**/weights/best.pt")) + list(path.glob("**/weights/last.pt"))
    if matches:
        return str(max(matches, key=lambda p: p.stat().st_mtime))
    raise FileNotFoundError(f"No YOLO checkpoint found under {model_dir}")


def evaluate_model_on_coco(
    model,
    images: str,
    ann: str,
    output_dir: str,
    threshold: float = 0.0,
    label_to_cat: int = 0,
    limit: int = 0,
    batch_size: int = 8,
    imgsz: int = 640,
    device=None,
    metric_prefix: str = "val",
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    wandb_step: int | None = None,
):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    from metrics import coco_stats_to_metrics, log_wandb_metrics

    device = pick_device() if device is None else device
    with open(ann) as f:
        coco = json.load(f)

    img_list = coco.get("images", [])
    if limit:
        img_list = img_list[:limit]

    results = []
    conf = max(float(threshold), 0.0)
    for start in range(0, len(img_list), batch_size):
        batch_infos = img_list[start:start + batch_size]
        batch_paths = [os.path.join(images, info["file_name"]) for info in batch_infos]
        predictions = model.predict(
            source=batch_paths,
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
            classes=[0],
        )
        for info, pred in zip(batch_infos, predictions):
            boxes = pred.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().tolist()
            scores = boxes.conf.cpu().tolist()
            for box, score in zip(xyxy, scores):
                x1, y1, x2, y2 = [float(v) for v in box]
                results.append(
                    {
                        "image_id": int(info["id"]),
                        "category_id": int(label_to_cat),
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score),
                    }
                )

    os.makedirs(output_dir, exist_ok=True)
    pred_file = os.path.join(output_dir, f"{metric_prefix}_predictions.json")
    with open(pred_file, "w") as f:
        json.dump(results, f)

    if not results:
        print(f"No detections produced for {metric_prefix} evaluation.", flush=True)
        return None

    coco_gt = COCO(ann)
    coco_dt = coco_gt.loadRes(pred_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if limit:
        coco_eval.params.imgIds = [int(i["id"]) for i in img_list]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats.tolist()

    metrics = coco_stats_to_metrics(stats, metric_prefix)
    log_wandb_metrics(metrics, wandb_project, wandb_run_name, step=wandb_step)
    return stats


def run_eval(
    model_dir: str,
    images: str,
    ann: str,
    threshold: float = 0.0,
    label_to_cat: int = 0,
    limit: int = 0,
    batch_size: int = 8,
    imgsz: int = 640,
    checkpoint: str | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    metric_prefix: str = "ssdd",
):
    from ultralytics import YOLO

    ckpt = pick_checkpoint(model_dir, checkpoint=checkpoint)
    print(f"Evaluating checkpoint: {ckpt}", flush=True)
    model = YOLO(ckpt)
    return evaluate_model_on_coco(
        model=model,
        images=images,
        ann=ann,
        output_dir=model_dir,
        threshold=threshold,
        label_to_cat=label_to_cat,
        limit=limit,
        batch_size=batch_size,
        imgsz=imgsz,
        device=pick_device(),
        metric_prefix=metric_prefix,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )


def parse_args():
    from config import DEFAULTS

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default=DEFAULTS["output_dir"])
    p.add_argument("--images", default=DEFAULTS["test_images"])
    p.add_argument("--ann", default=DEFAULTS["test_ann"])
    p.add_argument("--threshold", type=float, default=0.0,
                   help="keep detections above this score (0.0 for full mAP)")
    p.add_argument("--label-to-cat", type=int, default=DEFAULTS["test_category_id"],
                   help="COCO category_id assigned to YOLO class 0 (SSDD=0, HRSID=1)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--wandb-project", default=DEFAULTS["wandb_project"])
    p.add_argument("--wandb-run-name", default=None,
                   help="attach metrics to this W&B run id (same as training run)")
    p.add_argument("--no-wandb", action="store_true", help="disable W&B logging")
    p.add_argument("--metric-prefix", default="ssdd")
    return p.parse_args()


def main():
    args = parse_args()
    run_eval(
        model_dir=args.model_dir,
        images=args.images,
        ann=args.ann,
        threshold=args.threshold,
        label_to_cat=args.label_to_cat,
        limit=args.limit,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        checkpoint=args.checkpoint,
        wandb_project=None if args.no_wandb else args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        metric_prefix=args.metric_prefix,
    )


if __name__ == "__main__":
    main()
