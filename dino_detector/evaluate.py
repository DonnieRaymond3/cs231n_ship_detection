"""Evaluate official DINO checkpoints with pycocotools COCO metrics."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys

from config import DEFAULTS
from data import prepare_eval_layout, require_file
from gen_config import write_config
from metrics import coco_stats_to_metrics


def pick_device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_checkpoint(model_dir: str, checkpoint: str | None = None) -> str:
    if checkpoint:
        return require_file(checkpoint, "DINO checkpoint")
    candidates = [
        os.path.join(model_dir, "checkpoint_best_regular.pth"),
        os.path.join(model_dir, "checkpoint.pth"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No DINO checkpoint found in {model_dir}")


def load_model(dino_repo: str, config_file: str, checkpoint: str, device: str):
    import torch

    sys.path.insert(0, dino_repo)
    from main import build_model_main
    from util.slconfig import SLConfig
    from util.utils import clean_state_dict

    args = SLConfig.fromfile(config_file)
    args.device = device
    args.dataset_file = "coco"
    args.fix_size = False
    args.remove_difficult = False
    args.num_workers = 2
    args.output_dir = os.path.dirname(checkpoint)
    args.amp = False
    args.debug = False

    model, criterion, postprocessors = build_model_main(args)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(clean_state_dict(state["model"]), strict=False)
    model.to(device).eval()
    criterion.to(device).eval()
    return args, model, criterion, postprocessors


def predict_coco_results(
    dino_repo: str,
    config_file: str,
    checkpoint: str,
    coco_path: str,
    label_to_cat: int | None,
    threshold: float,
    batch_size: int,
    num_workers: int,
    device: str,
) -> list[dict]:
    import torch
    from torch.utils.data import DataLoader, SequentialSampler

    sys.path.insert(0, dino_repo)
    from datasets import build_dataset
    import util.misc as utils

    args, model, _criterion, postprocessors = load_model(dino_repo, config_file, checkpoint, device)
    args.coco_path = coco_path
    args.num_workers = num_workers
    dataset = build_dataset(image_set="val", args=args)
    sampler = SequentialSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=num_workers,
    )

    results = []
    with torch.no_grad():
        for samples, targets in loader:
            samples = samples.to(device)
            outputs = model(samples)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
            processed = postprocessors["bbox"](outputs, orig_target_sizes)

            for target, pred in zip(targets, processed):
                image_id = int(target["image_id"].item())
                for score, label, box in zip(pred["scores"], pred["labels"], pred["boxes"]):
                    score_value = float(score)
                    if score_value < threshold:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                    results.append(
                        {
                            "image_id": image_id,
                            "category_id": int(label_to_cat if label_to_cat is not None else label.item()),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": score_value,
                        }
                    )
    return results


def print_core_coco_stats(stats: list[float], prefix: str) -> None:
    names = ("AP", "AP50", "AP75", "AR100")
    values = (stats[0], stats[1], stats[2], stats[8])
    summary = "  ".join(f"{name}={value:.4f}" for name, value in zip(names, values))
    print(f"{prefix} COCOeval: {summary}", flush=True)


def evaluate_predictions(
    ann: str,
    predictions: list[dict],
    output_dir: str,
    metric_prefix: str,
    limit_ids: list[int] | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    os.makedirs(output_dir, exist_ok=True)
    pred_file = os.path.join(output_dir, f"{metric_prefix}_predictions.json")
    with open(pred_file, "w") as f:
        json.dump(predictions, f)

    coco_gt = COCO(ann)
    coco_dt = coco_gt.loadRes(pred_file) if predictions else coco_gt.loadRes([])
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if limit_ids:
        coco_eval.params.imgIds = [int(i) for i in limit_ids]
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    stats = coco_eval.stats.tolist()
    print_core_coco_stats(stats, metric_prefix)

    if wandb_project:
        import wandb

        if wandb.run is None:
            wandb.init(project=wandb_project, id=wandb_run_name, name=wandb_run_name, resume="allow")
        wandb.log(coco_stats_to_metrics(stats, metric_prefix))

    return stats


def run_eval(
    dino_repo: str,
    model_dir: str,
    images: str,
    ann: str,
    work_dir: str,
    config_file: str,
    checkpoint: str | None = None,
    label_to_cat: int | None = DEFAULTS["test_category_id"],
    threshold: float = 0.0,
    limit: int = 0,
    batch_size: int = 4,
    num_workers: int = DEFAULTS["num_workers"],
    metric_prefix: str = "ssdd",
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> list[float]:
    require_file(ann, "COCO annotations")
    if not os.path.isfile(config_file):
        write_config(config_file, dino_repo=dino_repo)
    checkpoint = pick_checkpoint(model_dir, checkpoint)
    coco_path = prepare_eval_layout(
        root=os.path.join(work_dir, f"{metric_prefix}_coco"),
        images=images,
        ann=ann,
        limit=limit,
    )
    eval_ann = os.path.join(coco_path, "annotations", "instances_val2017.json")

    with open(eval_ann) as f:
        limit_ids = [img["id"] for img in json.load(f)["images"]] if limit else None

    device = pick_device()
    predictions = predict_coco_results(
        dino_repo=dino_repo,
        config_file=config_file,
        checkpoint=checkpoint,
        coco_path=coco_path,
        label_to_cat=label_to_cat,
        threshold=threshold,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    return evaluate_predictions(
        ann=eval_ann,
        predictions=predictions,
        output_dir=model_dir,
        metric_prefix=metric_prefix,
        limit_ids=limit_ids,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dino-repo", default=DEFAULTS["dino_repo"])
    p.add_argument("--model-dir", default=DEFAULTS["output_dir"])
    p.add_argument("--images", default=DEFAULTS["test_images"])
    p.add_argument("--ann", default=DEFAULTS["test_ann"])
    p.add_argument("--work-dir", default=DEFAULTS["work_dir"])
    p.add_argument("--config-file", default=DEFAULTS["config_file"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--label-to-cat", type=int, default=DEFAULTS["test_category_id"])
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--metric-prefix", default="ssdd")
    p.add_argument("--wandb-project", default=DEFAULTS["wandb_project"])
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run_eval(
        dino_repo=args.dino_repo,
        model_dir=args.model_dir,
        images=args.images,
        ann=args.ann,
        work_dir=args.work_dir,
        config_file=args.config_file,
        checkpoint=args.checkpoint,
        label_to_cat=args.label_to_cat,
        threshold=args.threshold,
        limit=args.limit,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        metric_prefix=args.metric_prefix,
        wandb_project=None if args.no_wandb else args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()
