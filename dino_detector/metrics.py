"""Metric parsing and W&B logging for official DINO runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


COCO_STAT_NAMES = [
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
]

TRAIN_PROGRESS_RE = re.compile(r"Epoch:\s+\[\s*(?P<epoch>\d+)\]\s+\[\s*(?P<index>\d+)/(?P<total>\d+)\]")
SCALAR_RE = re.compile(r"(?<!\S)(?P<key>[A-Za-z_][\w./-]*):\s+(?P<value>-?(?:\d+\.\d+|\d+)(?:e[+-]?\d+)?)")
IMPORTANT_TRAIN_KEYS = {
    "loss",
    "loss_ce",
    "loss_bbox",
    "loss_giou",
    "class_error",
    "lr",
    "grad_norm",
}


def coco_stats_to_metrics(stats: Iterable[float], prefix: str) -> dict[str, float]:
    return {f"{prefix}/{name}": float(value) for name, value in zip(COCO_STAT_NAMES, stats)}


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    path = Path(path)
    if not path.exists():
        return records
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def official_log_record_to_wandb(record: dict) -> dict[str, float]:
    """Map official DINO log keys to this repo's W&B metric style."""

    metrics: dict[str, float] = {}
    for key, value in record.items():
        if not key.startswith("train_") or not isinstance(value, (int, float)):
            continue
        train_key = key.replace("train_", "", 1)
        if train_key in IMPORTANT_TRAIN_KEYS:
            metrics[f"train_epoch/{train_key}"] = float(value)

    stats = record.get("test_coco_eval_bbox")
    if isinstance(stats, list):
        metrics.update(coco_stats_to_metrics(stats, "val"))
    return metrics


def parse_official_train_progress(line: str, steps_per_epoch: int | None = None) -> tuple[int | None, dict[str, float]]:
    """Parse one official DINO training progress line into W&B metrics.

    Official DINO prints progress lines from ``MetricLogger`` during training,
    for example ``Epoch: [0] [10/911] ... loss: 12.3 ... lr: 0.0001``. Those
    lines are the closest equivalent to Hugging Face Trainer's live step logs.
    """

    match = TRAIN_PROGRESS_RE.search(line)
    if not match:
        return None, {}

    epoch = int(match.group("epoch"))
    index = int(match.group("index"))
    total = int(match.group("total"))
    metrics: dict[str, float] = {
        "train/epoch": float(epoch),
        "train/step_in_epoch": float(index),
        "train/progress": float(index / max(total, 1)),
    }

    for scalar in SCALAR_RE.finditer(line):
        key = scalar.group("key")
        if key not in IMPORTANT_TRAIN_KEYS:
            continue
        value = float(scalar.group("value"))
        metrics[f"train/{key}"] = value
        if key == "lr":
            metrics["train/learning_rate"] = value

    if len(metrics) == 3:
        return None, {}

    epoch_steps = steps_per_epoch or total
    global_step = epoch * epoch_steps + index + 1
    return global_step, metrics


def log_official_training_log(
    log_path: str,
    wandb_project: str | None,
    wandb_run_name: str | None,
    steps_per_epoch: int | None = None,
) -> None:
    """Replay official DINO epoch summaries into W&B after training completes."""

    if not wandb_project:
        return

    import wandb

    if wandb.run is None:
        wandb.init(project=wandb_project, id=wandb_run_name, name=wandb_run_name, resume="allow")
    wandb.define_metric("train_epoch/global_step")
    wandb.define_metric("train_epoch/*", step_metric="train_epoch/global_step")
    wandb.define_metric("val/global_step")
    wandb.define_metric("val/*", step_metric="val/global_step")

    for record in read_jsonl(log_path):
        metrics = official_log_record_to_wandb(record)
        if not metrics:
            continue
        step = record.get("epoch")
        if isinstance(step, int) and steps_per_epoch:
            global_step = (step + 1) * steps_per_epoch
            if any(key.startswith("train_epoch/") for key in metrics):
                metrics["train_epoch/global_step"] = float(global_step)
            if any(key.startswith("val/") for key in metrics):
                metrics["val/global_step"] = float(global_step)
        wandb.log(metrics)
