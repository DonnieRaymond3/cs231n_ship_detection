"""Metric helpers shared by YOLO training and evaluation."""

from __future__ import annotations

from typing import Iterable

COCO_STAT_NAMES = [
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
]


def coco_stats_to_metrics(stats: Iterable[float], prefix: str) -> dict[str, float]:
    return {f"{prefix}/{name}": float(value) for name, value in zip(COCO_STAT_NAMES, stats)}


def log_wandb_metrics(
    metrics: dict[str, float],
    wandb_project: str | None,
    wandb_run_name: str | None,
    step: int | None = None,
) -> None:
    if not wandb_project or not metrics:
        return

    import wandb

    if wandb.run is None:
        wandb.init(project=wandb_project, id=wandb_run_name, name=wandb_run_name, resume="allow")
    wandb.log(metrics, step=step)
    # W&B history receives wandb.log(), but writing summary explicitly makes the
    # final wandb-summary.json include the COCO AP_small/AP_medium/AP_large keys.
    for key, value in metrics.items():
        wandb.run.summary[key] = value
    wandb.run.summary.update()
