"""COCO layout helpers for the official IDEA-Research DINO code."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


def _replace_path(target: Path, source: Path) -> None:
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    target.symlink_to(source, target_is_directory=source.is_dir())


def _copy_json(src: Path, dst: Path, limit: int = 0) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not limit:
        shutil.copy2(src, dst)
        return

    with src.open() as f:
        coco = json.load(f)

    keep_images = coco["images"][:limit]
    keep_ids = {img["id"] for img in keep_images}
    coco["images"] = keep_images
    coco["annotations"] = [
        ann for ann in coco.get("annotations", [])
        if ann.get("image_id") in keep_ids
    ]
    with dst.open("w") as f:
        json.dump(coco, f)


def prepare_coco_layout(
    root: str,
    train_images: str,
    train_ann: str,
    val_images: str,
    val_ann: str,
    limit: int = 0,
) -> str:
    """Create the COCO directory names expected by official DINO.

    The image directories are symlinked, while annotations are copied so optional
    smoke-test limits do not mutate the original dataset files.
    """

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "annotations").mkdir(exist_ok=True)

    _replace_path(root_path / "train2017", Path(train_images).resolve())
    _replace_path(root_path / "val2017", Path(val_images).resolve())
    _copy_json(Path(train_ann), root_path / "annotations" / "instances_train2017.json", limit=limit)
    val_limit = max(1, limit // 5) if limit else 0
    _copy_json(Path(val_ann), root_path / "annotations" / "instances_val2017.json", limit=val_limit)
    return str(root_path)


def prepare_eval_layout(root: str, images: str, ann: str, limit: int = 0) -> str:
    """Create a val-only COCO layout for evaluation."""

    return prepare_coco_layout(
        root=root,
        train_images=images,
        train_ann=ann,
        val_images=images,
        val_ann=ann,
        limit=limit,
    )


def require_file(path: str, description: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description} not found: {path}")
    return path
