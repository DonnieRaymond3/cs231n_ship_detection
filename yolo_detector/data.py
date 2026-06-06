"""Dataset helpers for training Ultralytics YOLO on COCO-format ship data."""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


def require_file(path: str, description: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def require_dir(path: str, description: str) -> str:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _reset_dir(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _safe_symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _yolo_line(bbox: list[float], width: float, height: float) -> str | None:
    x, y, w, h = [float(v) for v in bbox]
    if width <= 0 or height <= 0 or w <= 0 or h <= 0:
        return None

    x1 = _clip(x, 0.0, width)
    y1 = _clip(y, 0.0, height)
    x2 = _clip(x + w, 0.0, width)
    y2 = _clip(y + h, 0.0, height)
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    xc = (x1 + bw / 2.0) / width
    yc = (y1 + bh / 2.0) / height
    return f"0 {xc:.6f} {yc:.6f} {bw / width:.6f} {bh / height:.6f}"


def _prepare_split(
    root: Path,
    split: str,
    images_dir: str,
    ann_file: str,
    limit: int = 0,
) -> int:
    require_dir(images_dir, f"{split} image directory")
    require_file(ann_file, f"{split} COCO annotation file")

    with open(ann_file) as f:
        coco = json.load(f)

    image_infos = coco.get("images", [])
    if limit:
        image_infos = image_infos[:limit]
    image_ids = {info["id"] for info in image_infos}

    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("image_id") in image_ids and not ann.get("iscrowd", 0):
            anns_by_image[int(ann["image_id"])].append(ann)

    split_image_dir = root / "images" / split
    split_label_dir = root / "labels" / split
    _reset_dir(split_image_dir)
    _reset_dir(split_label_dir)

    used_names: set[str] = set()
    for info in image_infos:
        source = Path(images_dir) / info["file_name"]
        if not source.is_file():
            raise FileNotFoundError(f"Image listed in {ann_file} not found: {source}")

        target_name = Path(info["file_name"]).name
        if target_name in used_names:
            target_name = f"{info['id']}_{target_name}"
        used_names.add(target_name)

        _safe_symlink(source.resolve(), split_image_dir / target_name)

        width = float(info["width"])
        height = float(info["height"])
        lines = []
        for ann in anns_by_image.get(int(info["id"]), []):
            line = _yolo_line(ann.get("bbox", []), width, height)
            if line is not None:
                lines.append(line)
        (split_label_dir / f"{Path(target_name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    return len(image_infos)


def prepare_yolo_dataset(
    root: str,
    train_images: str,
    train_ann: str,
    val_images: str,
    val_ann: str,
    limit: int = 0,
    class_names: list[str] | None = None,
) -> str:
    """Create an Ultralytics dataset directory and return its dataset YAML path."""

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    names = class_names or ["ship"]

    train_count = _prepare_split(root_path, "train", train_images, train_ann, limit=limit)
    val_limit = max(1, limit // 5) if limit else 0
    val_count = _prepare_split(root_path, "val", val_images, val_ann, limit=val_limit)

    yaml_path = root_path / "dataset.yaml"
    names_yaml = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(names))
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root_path}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(names)}",
                "names:",
                names_yaml,
                "",
            ]
        )
    )
    print(f"Prepared YOLO dataset at {root_path} ({train_count} train, {val_count} val images)", flush=True)
    return str(yaml_path)
