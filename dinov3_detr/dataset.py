"""HRSID (COCO format) dataset and collate function for the DETR image processor.

Each dataset item is a dict::

    {"image": PIL.Image (RGB), "objects": {...}, "image_id": int}

The COCO ``category_id`` (ship == 1 in HRSID) is remapped to a 0-indexed
label, which is what ``DetrForObjectDetection`` expects.
"""

import json
import os
from typing import Dict, Optional

from PIL import Image
from torch.utils.data import Dataset


class HRSIDDetectionDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        ann_file: str,
        cat_id_to_label: Optional[Dict[int, int]] = None,
    ):
        self.images_dir = images_dir
        with open(ann_file, "r") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.image_ids = list(self.images.keys())

        if cat_id_to_label is None:
            cat_ids = sorted(c["id"] for c in coco["categories"])
            cat_id_to_label = {cid: i for i, cid in enumerate(cat_ids)}
        self.cat_id_to_label = cat_id_to_label

        self.anns_by_image: Dict[int, list] = {img_id: [] for img_id in self.image_ids}
        for ann in coco["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict:
        image_id = self.image_ids[idx]
        info = self.images[image_id]
        path = os.path.join(self.images_dir, info["file_name"])
        image = Image.open(path).convert("RGB")

        bboxes, labels, areas, ids, iscrowd = [], [], [], [], []
        for a in self.anns_by_image.get(image_id, []):
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            bboxes.append([float(x), float(y), float(w), float(h)])
            labels.append(int(self.cat_id_to_label[a["category_id"]]))
            areas.append(float(a.get("area", w * h)))
            ids.append(int(a["id"]))
            iscrowd.append(int(a.get("iscrowd", 0)))

        objects = {
            "id": ids,
            "bbox": bboxes,
            "category_id": labels,
            "area": areas,
            "iscrowd": iscrowd,
        }
        return {"image": image, "objects": objects, "image_id": int(image_id)}


def format_annotations(objects: Dict, image_id: int) -> Dict:
    """Convert one item's objects into the COCO dict the processor consumes."""
    anns = []
    for i in range(len(objects["bbox"])):
        x, y, w, h = objects["bbox"][i]
        anns.append(
            {
                "id": int(objects["id"][i]) if objects.get("id") else i,
                "image_id": int(image_id),
                "category_id": int(objects["category_id"][i]),
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(objects["area"][i]) if objects.get("area") else float(w * h),
                "iscrowd": int(objects["iscrowd"][i]) if objects.get("iscrowd") else 0,
            }
        )
    return {"image_id": int(image_id), "annotations": anns}


def make_collate_fn(image_processor):
    """Build a collate_fn bound to a DETR image processor."""

    def collate_fn(examples):
        images = [ex["image"] for ex in examples]
        annotations = [format_annotations(ex["objects"], ex["image_id"]) for ex in examples]
        return image_processor(images=images, annotations=annotations, return_tensors="pt")

    return collate_fn
