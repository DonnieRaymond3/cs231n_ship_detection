"""Merge the SSDD COCO train + test splits into one full-dataset test file.

We train on HRSID and use the *entire* SSDD dataset (1,160 images) as a
held-out, cross-domain test set. SSDD's coco_style ships separate train.json
and test.json with disjoint image ids; all 1,160 images also exist together in
``voc_style/JPEGImages``, so the merged file points there.

Output: SSDD/ssdd_all.json  (category_id 0 == ship, matching SSDD's convention)
"""

import argparse
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC = os.path.join(REPO_ROOT, "SSDD", "BBox_SSDD", "coco_style", "annotations")
DEFAULT_OUT = os.path.join(REPO_ROOT, "SSDD", "ssdd_all.json")


def merge(src_dir: str, splits, out_path: str):
    images, annotations = [], []
    seen_image_ids = set()
    next_ann_id = 0
    categories = None

    for split in splits:
        path = os.path.join(src_dir, f"{split}.json")
        with open(path) as f:
            coco = json.load(f)
        if categories is None:
            categories = coco["categories"]

        for img in coco["images"]:
            if img["id"] in seen_image_ids:
                continue
            seen_image_ids.add(img["id"])
            images.append(img)

        # Re-index annotation ids so train + test ids never collide.
        for ann in coco["annotations"]:
            ann = dict(ann)
            ann["id"] = next_ann_id
            next_ann_id += 1
            annotations.append(ann)

    merged = {
        "info": {"description": "SSDD full set (train+test) for cross-domain testing"},
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    with open(out_path, "w") as f:
        json.dump(merged, f)

    print(f"Wrote {out_path}")
    print(f"  images: {len(images)}  annotations: {len(annotations)}")
    print(f"  categories: {categories}")
    return merged


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-dir", default=DEFAULT_SRC)
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()
    merge(args.src_dir, args.splits, args.out)


if __name__ == "__main__":
    main()
