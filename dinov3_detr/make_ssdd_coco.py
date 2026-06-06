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
