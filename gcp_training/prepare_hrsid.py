import argparse
import json
import os
import shutil
from pathlib import Path

import yaml

def coco_to_yolo(ann_file, images_src_dir, out_images_dir, out_labels_dir, link=True):
    out_images_dir = Path(out_images_dir)
    out_labels_dir = Path(out_labels_dir)
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_file) as f:
        data = json.load(f)

    img_info = {img["id"]: img for img in data["images"]}

    ann_by_img = {}
    for ann in data["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    src_index = {p.name: p for p in Path(images_src_dir).rglob("*.jpg")}

    converted, skipped = 0, 0
    for img_id, img in img_info.items():
        fname = img["file_name"]
        w, h = img["width"], img["height"]

        src_path = src_index.get(os.path.basename(fname))
        if src_path is None:
            skipped += 1
            continue

        dst_img = out_images_dir / os.path.basename(fname)
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        if link:
            dst_img.symlink_to(src_path.resolve())
        else:
            shutil.copy2(str(src_path), str(dst_img))

        label_path = out_labels_dir / (Path(fname).stem + ".txt")
        with open(label_path, "w") as lf:
            for ann in ann_by_img.get(img_id, []):
                x_min, y_min, bw, bh = ann["bbox"]
                cx = (x_min + bw / 2) / w
                cy = (y_min + bh / 2) / h
                nw = bw / w
                nh = bh / h

                cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
                nw, nh = min(max(nw, 0.0), 1.0), min(max(nh, 0.0), 1.0)
                lf.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        converted += 1

    print(f"  converted {converted} images, skipped {skipped} (missing source)")
    return converted

def main():
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hrsid-root",
        default=str(repo_root / "HRSID" / "HRSID_JPG"),
        help="Path to HRSID_JPG (contains JPEGImages/ and annotations/).",
    )
    ap.add_argument(
        "--out",
        default=str(repo_root / "datasets" / "HRSID_YOLO"),
        help="Output dataset root for the YOLO-format tree.",
    )
    ap.add_argument("--copy", action="store_true", help="Copy images instead of symlinking.")
    args = ap.parse_args()

    hrsid = Path(args.hrsid_root)
    images_src = hrsid / "JPEGImages"
    train_ann = hrsid / "annotations" / "train2017.json"
    test_ann = hrsid / "annotations" / "test2017.json"

    for p in (images_src, train_ann, test_ann):
        if not p.exists():
            raise FileNotFoundError(f"Expected HRSID file/dir not found: {p}")

    out = Path(args.out)
    link = not args.copy

    print("Converting train split...")
    coco_to_yolo(train_ann, images_src, out / "images/train", out / "labels/train", link)
    print("Converting val (test) split...")
    coco_to_yolo(test_ann, images_src, out / "images/val", out / "labels/val", link)

    yaml_path = out / "hrsid.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(
            {"path": str(out), "train": "images/train", "val": "images/val", "names": {0: "ship"}},
            f,
            sort_keys=False,
        )

    n_train = len(list((out / "images/train").glob("*.jpg")))
    n_val = len(list((out / "images/val").glob("*.jpg")))
    print(f"\nDataset ready at {out}")
    print(f"  train images: {n_train}  |  val images: {n_val}")
    print(f"  yaml: {yaml_path}")

if __name__ == "__main__":
    main()
