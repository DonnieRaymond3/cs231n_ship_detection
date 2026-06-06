import argparse
import json
import os
import shutil
from pathlib import Path

def build_split(split_dir: Path, ann_path: Path, img_src: Path, link: bool):
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(ann_path) as f:
        coco = json.load(f)

    src_index = {p.name: p for p in img_src.rglob("*.jpg")}
    missing = 0
    for im in coco["images"]:
        name = os.path.basename(im["file_name"])
        src = src_index.get(name)
        dst = split_dir / name
        if src is None:
            missing += 1
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if link:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(str(src), str(dst))

    with open(split_dir / "_annotations.coco.json", "w") as f:
        json.dump(coco, f)
    print(f"  {split_dir.name}: {len(coco['images'])} images "
          f"({missing} missing), {len(coco['annotations'])} annotations")

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--hrsid-root", default=str(repo_root / "HRSID" / "HRSID_JPG"))
    ap.add_argument("--out", default=str(repo_root / "datasets" / "HRSID_RFDETR"))
    ap.add_argument("--copy", action="store_true", help="Copy images instead of symlinking.")
    args = ap.parse_args()

    hrsid = Path(args.hrsid_root)
    img_src = hrsid / "JPEGImages"
    train_ann = hrsid / "annotations" / "train2017.json"
    test_ann = hrsid / "annotations" / "test2017.json"
    for p in (img_src, train_ann, test_ann):
        if not p.exists():
            raise FileNotFoundError(f"Missing HRSID path: {p}")

    out = Path(args.out)
    link = not args.copy
    print(f"Building RF-DETR dataset at {out}")
    build_split(out / "train", train_ann, img_src, link)
    build_split(out / "valid", test_ann, img_src, link)
    build_split(out / "test", test_ann, img_src, link)
    print(f"\nDone. dataset_dir = {out}")

if __name__ == "__main__":
    main()
