#!/usr/bin/env python3
"""
Compute the full COCO AP breakdown (incl. AP small/medium/large) for RT-DETR
(ultralytics) or RF-DETR on the HRSID test split, via inference + pycocotools.

DEIM already logs the full breakdown in its log.txt; this fills in the other two
so you get a consistent size-stratified table across all models.

Run ON the VM (GPU), in the matching venv:
  RT-DETR  (~/.venv):        python eval_size_breakdown.py --framework rtdetr \
               --ckpt <best.pt>
  RF-DETR  (~/.venv-rfdetr): python eval_size_breakdown.py --framework rfdetr \
               --ckpt <checkpoint_best_ema.pth> --model-cls RFDETRLarge --resolution 768
"""
import argparse
import json
import os
from pathlib import Path


def load_gt(gt_path):
    coco = json.load(open(gt_path))
    return coco["images"]  # each has id, file_name, width, height


def predict_rtdetr(ckpt, images, img_dir, conf, cat_id):
    from ultralytics import RTDETR
    m = RTDETR(ckpt)
    dets = []
    for im in images:
        r = m.predict(os.path.join(img_dir, im["file_name"]), conf=conf,
                      verbose=False, imgsz=800)[0]
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets.append({"image_id": im["id"], "category_id": cat_id,
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(b.conf[0])})
    return dets


def predict_rfdetr(ckpt, images, img_dir, conf, model_cls, resolution, cat_id):
    import rfdetr
    from PIL import Image
    M = getattr(rfdetr, model_cls)(pretrain_weights=ckpt, resolution=resolution)
    dets = []
    for im in images:
        img = Image.open(os.path.join(img_dir, im["file_name"])).convert("RGB")
        d = M.predict(img, threshold=conf)
        for (x1, y1, x2, y2), s in zip(d.xyxy, d.confidence):
            dets.append({"image_id": im["id"], "category_id": cat_id,
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                         "score": float(s)})
    return dets


def main():
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", required=True, choices=["rtdetr", "rfdetr"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gt", default=str(repo / "HRSID/HRSID_JPG/annotations/test2017.json"))
    ap.add_argument("--img-dir", default=str(repo / "HRSID/HRSID_JPG/JPEGImages"))
    ap.add_argument("--conf", type=float, default=0.01, help="low threshold for proper AP")
    ap.add_argument("--model-cls", default="RFDETRLarge", help="rfdetr only")
    ap.add_argument("--resolution", type=int, default=768, help="rfdetr only")
    ap.add_argument("--cat-id", type=int, default=1,
                    help="category_id to assign predictions (HRSID=1, SSDD=0)")
    ap.add_argument("--out", default=None, help="optional: save predictions json")
    args = ap.parse_args()

    images = load_gt(args.gt)
    print(f"Running {args.framework} on {len(images)} test images (cat_id={args.cat_id})...")
    if args.framework == "rtdetr":
        dets = predict_rtdetr(args.ckpt, images, args.img_dir, args.conf, args.cat_id)
    else:
        dets = predict_rfdetr(args.ckpt, images, args.img_dir, args.conf,
                              args.model_cls, args.resolution, args.cat_id)
    print(f"  {len(dets)} detections")

    if args.out:
        json.dump(dets, open(args.out, "w"))

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    cocoGt = COCO(args.gt)
    cocoDt = cocoGt.loadRes(dets)
    E = COCOeval(cocoGt, cocoDt, "bbox")
    E.evaluate(); E.accumulate(); E.summarize()
    s = E.stats
    print("\n===== SIZE BREAKDOWN =====")
    print(f"mAP@50-95 = {s[0]:.4f}")
    print(f"mAP@50    = {s[1]:.4f}")
    print(f"AP_small  = {s[3]:.4f}")
    print(f"AP_medium = {s[4]:.4f}")
    print(f"AP_large  = {s[5]:.4f}")


if __name__ == "__main__":
    main()
