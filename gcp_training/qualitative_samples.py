import argparse
import json
import os
import random
from collections import defaultdict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

def predict(framework, ckpt, img_path, conf, model_cls, resolution):
    boxes = []                               
    if framework == "rtdetr":
        from ultralytics import RTDETR
        if not hasattr(predict, "_m"):
            predict._m = RTDETR(ckpt)
        r = predict._m.predict(img_path, conf=conf, verbose=False, imgsz=800)[0]
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            boxes.append((x1, y1, x2, y2, float(b.conf[0])))
    else:
        import rfdetr
        if not hasattr(predict, "_m"):
            predict._m = getattr(rfdetr, model_cls)(pretrain_weights=ckpt, resolution=resolution)
        d = predict._m.predict(Image.open(img_path).convert("RGB"), threshold=conf)
        for (x1, y1, x2, y2), s in zip(d.xyxy, d.confidence):
            boxes.append((float(x1), float(y1), float(x2), float(y2), float(s)))
    return boxes

def main():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", required=True, choices=["rtdetr", "rfdetr"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gt", default=os.path.join(repo, "HRSID/HRSID_JPG/annotations/test2017.json"))
    ap.add_argument("--img-dir", default=os.path.join(repo, "HRSID/HRSID_JPG/JPEGImages"))
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conf", type=float, default=0.3, help="display threshold")
    ap.add_argument("--model-cls", default="RFDETRLarge")
    ap.add_argument("--resolution", type=int, default=768)
    ap.add_argument("--out", default="qualitative.png")
    args = ap.parse_args()

    coco = json.load(open(args.gt))
    gt = defaultdict(list)
    id2img = {im["id"]: im for im in coco["images"]}
    for a in coco["annotations"]:
        gt[a["image_id"]].append(a["bbox"])             

    random.seed(args.seed)
    ids = random.sample(list(id2img), args.n)

    fig, axes = plt.subplots(args.n, 2, figsize=(8, 4 * args.n))
    if args.n == 1:
        axes = [axes]
    for row, iid in enumerate(ids):
        im = id2img[iid]
        path = os.path.join(args.img_dir, im["file_name"])
        img = Image.open(path).convert("L")

        axg, axp = axes[row]
        for ax in (axg, axp):
            ax.imshow(img, cmap="gray"); ax.axis("off")

        for (x, y, w, h) in gt[iid]:
            axg.add_patch(patches.Rectangle((x, y), w, h, lw=1.4, edgecolor="lime", facecolor="none"))
        axg.set_title(f"Ground truth ({len(gt[iid])})", fontsize=10)

        preds = predict(args.framework, args.ckpt, path, args.conf, args.model_cls, args.resolution)
        for (x1, y1, x2, y2, s) in preds:
            axp.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=1.4,
                          edgecolor="red", facecolor="none"))
        axp.set_title(f"Predicted ({len(preds)}) @conf{args.conf}", fontsize=10)

    plt.suptitle(f"{args.framework.upper()} — green: GT, red: predictions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")

if __name__ == "__main__":
    main()
