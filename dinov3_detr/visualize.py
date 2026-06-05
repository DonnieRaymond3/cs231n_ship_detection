"""Visualize predictions from the trained DINO+DETR model against COCO boxes.

Examples:
    # Local, after pulling a checkpoint from Modal.
    python visualize.py --model-dir outputs --dataset ssdd --index 0 --threshold 0.3

    # HRSID validation image by file name.
    python visualize.py --model-dir outputs --dataset hrsid-val --file-name P0001_0_800_10190_10990.jpg
"""

import argparse
import json
import os

from PIL import Image, ImageDraw
from transformers import AutoImageProcessor

from config import DEFAULTS
from evaluate import load_trained_model, pick_device


DATASETS = {
    "hrsid-val": {
        "images": DEFAULTS["val_images"],
        "ann": DEFAULTS["val_ann"],
        "category_id": DEFAULTS["coco_category_id"],
    },
    "ssdd": {
        "images": DEFAULTS["test_images"],
        "ann": DEFAULTS["test_ann"],
        "category_id": DEFAULTS["test_category_id"],
    },
}


def load_coco(ann_file):
    with open(ann_file) as f:
        coco = json.load(f)
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    return coco, anns_by_image


def choose_image(coco, image_id=None, file_name=None, index=0):
    if image_id is not None:
        for info in coco["images"]:
            if int(info["id"]) == int(image_id):
                return info
        raise ValueError(f"No image with id {image_id}")
    if file_name is not None:
        for info in coco["images"]:
            if info["file_name"] == file_name:
                return info
        raise ValueError(f"No image with file_name {file_name}")
    return coco["images"][index]


def draw_boxes(draw, boxes, color, label_fn, width=3):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        label = label_fn(i)
        if label:
            draw.text((x1 + 3, max(0, y1 - 12)), label, fill=color)


def visualize_prediction(
    model_dir,
    images_dir,
    ann_file,
    output,
    image_id=None,
    file_name=None,
    index=0,
    threshold=0.3,
    max_preds=50,
    backbone=DEFAULTS["backbone"],
    pretrained_detr=DEFAULTS["pretrained_detr"],
):
    import torch

    device = pick_device()
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = load_trained_model(model_dir, backbone, pretrained_detr=pretrained_detr, device=device)

    coco, anns_by_image = load_coco(ann_file)
    info = choose_image(coco, image_id=image_id, file_name=file_name, index=index)
    image = Image.open(os.path.join(images_dir, info["file_name"])).convert("RGB")

    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        target_sizes = torch.tensor([[info["height"], info["width"]]], device=device)
        pred = processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

    gt_boxes = []
    for ann in anns_by_image.get(info["id"], []):
        x, y, w, h = ann["bbox"]
        gt_boxes.append((x, y, x + w, y + h))

    pred_boxes = []
    pred_scores = []
    for score, box in zip(pred["scores"][:max_preds], pred["boxes"][:max_preds]):
        x1, y1, x2, y2 = box.tolist()
        pred_boxes.append((x1, y1, x2, y2))
        pred_scores.append(float(score))

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    draw_boxes(draw, gt_boxes, "lime", lambda i: "GT", width=3)
    draw_boxes(draw, pred_boxes, "red", lambda i: f"{pred_scores[i]:.2f}", width=2)

    caption = (
        f"{info['file_name']} | GT={len(gt_boxes)} | "
        f"preds@{threshold:.2f}={len(pred_boxes)} (green=GT, red=pred)"
    )
    draw.rectangle((0, 0, min(canvas.width, 900), 18), fill="black")
    draw.text((4, 3), caption, fill="white")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    canvas.save(output)
    print(f"Saved visualization to {output}")
    return output


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default=DEFAULTS["output_dir"])
    p.add_argument("--dataset", choices=sorted(DATASETS), default="ssdd")
    p.add_argument("--images", default=None, help="override image directory")
    p.add_argument("--ann", default=None, help="override COCO annotation file")
    p.add_argument("--image-id", type=int, default=None)
    p.add_argument("--file-name", default=None)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--max-preds", type=int, default=50)
    p.add_argument("--output", default="prediction_vs_gt.png")
    p.add_argument("--backbone", default=DEFAULTS["backbone"])
    p.add_argument("--pretrained-detr", default=DEFAULTS["pretrained_detr"])
    return p.parse_args()


def main():
    args = parse_args()
    ds = DATASETS[args.dataset]
    visualize_prediction(
        model_dir=args.model_dir,
        images_dir=args.images or ds["images"],
        ann_file=args.ann or ds["ann"],
        output=args.output,
        image_id=args.image_id,
        file_name=args.file_name,
        index=args.index,
        threshold=args.threshold,
        max_preds=args.max_preds,
        backbone=args.backbone,
        pretrained_detr=args.pretrained_detr or None,
    )


if __name__ == "__main__":
    main()
