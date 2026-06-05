#!/usr/bin/env python3
"""
Qualitative GT-vs-prediction panel for DEIM (ground truth green, predictions red).
Loads the model the same way DEIM's tools/inference/torch_inf.py does, draws with
PIL (no matplotlib needed in ~/.venv-deim), and uses the same --seed as
qualitative_samples.py so the sampled images match across models.

Run config must match the checkpoint's training resolution (640 for best_stg2).
Regenerate it first with gen_config.py --imgsz 640 (see the README/eval flow).
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deim-root", required=True)
    ap.add_argument("--config", required=True, help="path relative to deim-root")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gt", required=True, help="COCO test json")
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--resolution", type=int, default=640)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="qual_deim.png")
    args = ap.parse_args()

    sys.path.insert(0, args.deim_root)
    os.chdir(args.deim_root)
    from engine.core import YAMLConfig

    cfg = YAMLConfig(args.config, resume=args.ckpt)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_sizes):
            return self.postprocessor(self.model(images), orig_sizes)

    device = args.device
    model = Model().to(device).eval()
    tfm = T.Compose([T.Resize((args.resolution, args.resolution)), T.ToTensor()])

    coco = json.load(open(args.gt))
    id2img = {im["id"]: im for im in coco["images"]}
    gt = defaultdict(list)
    for a in coco["annotations"]:
        gt[a["image_id"]].append(a["bbox"])  # [x,y,w,h]

    random.seed(args.seed)
    ids = random.sample(list(id2img), args.n)

    rows = []
    for iid in ids:
        im = id2img[iid]
        pil = Image.open(os.path.join(args.img_dir, im["file_name"])).convert("RGB")
        w, h = pil.size

        gt_img = pil.copy()
        dg = ImageDraw.Draw(gt_img)
        for (x, y, bw, bh) in gt[iid]:
            dg.rectangle([x, y, x + bw, y + bh], outline="lime", width=2)
        dg.text((5, 5), f"Ground truth ({len(gt[iid])})", fill="lime")

        with torch.no_grad():
            data = tfm(pil).unsqueeze(0).to(device)
            orig = torch.tensor([[w, h]]).to(device)
            labels, boxes, scores = model(data, orig)
        boxes, scores = boxes[0].cpu(), scores[0].cpu()
        keep = scores > args.conf
        pred_img = pil.copy()
        dp = ImageDraw.Draw(pred_img)
        for b in boxes[keep]:
            dp.rectangle([float(b[0]), float(b[1]), float(b[2]), float(b[3])], outline="red", width=2)
        dp.text((5, 5), f"DEIM pred ({int(keep.sum())}) @conf{args.conf}", fill="red")

        row = Image.new("RGB", (w * 2 + 8, h), "white")
        row.paste(gt_img, (0, 0))
        row.paste(pred_img, (w + 8, 0))
        rows.append(row)

    W = rows[0].width
    panel = Image.new("RGB", (W, sum(r.height for r in rows) + 8 * (len(rows) - 1)), "white")
    y = 0
    for r in rows:
        panel.paste(r, (0, y)); y += r.height + 8
    out = os.path.join(os.path.dirname(args.ckpt), args.out) if not os.path.isabs(args.out) else args.out
    out = os.path.expanduser(args.out)
    panel.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
