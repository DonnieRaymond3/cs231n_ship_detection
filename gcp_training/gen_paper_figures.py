#!/usr/bin/env python3
"""Generate paper figures from the local metric files. Run from repo root."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("paper_figures"); OUT.mkdir(exist_ok=True)
BLUE, GREEN, ORANGE, RED, PURPLE, GRAY = "#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B3", "#aaaaaa"


def deim_curve(path):
    ep, m = [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        b = d.get("test_coco_eval_bbox")
        if b:
            ep.append(d.get("epoch", len(ep))); m.append(b[0])
    return ep, m


def csv_curve(path, col):
    df = pd.read_csv(path); df.columns = df.columns.str.strip()
    c = next((c for c in df.columns if c.lower() == col.lower()), None)
    if c is None:
        return [], []
    s = df[["epoch", c]].dropna() if "epoch" in df.columns else df[[c]].dropna()
    x = s["epoch"].tolist() if "epoch" in s else list(range(len(s)))
    return x, s[c].tolist()


# ---------- 1. Training-curve overlay (mAP@50-95 vs epoch) ----------
runs = [
    ("RT-DETR-L @800", *csv_curve("checkpoints/rtdetr_hrsid_50ep/results.csv", "metrics/mAP50-95(B)"), RED),
    ("RF-DETR-M @576", *csv_curve("checkpoints_rfdetr/metrics.csv", "val/mAP_50_95"), ORANGE),
    ("RF-DETR-L @768", *csv_curve("checkpoints_rfdetr_large/metrics.csv", "val/mAP_50_95"), PURPLE),
    ("DEIM-S @640", *deim_curve("checkpoints_deim/deim_hrsid_s/log.txt"), GREEN),
    ("DEIM-S @800", *deim_curve("checkpoints_deim_800/log.txt"), BLUE),
]
plt.figure(figsize=(9, 6))
for name, x, y, c in runs:
    if x:
        plt.plot(x, y, label=name, color=c, lw=2.2)
plt.xlabel("epoch"); plt.ylabel("val mAP@50-95")
plt.title("Convergence on HRSID (mAP@50-95 vs. epoch)", fontsize=13, fontweight="bold")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(OUT / "fig_convergence_overlay.png", dpi=150, bbox_inches="tight")
print("saved fig_convergence_overlay.png")

# ---------- 2. Grand comparison (best config per model + literature) ----------
names = ["Faster R-CNN*", "Cascade R-CNN*", "YOLOv8*", "RT-DETR-L", "YOLOv11l", "RF-DETR-L", "DEIM-S"]
m50 = [0.867, 0.877, 0.887, 0.814, 0.863, 0.934, 0.936]
m5095 = [0.635, 0.666, 0.628, 0.539, 0.614, 0.707, 0.718]
colors = [GRAY, GRAY, GRAY, RED, ORANGE, PURPLE, GREEN]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, vals, title, lo in ((axes[0], m50, "mAP@50 on HRSID", 0.5),
                            (axes[1], m5095, "mAP@50-95 on HRSID", 0.3)):
    ax.bar(names, vals, color=colors, edgecolor="black", lw=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold"); ax.set_ylim(lo, 1.0)
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
plt.suptitle("SAR ship detection on HRSID (official split)\n*reported, not reproduced",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "fig_comparison_all.png", dpi=150, bbox_inches="tight")
print("saved fig_comparison_all.png")

# ---------- 3. AP by object size (all models) ----------
sizes = ["small", "medium", "large"]
ap_models = [
    ("RT-DETR-L", [0.556, 0.523, 0.205], RED),
    ("YOLOv11l", [0.629, 0.591, 0.020], ORANGE),
    ("RF-DETR-L", [0.711, 0.763, 0.667], PURPLE),
    ("DEIM-S", [0.725, 0.757, 0.616], GREEN),
]
xpos = np.arange(len(sizes))
bw = 0.8 / len(ap_models)
plt.figure(figsize=(8, 5))
for i, (name, vals, c) in enumerate(ap_models):
    plt.bar(xpos + (i - (len(ap_models) - 1) / 2) * bw, vals, bw,
            label=name, color=c, edgecolor="black", lw=0.5)
plt.xticks(xpos, sizes); plt.ylabel("AP@50-95"); plt.ylim(0, 0.9)
plt.xlabel("object size (COCO area bucket)")
plt.title("AP by object size on HRSID", fontsize=13, fontweight="bold")
plt.legend(); plt.tight_layout()
plt.savefig(OUT / "fig_ap_by_size.png", dpi=150, bbox_inches="tight")
print("saved fig_ap_by_size.png")
