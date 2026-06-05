#!/usr/bin/env python3
"""
Grand comparison bar chart: your trained models vs. published HRSID baselines.

Literature baselines are built in (gray). Add each of your models with:
  --model "NAME" MAP50 MAP5095

Example:
  python make_comparison.py \
    --model "RT-DETR (ours)" 0.833 0.544 \
    --model "DEIM (ours)"    0.936 0.718 \
    --model "RF-DETR (ours)" 0.94  0.73 \
    --out comparison.png
"""
import argparse

import matplotlib.pyplot as plt

LITERATURE = [
    ("Faster R-CNN", 0.867, 0.635),
    ("Cascade R-CNN", 0.877, 0.666),
    ("YOLOv8 (paper)", 0.887, 0.628),
]
GRAY = "#aaaaaa"
OURS_COLORS = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs=3, action="append", metavar=("NAME", "MAP50", "MAP5095"),
                    default=[], help="Add one of your models (repeatable).")
    ap.add_argument("--out", default="comparison.png")
    args = ap.parse_args()

    names = [n for n, _, _ in LITERATURE]
    m50 = [a for _, a, _ in LITERATURE]
    m5095 = [b for _, _, b in LITERATURE]
    colors = [GRAY] * len(LITERATURE)

    for i, (name, a, b) in enumerate(args.model):
        names.append(name)
        m50.append(float(a))
        m5095.append(float(b))
        colors.append(OURS_COLORS[i % len(OURS_COLORS)])

    fig, axes = plt.subplots(1, 2, figsize=(max(12, 2 * len(names)), 5))
    for ax, vals, title, lo in (
        (axes[0], m50, "mAP@50 on HRSID", 0.5),
        (axes[1], m5095, "mAP@50-95 on HRSID", 0.3),
    ):
        ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(lo, 1.0)
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.suptitle("SAR Ship Detection on HRSID — models vs. reported baselines",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
