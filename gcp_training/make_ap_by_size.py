#!/usr/bin/env python3
"""
Grouped bar chart of AP by object size (small/medium/large) across models.

  python make_ap_by_size.py \
    --model "RT-DETR"   <s> <m> <l> \
    --model "RF-DETR-L" <s> <m> <l> \
    --model "DEIM"      0.725 0.757 0.616 \
    --out paper_figures/fig_ap_by_size.png
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs=4, action="append",
                    metavar=("NAME", "SMALL", "MEDIUM", "LARGE"), default=[])
    ap.add_argument("--out", default="paper_figures/fig_ap_by_size.png")
    args = ap.parse_args()

    sizes = ["small", "medium", "large"]
    x = np.arange(len(sizes))
    n = len(args.model)
    w = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(args.model):
        name = m[0]
        vals = [float(v) for v in m[1:]]
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=name,
               color=COLORS[i % len(COLORS)], edgecolor="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_ylabel("AP@50-95"); ax.set_ylim(0, 0.9)
    ax.set_xlabel("object size (COCO area bucket)")
    ax.set_title("AP by object size on HRSID", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
