#!/usr/bin/env python3
"""
Cross-dataset generalization plot: each model's mAP on HRSID (in-domain) vs.
SSDD (zero-shot, trained only on HRSID). Shows the generalization gap.

  python make_generalization_plot.py \
    --model "RT-DETR" 0.544 0.30 \
    --model "RF-DETR-L" 0.695 0.40 \
    --model "DEIM" 0.718 0.42 \
    --metric "mAP@50-95" --out generalization.png
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#4C72B0", "#DD8452"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs=3, action="append", metavar=("NAME", "HRSID", "SSDD"),
                    default=[], help="model name, HRSID score, SSDD score (repeatable)")
    ap.add_argument("--metric", default="mAP@50-95")
    ap.add_argument("--out", default="generalization.png")
    args = ap.parse_args()

    names = [m[0] for m in args.model]
    hrsid = [float(m[1]) for m in args.model]
    ssdd = [float(m[2]) for m in args.model]
    x = np.arange(len(names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(names)), 5))
    b1 = ax.bar(x - w / 2, hrsid, w, label="HRSID (in-domain)", color=BLUE, edgecolor="black")
    b2 = ax.bar(x + w / 2, ssdd, w, label="SSDD (zero-shot)", color=ORANGE, edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel(args.metric); ax.set_ylim(0, 1.0)
    ax.set_title(f"Cross-dataset generalization ({args.metric})\ntrained on HRSID, tested zero-shot on SSDD",
                 fontsize=12, fontweight="bold")
    ax.legend()
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
