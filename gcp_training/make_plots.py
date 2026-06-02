#!/usr/bin/env python3
"""
Generate report plots from a finished RT-DETR run:
  1. baseline_comparison.png  -- our mAP vs. published HRSID baselines (bar chart)
  2. training_curves.png      -- loss + mAP / precision / recall over epochs

Reads the run's results.csv (no GPU needed), so you can run it locally after
scp-ing the run folder down:

    python make_plots.py --run-dir checkpoints/rtdetr_hrsid_50ep
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Published HRSID test-split numbers (from Wei et al. and the notebook).
LITERATURE = [
    ("Faster R-CNN",  0.867, 0.635),
    ("Cascade R-CNN", 0.877, 0.666),
    ("YOLOv8 (paper)", 0.887, 0.628),
]

BLUE, ORANGE, GRAY = "#4C72B0", "#DD8452", "#aaaaaa"


def find_col(df, *needles):
    """Return first column whose name contains all needles (case-insensitive)."""
    for c in df.columns:
        low = c.lower()
        if all(n.lower() in low for n in needles):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Run folder containing results.csv")
    ap.add_argument("--label", default="RT-DETR (ours)", help="Legend label for our model")
    args = ap.parse_args()

    run = Path(args.run_dir)
    csv = run / "results.csv"
    if not csv.exists():
        raise FileNotFoundError(f"No results.csv in {run}")

    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()

    map5095_first = find_col(df, "map50-95")
    map50_col = find_col(df, "map50(b)") or next(
        (c for c in df.columns if "map50" in c.lower() and c != map5095_first), None
    )
    map5095_col = find_col(df, "map50-95")
    prec_col = find_col(df, "precision")
    rec_col = find_col(df, "recall")

    # Our best epoch (by mAP50-95, the stricter metric).
    best_idx = df[map5095_col].idxmax()
    our_map50 = float(df[map50_col].iloc[best_idx])
    our_map5095 = float(df[map5095_col].iloc[best_idx])
    best_ep = int(df["epoch"].iloc[best_idx]) if "epoch" in df.columns else best_idx
    print(f"Best epoch {best_ep}: mAP50={our_map50:.3f}  mAP50-95={our_map5095:.3f}")

    # ---- 1. Baseline comparison bar chart ----
    names = [n for n, _, _ in LITERATURE] + [args.label]
    m50 = [v for _, v, _ in LITERATURE] + [our_map50]
    m5095 = [v for _, _, v in LITERATURE] + [our_map5095]
    colors = [GRAY] * len(LITERATURE) + [ORANGE]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, vals, title, lo in (
        (axes[0], m50, "mAP@50 on HRSID", 0.5),
        (axes[1], m5095, "mAP@50-95 on HRSID", 0.3),
    ):
        ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(lo, 1.0)
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    plt.suptitle("RT-DETR vs. baselines (HRSID)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out1 = run / "baseline_comparison.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"saved {out1}")

    # ---- 2. Training curves ----
    panels = [
        (find_col(df, "train", "box"), "Train box loss", "loss"),
        (map50_col, "Val mAP@50", "mAP"),
        (map5095_col, "Val mAP@50-95", "mAP"),
    ]
    if prec_col and rec_col:
        panels.append((None, "Val precision / recall", "score"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    x = df["epoch"] if "epoch" in df.columns else range(len(df))
    for ax, (col, title, ylab) in zip(axes, panels):
        if col is None:  # precision + recall together
            ax.plot(x, df[prec_col], color=BLUE, label="precision")
            ax.plot(x, df[rec_col], color=ORANGE, label="recall")
            ax.legend()
        elif col in df.columns:
            ax.plot(x, df[col], color=ORANGE if "mAP" in (col or "") else BLUE, linewidth=2)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    out2 = run / "training_curves.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
