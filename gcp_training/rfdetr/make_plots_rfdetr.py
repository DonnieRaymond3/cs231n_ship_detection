#!/usr/bin/env python3
"""
Plots + best stats from an RF-DETR run's metrics.csv (always written to the
output dir by RF-DETR's CSVLogger).

Logged keys: val/mAP_50_95, val/mAP_50, val/mAP_75, val/mAR, train/loss.

  python make_plots_rfdetr.py --csv runs_rfdetr/rfdetr_medium_hrsid_50ep/metrics.csv
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ORANGE, BLUE = "#DD8452", "#4C72B0"


def series(df, col):
    if col not in df.columns:
        return None, None
    s = df[["epoch", col]].dropna()
    return s["epoch"].tolist(), s[col].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to RF-DETR metrics.csv")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    csv = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv.parent
    df = pd.read_csv(csv)
    if "epoch" not in df.columns:
        df["epoch"] = range(len(df))

    e50, v50 = series(df, "val/mAP_50")
    e5095, v5095 = series(df, "val/mAP_50_95")
    eloss, vloss = series(df, "train/loss")

    if not v5095:
        raise SystemExit(f"No val/mAP_50_95 in {csv}. Columns: {list(df.columns)}")

    bi = max(range(len(v5095)), key=lambda i: v5095[i])
    best50 = v50[max(range(len(v50)), key=lambda i: v50[i])] if v50 else float("nan")
    print("===== RF-DETR best results (HRSID test) =====")
    print(f"  best epoch (by mAP@50-95): {int(e5095[bi])}")
    print(f"  mAP@50-95: {v5095[bi]:.4f}")
    print(f"  mAP@50:    {best50:.4f}")

    panels = [("Val mAP@50", e50, v50, ORANGE),
              ("Val mAP@50-95", e5095, v5095, ORANGE),
              ("Train loss", eloss, vloss, BLUE)]
    panels = [p for p in panels if p[1]]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, x, y, c) in zip(axes, panels):
        ax.plot(x, y, color=c, lw=2)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    out = out_dir / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
