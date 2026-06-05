#!/usr/bin/env python3
"""
Plots from a DEIM run's log.txt:
  1. training_curves.png   -- mAP@50 / mAP@50-95 (and train loss) over epochs
  2. baseline_comparison.png -- DEIM vs. RT-DETR (yours) vs. published HRSID baselines

DEIM writes one JSON object per epoch to log.txt with `test_coco_eval_bbox`
(12 COCO AP values): index 0 = mAP@50-95, index 1 = mAP@50.

Run locally after downloading the run:
  python make_plots_deim.py --log outputs/deim_hrsid_s/log.txt \
      --rtdetr-map50 0.90 --rtdetr-map5095 0.65
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

LITERATURE = [
    ("Faster R-CNN", 0.867, 0.635),
    ("Cascade R-CNN", 0.877, 0.666),
    ("YOLOv8 (paper)", 0.887, 0.628),
]
BLUE, ORANGE, GREEN, GRAY = "#4C72B0", "#DD8452", "#55A868", "#aaaaaa"


def load_log(path):
    epochs, m50, m5095, loss = [], [], [], []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            bbox = d.get("test_coco_eval_bbox")
            if not bbox:
                continue
            epochs.append(d.get("epoch", i))
            m5095.append(bbox[0])
            m50.append(bbox[1])
            # train loss key varies; grab the first that looks like a loss
            lv = d.get("train_loss", d.get("loss"))
            loss.append(lv if isinstance(lv, (int, float)) else None)
    return epochs, m50, m5095, loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to DEIM log.txt")
    ap.add_argument("--out-dir", default=None, help="Where to save PNGs (default: log's dir)")
    ap.add_argument("--label", default="DEIM (ours)")
    ap.add_argument("--rtdetr-map50", type=float, default=None,
                    help="Your RT-DETR mAP@50 (adds it to the comparison)")
    ap.add_argument("--rtdetr-map5095", type=float, default=None)
    args = ap.parse_args()

    log = Path(args.log)
    out_dir = Path(args.out_dir) if args.out_dir else log.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs, m50, m5095, loss = load_log(log)
    if not epochs:
        raise SystemExit(f"No 'test_coco_eval_bbox' entries found in {log}")

    best_i = max(range(len(m5095)), key=lambda i: m5095[i])
    our50, our5095 = m50[best_i], m5095[best_i]
    print(f"Best epoch {epochs[best_i]}: mAP50={our50:.3f}  mAP50-95={our5095:.3f}")

    # ---- training curves ----
    have_loss = any(v is not None for v in loss)
    n = 3 if have_loss else 2
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    axes[0].plot(epochs, m50, color=ORANGE, lw=2)
    axes[0].set_title("Val mAP@50", fontweight="bold")
    axes[1].plot(epochs, m5095, color=ORANGE, lw=2)
    axes[1].set_title("Val mAP@50-95", fontweight="bold")
    for ax in axes[:2]:
        ax.scatter([epochs[best_i]], [m50[best_i] if ax is axes[0] else m5095[best_i]],
                   color="red", zorder=5)
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    if have_loss:
        xs = [e for e, v in zip(epochs, loss) if v is not None]
        ys = [v for v in loss if v is not None]
        axes[2].plot(xs, ys, color=BLUE, lw=2)
        axes[2].set_title("Train loss", fontweight="bold")
        axes[2].set_xlabel("epoch"); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    p1 = out_dir / "training_curves.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight"); print(f"saved {p1}")

    # ---- baseline comparison ----
    names = [n for n, _, _ in LITERATURE]
    v50 = [a for _, a, _ in LITERATURE]
    v5095 = [b for _, _, b in LITERATURE]
    colors = [GRAY] * len(LITERATURE)
    if args.rtdetr_map50 is not None and args.rtdetr_map5095 is not None:
        names.append("RT-DETR (ours)"); v50.append(args.rtdetr_map50)
        v5095.append(args.rtdetr_map5095); colors.append(BLUE)
    names.append(args.label); v50.append(our50); v5095.append(our5095); colors.append(GREEN)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, vals, title, lo in (
        (axes[0], v50, "mAP@50 on HRSID", 0.5),
        (axes[1], v5095, "mAP@50-95 on HRSID", 0.3),
    ):
        ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(lo, 1.0)
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.suptitle("DEIM vs. baselines (HRSID)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p2 = out_dir / "baseline_comparison.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight"); print(f"saved {p2}")


if __name__ == "__main__":
    main()
