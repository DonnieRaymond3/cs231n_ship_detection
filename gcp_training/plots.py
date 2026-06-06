#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BLUE, GREEN, ORANGE, RED, PURPLE, GRAY = "#4C72B0","#55A868","#DD8452","#C44E52","#8172B3","#aaaaaa"
LIT = [("Faster R-CNN", 0.867, 0.635), ("Cascade R-CNN", 0.877, 0.666), ("YOLOv8 (paper)", 0.887, 0.628)]


def _bar_labels(ax, vals, names, colors, lo):
    ax.bar(names, vals, color=colors, edgecolor="black", lw=0.5)
    ax.set_ylim(lo, 1.0)
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)


def cmd_comparison(args):
    names = [n for n, *_ in LIT]
    m50 = [a for _, a, _ in LIT]
    m5095 = [b for _, _, b in LIT]
    colors = [GRAY] * len(LIT)
    palette = [BLUE, GREEN, ORANGE, RED, PURPLE]
    for i, (name, a, b) in enumerate(args.model):
        names.append(name); m50.append(float(a)); m5095.append(float(b))
        colors.append(palette[i % len(palette)])

    fig, axes = plt.subplots(1, 2, figsize=(max(12, 2 * len(names)), 5))
    for ax, vals, title, lo in ((axes[0], m50, "mAP@50 on HRSID", 0.5),
                                (axes[1], m5095, "mAP@50-95 on HRSID", 0.3)):
        ax.set_title(title, fontsize=13, fontweight="bold")
        _bar_labels(ax, vals, names, colors, lo)
    plt.suptitle("SAR ship detection on HRSID (official split)\n*reported, not reproduced",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


def cmd_size(args):
    sizes = ["small", "medium", "large"]
    x = np.arange(3)
    n = len(args.model)
    w = 0.8 / max(n, 1)
    palette = [BLUE, GREEN, ORANGE, RED, PURPLE]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(args.model):
        vals = [float(v) for v in m[1:]]
        ax.bar(x + (i - (n-1)/2)*w, vals, w, label=m[0],
               color=palette[i % len(palette)], edgecolor="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_ylabel("AP@50-95"); ax.set_ylim(0, 0.9)
    ax.set_xlabel("object size (COCO area bucket)")
    ax.set_title("AP by object size on HRSID", fontsize=13, fontweight="bold")
    ax.legend(); plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


def cmd_generalization(args):
    names = [m[0] for m in args.model]
    hrsid = [float(m[1]) for m in args.model]
    ssdd  = [float(m[2]) for m in args.model]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.8*len(names)), 5))
    b1 = ax.bar(x - w/2, hrsid, w, label="HRSID (in-domain)", color=BLUE, edgecolor="black")
    b2 = ax.bar(x + w/2, ssdd,  w, label="SSDD (zero-shot)",  color=ORANGE, edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel(args.metric); ax.set_ylim(0, 1.0)
    ax.set_title(f"Cross-dataset generalization ({args.metric})\ntrained on HRSID, tested zero-shot on SSDD",
                 fontsize=12, fontweight="bold")
    ax.legend()
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


def _deim_curve(path):
    ep, m = [], []
    for line in open(path):
        line = line.strip()
        if not line: continue
        try: d = json.loads(line)
        except Exception: continue
        b = d.get("test_coco_eval_bbox")
        if b:
            ep.append(d.get("epoch", len(ep))); m.append(b[0])
    return ep, m


def _csv_curve(path, col):
    import pandas as pd
    df = pd.read_csv(path); df.columns = df.columns.str.strip()
    if col not in df.columns: return [], []
    s = df[["epoch", col]].dropna() if "epoch" in df.columns else df[[col]].dropna()
    x = s["epoch"].tolist() if "epoch" in s.columns else list(range(len(s)))
    return x, s[col].tolist()


def cmd_convergence(args):
    palette = [RED, ORANGE, PURPLE, GREEN, BLUE]
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, run in enumerate(args.run):
        name, kind, src, col = run[0], run[1], run[2], run[3] if len(run) > 3 else None
        if kind == "deim":
            x, y = _deim_curve(src)
        else:
            x, y = _csv_curve(src, col or "val/mAP_50_95")
        if x:
            ax.plot(x, y, label=name, color=palette[i % len(palette)], lw=2.2)
    ax.set_xlabel("epoch"); ax.set_ylabel("val mAP@50-95")
    ax.set_title("Convergence on HRSID (mAP@50-95 vs. epoch)", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("comparison")
    c.add_argument("--model", nargs=3, action="append", metavar=("NAME","MAP50","MAP5095"), default=[])
    c.add_argument("--out", default="paper_figures/fig_comparison_all.png")

    s = sub.add_parser("size")
    s.add_argument("--model", nargs=4, action="append", metavar=("NAME","S","M","L"), default=[])
    s.add_argument("--out", default="paper_figures/fig_ap_by_size.png")

    g = sub.add_parser("generalization")
    g.add_argument("--model", nargs=3, action="append", metavar=("NAME","HRSID","SSDD"), default=[])
    g.add_argument("--metric", default="mAP@50-95")
    g.add_argument("--out", default="paper_figures/fig_generalization.png")

    v = sub.add_parser("convergence")
    v.add_argument("--run", nargs="+", action="append", default=[],
                   help="NAME KIND(deim|csv) SRC [COL]")
    v.add_argument("--out", default="paper_figures/fig_convergence_overlay.png")

    args = p.parse_args()
    Path(args.out).parent.mkdir(exist_ok=True)
    {"comparison": cmd_comparison, "size": cmd_size,
     "generalization": cmd_generalization, "convergence": cmd_convergence}[args.cmd](args)


if __name__ == "__main__":
    main()
