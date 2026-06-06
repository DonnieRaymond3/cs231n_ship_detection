import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

PAT = re.compile(r"(P\d+)_(\d+)_(\d+)_(\d+)_(\d+)$")

def parse(fn):
    m = PAT.match(fn.rsplit(".", 1)[0])
    if not m:
        return None, None
    s = m.group(1)
    x1, x2, y1, y2 = map(int, m.groups()[1:])
    return s, (x1, x2, y1, y2)

def overlap(a, b):
    ax1, ax2, ay1, ay2 = a
    bx1, bx2, by1, by2 = b
    return max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))

def main():
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", default=str(repo / "HRSID" / "HRSID_JPG" / "annotations"))
    ap.add_argument("--out", default=str(repo / "leakage_figure.png"))
    args = ap.parse_args()

    tr = json.load(open(Path(args.ann_dir) / "train2017.json"))
    te = json.load(open(Path(args.ann_dir) / "test2017.json"))
    trf = [i["file_name"] for i in tr["images"]]
    tef = [i["file_name"] for i in te["images"]]

    tr_s, te_s = defaultdict(list), defaultdict(list)
    for fn in trf:
        s, w = parse(fn)
        if w:
            tr_s[s].append(w)
    for fn in tef:
        s, w = parse(fn)
        if w:
            te_s[s].append(w)

    shared = set(tr_s) & set(te_s)
    checked = leaked = 0
    for s in shared:
        for tw in te_s[s]:
            checked += 1
            if any(overlap(tw, rw) > 0 for rw in tr_s[s]):
                leaked += 1
    pct = 100 * leaked / len(tef)
    print(f"train scenes={len(tr_s)} test scenes={len(te_s)} shared={len(shared)}")
    print(f"test tiles overlapping a train tile: {leaked}/{len(tef)} ({pct:.1f}%)")

    scene = max(shared, key=lambda s: min(len(tr_s[s]), len(te_s[s])))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.5),
                                   gridspec_kw={"width_ratios": [1.4, 1]})

    allw = tr_s[scene] + te_s[scene]
    maxx = max(w[1] for w in allw)
    maxy = max(w[3] for w in allw)
    for (x1, x2, y1, y2) in tr_s[scene]:
        axA.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                      facecolor="#4C72B0", edgecolor="#1f3b66", alpha=0.35, lw=0.8))
    for (x1, x2, y1, y2) in te_s[scene]:
        axA.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                      facecolor="#C44E52", edgecolor="#7a2b2e", alpha=0.45, lw=0.8))
    axA.set_xlim(0, maxx)
    axA.set_ylim(maxy, 0)
    axA.set_aspect("equal")
    axA.set_title(f"Scene {scene}: train (blue) vs test (red) tiles\n"
                  f"overlap (purple) = leaked content", fontsize=11, fontweight="bold")
    axA.set_xlabel("x (px)")
    axA.set_ylabel("y (px)")
    axA.legend(handles=[patches.Patch(color="#4C72B0", alpha=0.35, label=f"train ({len(tr_s[scene])})"),
                        patches.Patch(color="#C44E52", alpha=0.45, label=f"test ({len(te_s[scene])})")],
               loc="upper right", fontsize=9)

    axB.bar(["leaked", "clean"], [pct, 100 - pct],
            color=["#C44E52", "#55A868"], edgecolor="black")
    axB.set_ylim(0, 100)
    axB.set_ylabel("% of test tiles")
    axB.set_title("HRSID official split:\ntest tiles sharing pixels with train",
                  fontsize=11, fontweight="bold")
    for i, v in enumerate([pct, 100 - pct]):
        axB.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
    axB.text(0.5, -14, f"{len(shared)}/{len(te_s)} test scenes also appear in train",
             ha="center", transform=axB.transData, fontsize=9, color="#444")

    plt.suptitle("Train/Test Leakage in HRSID's Official Split", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")

if __name__ == "__main__":
    main()
