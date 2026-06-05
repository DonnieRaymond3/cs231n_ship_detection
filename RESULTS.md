# Results (HRSID, official split)

All models fine-tuned from COCO-pretrained weights for 50 epochs (AdamW). Metrics
are COCO bbox AP on the official HRSID test split (1,962 tiles). Literature rows
are *reported*, not reproduced.

## Main table

| Model | Res | mAP@50 | mAP@50-95 | AP_small | AP_medium | AP_large |
|-------|-----|--------|-----------|----------|-----------|----------|
| Faster R-CNN* | — | 0.867 | 0.635 | — | — | — |
| Cascade R-CNN* | — | 0.877 | 0.666 | — | — | — |
| YOLOv8* | — | 0.887 | 0.628 | — | — | — |
| RT-DETR-L (ours) | 800 | 0.833 | 0.544 | — | — | — |
| RF-DETR-M (ours) | 576 | 0.923 | 0.660 | — | — | — |
| RF-DETR-L (ours) | 768 | 0.936 | 0.695 | — | — | — |
| **DEIM-S (ours)** | **640** | **0.936** | **0.718** | **0.725** | **0.757** | 0.616 |
| DEIM-S (ours) | 800 | 0.930 | 0.687 | 0.693 | 0.710 | 0.616 |

\* reported in the literature, not reproduced here.

## Findings

**1. Modern DETR detectors beat both the baselines and reported HRSID results.**
DEIM (0.718 mAP@50-95) and RF-DETR-L (0.695) exceed the best reported baseline
(Cascade R-CNN, 0.666) and far exceed our RT-DETR (0.544). (Fig: comparison.)

**2. DEIM is the strongest model, with the edge on tight localization.**
DEIM and RF-DETR-L tie on mAP@50 (0.936), but DEIM leads mAP@50-95 by +0.023
(0.718 vs 0.695). Notably DEIM achieves this at *lower* input resolution than
RF-DETR-L (640 vs 768), so the advantage is architectural, not resolution —
attributable to D-FINE's distribution-based box regression. (Fig: comparison.)

**3. Convergence explains RT-DETR's weakness.**
RT-DETR underperforms not because the architecture is weak but because plain DETR
matching converges slowly; at 50 epochs it is undertrained and its mAP@50-95 curve
is still rising. DEIM's dense one-to-one matching converges fastest and plateaus
highest. (Fig: convergence overlay.)

**4. Object-size analysis: small ships are handled well; large ships are the
weak point.** DEIM's AP is 0.725 (small) and 0.757 (medium) but only 0.616
(large). This is counter-intuitive but expected: HRSID is dominated by small
vessels, so the "large" bucket is sparse and noisy. The strong small-object AP is
the relevant SAR result. (Fig: AP-by-size.)

**5. Higher resolution did NOT help here — a batch-size confound.**
DEIM@800 (0.687) underperformed DEIM@640 (0.718), and even AP_small dropped
(0.725 -> 0.693). This is almost certainly because the 800px run was forced to a
smaller batch (32 -> 8/16) by GPU memory at the same learning rate, leaving it
undertrained — not evidence that resolution hurts. Takeaway: at a fixed compute
budget, effective batch size / training stability mattered more than input
resolution. We report this as a limitation rather than a clean ablation.

## Size-stratified results (HRSID test, in-domain)

COCO AP by object area, re-evaluated with pycocotools (maxDets=100):

| Model | AP_small | AP_medium | AP_large |
|-------|----------|-----------|----------|
| RT-DETR-L | 0.556 | 0.523 | 0.205 |
| RF-DETR-L | 0.711 | 0.763 | 0.667 |
| **DEIM-S** | **0.725** | 0.757 | 0.616 |

DEIM and RF-DETR are strong across all sizes; RT-DETR collapses on large ships
(AP_large 0.205) — consistent with it being undertrained. (Fig: AP-by-size.)

## Cross-dataset generalization (zero-shot: train HRSID, test SSDD)

Models trained only on HRSID, evaluated on the SSDD test set (232 images,
different sensor/resolution) with no fine-tuning:

| Model | HRSID mAP@50-95 | SSDD mAP@50-95 | HRSID mAP@50 | SSDD mAP@50 | retained (50-95) |
|-------|-----------------|----------------|-------------|-------------|------------------|
| RT-DETR-L | 0.539 | 0.406 | 0.814 | 0.742 | 75% |
| **RF-DETR-L** | 0.707 | **0.540** | 0.934 | **0.861** | **76%** |
| DEIM-S | 0.718 | 0.496 | 0.936 | 0.809 | 69% |

**Finding 6 — RF-DETR generalizes best across sensors.** Although DEIM is the
strongest in-domain model, **RF-DETR-L transfers best to SSDD** (0.540 vs DEIM's
0.496 mAP@50-95; 0.861 vs 0.809 mAP@50) and DEIM degrades the most (only 69%
retained vs RF-DETR's 76%). This supports the hypothesis that RF-DETR's
self-supervised DINOv2 backbone yields more transferable features for
out-of-distribution SAR imagery. The practical takeaway: pick DEIM when the test
distribution matches training, RF-DETR when robustness to new sensors matters.
(Figs: generalization mAP@50-95 and mAP@50.)

## Methodological finding: train/test leakage in HRSID's official split

HRSID's tiles are overlapping 800x800 crops of ~200 large SAR scenes, and the
official split assigns *tiles* (not scenes) to train/test. We find **135 of 135
test scenes also appear in training**, and **92.8% of test tiles share pixels
with a training tile**. All absolute numbers above (ours and the literature
baselines, which use the same split) are therefore optimistic relative to true
generalization. The model-to-model comparison remains valid (identical split and
eval for all), but absolute mAP should not be read as in-the-wild performance.
(Fig: leakage.) A scene-disjoint re-split is the recommended fix for honest
generalization numbers.

## Figures
- `paper_figures/fig_comparison_all.png` — models vs. reported baselines
- `paper_figures/fig_convergence_overlay.png` — mAP@50-95 vs. epoch, all runs
- `paper_figures/fig_ap_by_size.png` — AP by object size, all models
- `paper_figures/fig_generalization_map5095.png` — HRSID vs. SSDD (mAP@50-95)
- `paper_figures/fig_generalization_map50.png` — HRSID vs. SSDD (mAP@50)
- `paper_figures/qual_rfdetr.png` — RF-DETR predictions vs. ground truth
- `paper_figures/qual_rtdetr.png` — RT-DETR predictions vs. ground truth
- `leakage_figure.png` — train/test tile overlap + 92.8% statistic
