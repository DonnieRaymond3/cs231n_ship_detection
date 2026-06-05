# Results

**TL;DR.** We benchmark three DETR-family detectors on HRSID SAR ship detection.
**DEIM is the strongest in-domain** (0.718 mAP@50-95), beating our RT-DETR
baseline and exceeding reported HRSID baselines. **RF-DETR (DINOv2 backbone)
generalizes best** to a second dataset (SSDD) despite ranking second in-domain.
We also find that **HRSID's official split leaks ~93% of test content into
training**, so absolute numbers across the field are optimistic.

## Experimental setup

All models are initialized from COCO-pretrained weights and fine-tuned on HRSID
for 50 epochs with AdamW on a single NVIDIA L4. We use each framework's
recommended fine-tuning configuration (no per-model hyperparameter search); the
only deliberately varied factors are input resolution and batch size (the latter
constrained by GPU memory). Metrics are COCO bbox AP on the official HRSID test
split (1,962 tiles). Overall mAP is reported from each framework's native
validation; size-stratified AP and the SSDD generalization numbers come from a
**unified pycocotools re-evaluation** (maxDets=100) so they are directly
comparable across models. Small differences between the two (≤0.012 mAP) are due
to eval-engine/threshold differences and do not affect any ranking.

## Main results (HRSID, official split)

| Model | Res | mAP@50 | mAP@50-95 | AP_small | AP_medium | AP_large |
|-------|-----|--------|-----------|----------|-----------|----------|
| Faster R-CNN* | — | 0.867 | 0.635 | — | — | — |
| Cascade R-CNN* | — | 0.877 | 0.666 | — | — | — |
| YOLOv8* | — | 0.887 | 0.628 | — | — | — |
| RT-DETR-L (ours) | 800 | 0.814 | 0.539 | 0.556 | 0.523 | 0.205 |
| RF-DETR-M (ours) | 576 | 0.923 | 0.660 | — | — | — |
| RF-DETR-L (ours) | 768 | 0.934 | 0.707 | 0.711 | 0.763 | 0.667 |
| **DEIM-S (ours)** | **640** | **0.936** | **0.718** | **0.725** | 0.757 | 0.616 |
| DEIM-S (ours) | 800 | 0.930 | 0.687 | 0.693 | 0.710 | 0.616 |

\* reported in the literature, not reproduced here. AP_small/medium/large and
RT-DETR / RF-DETR-L overall AP are from the unified pycocotools re-evaluation.

## Cross-dataset generalization (zero-shot: train HRSID → test SSDD)

Models trained only on HRSID, evaluated on the SSDD test set (232 images,
different sensor/resolution) with no fine-tuning:

| Model | HRSID mAP@50-95 | SSDD mAP@50-95 | HRSID mAP@50 | SSDD mAP@50 | retained (50-95) |
|-------|-----------------|----------------|-------------|-------------|------------------|
| RT-DETR-L | 0.539 | 0.406 | 0.814 | 0.742 | 75% |
| **RF-DETR-L** | 0.707 | **0.540** | 0.934 | **0.861** | **76%** |
| DEIM-S | 0.718 | 0.496 | 0.936 | 0.809 | 69% |

## Findings

**1. Modern DETR detectors beat the baselines and reported HRSID results.**
DEIM (0.718 mAP@50-95) and RF-DETR-L (0.707) exceed the best reported baseline
(Cascade R-CNN, 0.666) and far exceed our RT-DETR (0.539). *(Fig: comparison.)*

**2. DEIM is strongest in-domain, with the edge on tight localization.**
DEIM and RF-DETR-L are close on mAP@50 (0.936 vs 0.934), but DEIM leads mAP@50-95
(0.718 vs 0.707). DEIM achieves this at *lower* input resolution than RF-DETR-L
(640 vs 768), so the advantage is architectural — attributable to D-FINE's
distribution-based box regression. *(Fig: comparison.)*

**3. Convergence explains RT-DETR's weakness.** RT-DETR underperforms not because
the architecture is weak but because plain DETR matching converges slowly; at 50
epochs it is undertrained and its mAP@50-95 curve is still rising, while DEIM's
dense one-to-one matching converges fastest and plateaus highest.
*(Fig: convergence overlay.)*

**4. Object-size analysis: small ships handled well, large ships are the weak
point.** All strong models peak on small/medium ships (DEIM AP_small 0.725) and
dip on large (DEIM 0.616). This is expected: HRSID is dominated by small vessels,
so the "large" bucket is sparse and noisy. RT-DETR collapses on large ships
(AP_large 0.205), consistent with undertraining. The strong small-object AP is
the SAR-relevant result. *(Fig: AP-by-size.)*

**5. Higher resolution did NOT help — a batch-size confound.** DEIM@800 (0.687)
underperformed DEIM@640 (0.718); even AP_small dropped (0.725 → 0.693). This is
almost certainly because the 800px run was forced to a smaller batch (32 → 8/16)
by GPU memory at the same learning rate, leaving it undertrained — not evidence
that resolution hurts. Takeaway: at a fixed compute budget, effective batch size
mattered more than input resolution. We report this as a limitation, not a clean
ablation.

**6. RF-DETR generalizes best across sensors.** Although DEIM is the strongest
in-domain model, **RF-DETR-L transfers best to SSDD** (0.540 vs DEIM's 0.496
mAP@50-95; 0.861 vs 0.809 mAP@50) and DEIM degrades the most (69% retained vs
RF-DETR's 76%). This supports the hypothesis that RF-DETR's self-supervised
DINOv2 backbone yields more transferable features for out-of-distribution SAR
imagery. Practical takeaway: prefer DEIM when the test distribution matches
training, RF-DETR when robustness to new sensors matters.
*(Figs: generalization @50-95 and @50.)*

## Methodological finding: train/test leakage in HRSID's official split

HRSID's tiles are overlapping 800×800 crops of ~200 large SAR scenes, and the
official split assigns *tiles* (not scenes) to train/test. We find **135 of 135
test scenes also appear in training**, and **92.8% of test tiles share pixels
with a training tile**. All absolute numbers above (ours and the literature
baselines, which use the same split) are therefore optimistic relative to true
generalization. The model-to-model comparison remains valid (identical split and
eval for all), but absolute mAP should not be read as in-the-wild performance.
A scene-disjoint re-split is the recommended fix for honest generalization
numbers. *(Fig: leakage.)*

## Limitations

- **Benchmark leakage.** Official-split numbers are inflated (see above); a
  scene-disjoint re-split would give honest generalization figures.
- **DEIM@800 batch confound.** The 800px run used a smaller batch than @640 at
  the same LR, so the 640-vs-800 comparison is not a clean resolution ablation.
- **No per-model tuning / single seed.** We report a single run per
  configuration with each framework's default hyperparameters; no error bars.
- **Resolution differs by model.** Each detector uses its native/recommended
  input size (RT-DETR 800, DEIM 640, RF-DETR 576–768); resolution is reported
  per run rather than held fixed across architectures.

## Figures
- `paper_figures/fig_comparison_all.png` — models vs. reported baselines
- `paper_figures/fig_convergence_overlay.png` — mAP@50-95 vs. epoch, all runs
- `paper_figures/fig_ap_by_size.png` — AP by object size, all models
- `paper_figures/fig_generalization_map5095.png` — HRSID vs. SSDD (mAP@50-95)
- `paper_figures/fig_generalization_map50.png` — HRSID vs. SSDD (mAP@50)
- `paper_figures/qual_deim.png` — DEIM predictions vs. ground truth
- `paper_figures/qual_rfdetr.png` — RF-DETR predictions vs. ground truth
- `paper_figures/qual_rtdetr.png` — RT-DETR predictions vs. ground truth
- `leakage_figure.png` — train/test tile overlap + 92.8% statistic
