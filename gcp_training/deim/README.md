# DEIM (D-FINE) on HRSID — GCP training

Fine-tune **DEIM** (DETR with Improved Matching, CVPR 2025, on the D-FINE
backbone) on HRSID — a SOTA real-time detector with almost no published SAR
ship-detection work, so it's a strong "novel application" angle next to your
RT-DETR baseline.

**Why DEIM for SAR:** distribution-based box regression (better localization of
tiny ships → directly lifts mAP@50-95), no NMS (cleaner dense-harbor scenes),
and fast convergence (more model per GPU-dollar). It trains on **COCO-format**
annotations, which HRSID already ships — *no COCO→YOLO conversion needed*.

## Files

| File | Where | Purpose |
|------|-------|---------|
| `setup_deim.sh` | VM | clone DEIM, venv, deps + wandb, download pretrained ckpt |
| `gen_config.py` | VM | write HRSID config into the DEIM repo |
| `train_deim_wandb.py` | VM | run DEIM training with live W&B (TensorBoard sync) |
| `run_deim.sh` | VM | orchestrates gen_config + training |
| `make_plots_deim.py` | laptop | training curves + DEIM-vs-RT-DETR-vs-literature chart |

---

## Run it

### 1. Create a GPU VM (laptop)
Reuse the RT-DETR creator (same box works fine):
```bash
cd gcp_training
./create_vm.sh          # L4, us-central1-a
```

### 2. SSH in, clone repo, set up (VM)
```bash
gcloud compute ssh rtdetr-hrsid --zone=us-central1-a

git clone https://github.com/DonnieRaymond3/cs231n_ship_detection.git 2>/dev/null || \
  (cd cs231n_ship_detection && git pull)
cd cs231n_ship_detection/gcp_training/deim
MODEL=s bash setup_deim.sh        # n|s|m|l|x — s is fast+strong; l ≈ RT-DETR-L scale
```

### 3. Train (VM)
```bash
source ~/.venv-deim/bin/activate
tmux new -s deim
wandb login                       # paste key
MODEL=s EPOCHS=50 bash run_deim.sh
```
You'll see `W&B live logging -> project 'deim-hrsid'` and a run link. Metrics
stream to wandb.ai live (DEIM's TensorBoard scalars are mirrored automatically).

Knobs (env vars): `MODEL` (n/s/m/l/x), `EPOCHS` (50), `BATCH` (32 — drop to 16
if you hit OOM), `NO_WANDB=1` to disable W&B.

Outputs land in `~/DEIM/outputs/deim_hrsid_<size>/`:
- `log.txt` — per-epoch COCO metrics
- `best_stg1.pth` / `best_stg2.pth` / `last.pth` — checkpoints
- `summary/` — TensorBoard event files

---

## Download checkpoints + plots (laptop)
```bash
cd /Users/nishank/cs231n_ship_detection
mkdir -p checkpoints_deim
gcloud compute scp --recurse \
  rtdetr-hrsid:~/DEIM/outputs/deim_hrsid_s \
  checkpoints_deim/ --zone=us-central1-a
```

Generate the report plots locally (including the baseline comparison you liked,
now with your RT-DETR result alongside DEIM). Pass your RT-DETR numbers:
```bash
python3 gcp_training/deim/make_plots_deim.py \
  --log checkpoints_deim/deim_hrsid_s/log.txt \
  --rtdetr-map50 0.90 --rtdetr-map5095 0.65
```
→ writes `training_curves.png` and `baseline_comparison.png` next to the log.

---

## Stop the VM when done
```bash
gcloud compute instances delete rtdetr-hrsid --zone=us-central1-a
```

---

## Notes / gotchas
- **`num_classes: 2`** is intentional. HRSID's one category has `id=1`, and DEIM
  (with `remap_mscoco_category: False`) uses the raw id as the label index, so a
  1-class head would be out of range. Index 0 is simply unused.
- **Resolution:** DEIM trains at 640×640 (vs. your RT-DETR run at 800). It's the
  config's tuned size and guarantees a clean first run. To match 800 exactly,
  see DEIM's "Customizing Input Size" section in their README (edit
  `eval_spatial_size` + base size) — worth trying later since HRSID ships are tiny.
- **Schedule:** `gen_config.py` rescales DEIM's 132-epoch aug/LR recipe to your
  epoch count using their own formula (`flat_epoch = 4 + (epochs−no_aug)//2`).
- **Fair comparison:** for an apples-to-apples table against RT-DETR-L, train
  `MODEL=l` (31M params, ≈ RT-DETR-L scale). `s` (10M) is cheaper and still strong.
