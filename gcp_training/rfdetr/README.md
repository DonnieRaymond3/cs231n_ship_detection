# RF-DETR on HRSID — GCP training

Fine-tune **RF-DETR** (Roboflow, ICLR 2026) on HRSID — a DINOv2-backbone DETR
that's SOTA on COCO, designed for fine-tuning, and essentially unused on SAR.
The self-supervised DINOv2 backbone transfers well to out-of-domain imagery,
giving a real shot at beating DEIM. Third model in the comparison: RT-DETR
(baseline) → DEIM → RF-DETR.

Much simpler than the DEIM pipeline: `pip install rfdetr`, native W&B, and
automatic class detection (HRSID's `category_id=1` is handled for you).

## Files
| File | Where | Purpose |
|------|-------|---------|
| `setup_rfdetr.sh` | VM | venv + `pip install rfdetr wandb` + build dataset |
| `prep_rfdetr_data.py` | VM | HRSID COCO → RF-DETR's train/valid/test layout |
| `train_rfdetr.py` | VM | fine-tune with native W&B |
| `../make_comparison.py` | laptop | grand bar chart: all models vs. baselines |

## Run it
```bash
# laptop — make a VM (reuse the creator; note the zone you're in):
cd gcp_training && ZONE=us-central1-c ./create_vm.sh

# VM — clone, set up:
gcloud compute ssh rtdetr-hrsid --zone=us-central1-c
git clone https://github.com/DonnieRaymond3/cs231n_ship_detection.git 2>/dev/null || \
  (cd cs231n_ship_detection && git pull)
cd cs231n_ship_detection/gcp_training/rfdetr
bash setup_rfdetr.sh

# VM — train:
source ~/.venv-rfdetr/bin/activate
tmux new -s rfdetr
wandb login
python train_rfdetr.py --model medium --epochs 50
```

Knobs: `--model nano|small|medium|large` (medium is a strong balance; large is
heaviest/most accurate), `--epochs`, `--batch` (default 4) / `--grad-accum`
(default 4 → effective 16), `--resolution N` (must be ÷64; default is the
model's native, e.g. medium=576 — try `--resolution 768` for a high-res run
comparable to DEIM@800). `--no-wandb` to disable logging.

Outputs (checkpoints + metrics) go to `runs_rfdetr/<run>/`.

## Download + compare (laptop)
```bash
cd /Users/nishank/cs231n_ship_detection
gcloud compute scp --recurse \
  rtdetr-hrsid:~/cs231n_ship_detection/runs_rfdetr/rfdetr_medium_hrsid_50ep \
  checkpoints_rfdetr/ --zone=us-central1-c

# grand comparison chart (fill in your real numbers):
python3 gcp_training/make_comparison.py \
  --model "RT-DETR (ours)" 0.833 0.544 \
  --model "DEIM (ours)"    0.936 0.718 \
  --model "RF-DETR (ours)" 0.94  0.73 \
  --out comparison.png
```

## Notes
- **Fairness:** like DEIM, RF-DETR has its own native resolution. For the
  cleanest head-to-head, report the resolution each model used, or align them
  (DEIM@800 ↔ RF-DETR@768).
- W&B metrics (incl. per-class AP) stream live; final mAP is on the run page and
  in the saved checkpoint. RF-DETR saves `checkpoint_best_*.pth` in the output dir.
- Don't forget to delete the VM when done (`gcloud compute instances delete ...`).
