# RT-DETR on HRSID — GCP training

Train RT-DETR-L on the HRSID SAR ship dataset (50 epochs) on a GCP GPU VM,
with Weights & Biases logging. The HRSID data is already committed in this repo
(`HRSID/HRSID_JPG/`), so the VM just clones the repo — no separate download.

## Files

| File | Where it runs | Purpose |
|------|---------------|---------|
| `create_vm.sh` | your laptop | spin up a GPU VM on GCP |
| `setup_vm.sh` | the VM | install Python deps into a venv |
| `prepare_hrsid.py` | the VM | COCO → YOLO format + dataset yaml |
| `train_rtdetr.py` | the VM | train RT-DETR-L, log to W&B |
| `requirements.txt` | the VM | pinned deps |

---

## One-time local setup

1. Install the gcloud CLI and log in:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```
2. **GPU quota** — new accounts often have 0 GPU quota. Check / request it at
   *IAM & Admin → Quotas*, filter for `GPUs (all regions)` (and the specific
   `NVIDIA L4 GPUs`), and request at least 1. Approval is usually quick.
3. Get your **Weights & Biases API key** from https://wandb.ai/authorize.

> Cost: an L4 `g2-standard-8` is ~\$0.85/hr on-demand. This job runs in
> ~2–4 hrs, so roughly **\$2–4** — tiny against your \$300 credit. Add `SPOT=1`
> to `create_vm.sh` to cut that ~60%, at the risk of preemption.

---

## Run it

### 1. Create the VM (local)
```bash
cd gcp_training
chmod +x *.sh
./create_vm.sh
# or cheaper T4:  GPU_TYPE=nvidia-tesla-t4 MACHINE=n1-standard-8 ./create_vm.sh
# or spot/cheap:  SPOT=1 ./create_vm.sh
```

### 2. SSH in and clone the repo (VM)
```bash
gcloud compute ssh rtdetr-hrsid --zone=us-central1-a

# on the VM:
git clone https://github.com/DonnieRaymond3/cs231n_ship_detection.git
cd cs231n_ship_detection/gcp_training
bash setup_vm.sh
```
The repo includes the 640MB HRSID image set, so the clone takes a couple minutes.

### 3. Prepare data + train (VM)
```bash
source ~/.venv/bin/activate
cd ~/cs231n_ship_detection/gcp_training

python prepare_hrsid.py            # builds datasets/HRSID_YOLO + hrsid.yaml

wandb login                        # paste your API key (once)
python train_rtdetr.py --epochs 50
```

Tips:
- Run training inside `tmux` (`tmux new -s train`) so it survives SSH drops.
  Reattach with `tmux attach -t train`.
- Watch live metrics on your W&B project page (`hrsid-rtdetr`).
- Bump `--batch 8` on an L4 (24GB) if you want it faster; drop to `--batch 2`
  if a T4 (16GB) runs out of memory.

Outputs land in `~/cs231n_ship_detection/runs/hrsid-rtdetr/rtdetr_hrsid_50ep/`
(or under `./hrsid-rtdetr/...` when W&B is on) — `weights/best.pt`,
`weights/last.pt`, `results.csv`, and plots.

---

## Get the checkpoints back (local)

RT-DETR-L checkpoints are ~63MB each (best + last ≈ 130MB total), so just pull
them to your laptop — no cloud bucket needed.

```bash
# from your laptop, in the repo root:
RUN=hrsid-rtdetr/rtdetr_hrsid_50ep      # adjust if W&B was off (then it's runs/...)
mkdir -p checkpoints
gcloud compute scp --recurse \
  rtdetr-hrsid:~/cs231n_ship_detection/gcp_training/$RUN/weights \
  checkpoints/ --zone=us-central1-a
gcloud compute scp \
  rtdetr-hrsid:~/cs231n_ship_detection/gcp_training/$RUN/results.csv \
  checkpoints/ --zone=us-central1-a
```

(W&B also stores metrics/plots online, and you can enable model artifact upload
there if you ever want a cloud copy.)

---

## Stop paying

```bash
# stop (keeps the disk, cheap to resume later):
gcloud compute instances stop rtdetr-hrsid --zone=us-central1-a
# or fully delete:
gcloud compute instances delete rtdetr-hrsid --zone=us-central1-a
```
A running GPU VM bills by the second even while idle — stop it as soon as
training + scp are done.
