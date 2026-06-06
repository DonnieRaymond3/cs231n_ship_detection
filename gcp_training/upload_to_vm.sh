#!/usr/bin/env bash
set -euo pipefail

VM_NAME="${VM_NAME:-rtdetr-hrsid}"
ZONE="${ZONE:-us-central1-a}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"          # .../gcp_training/
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"             # .../cs231n_ship_detection-master/
HRSID_LOCAL="$REPO_ROOT/HRSID"                        # has HRSID_JPG/ inside
DEIM_SCRIPTS="$SCRIPT_DIR/deim"

REMOTE_REPO="$HOME/cs231n_ship_detection"             # NOTE: $HOME here = YOUR laptop home
REMOTE="$VM_NAME"

gcloud_scp() {
  gcloud compute scp "$@" --zone="$ZONE"
}

echo "==> Creating directory structure on VM"
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
  mkdir -p ~/cs231n_ship_detection/HRSID
  mkdir -p ~/cs231n_ship_detection/gcp_training/deim
"

echo "==> Uploading HRSID data (~645 MB, takes 3-8 min on typical broadband)"
gcloud_scp --recurse "$HRSID_LOCAL" "$VM_NAME":~/cs231n_ship_detection/

echo "==> Uploading DEIM training scripts"
gcloud_scp --recurse "$DEIM_SCRIPTS/." "$VM_NAME":~/cs231n_ship_detection/gcp_training/deim/

echo "==> Uploading setup_vm.sh"
gcloud_scp "$SCRIPT_DIR/setup_vm.sh" "$VM_NAME":~/cs231n_ship_detection/gcp_training/

echo
echo "Upload complete. SSH in and run:"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "On the VM:"
echo "  cd ~/cs231n_ship_detection/gcp_training/deim"
echo "  MODEL=s bash setup_deim.sh        # ~3 min: install deps + download weights"
echo "  source ~/.venv-deim/bin/activate"
echo "  tmux new -s deim"
echo "  wandb login                       # optional — skip with NO_WANDB=1"
echo "  MODEL=s EPOCHS=50 bash run_deim.sh"
