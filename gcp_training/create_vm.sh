#!/usr/bin/env bash
set -euo pipefail

VM_NAME="${VM_NAME:-rtdetr-hrsid}"
ZONE="${ZONE:-us-central1-a}"
MACHINE="${MACHINE:-g2-standard-8}"          # L4 host. For T4 use n1-standard-8.
GPU_TYPE="${GPU_TYPE:-nvidia-l4}"            # or nvidia-tesla-t4
GPU_COUNT="${GPU_COUNT:-1}"
BOOT_DISK_GB="${BOOT_DISK_GB:-100}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
SPOT="${SPOT:-0}"

ACCEL="type=${GPU_TYPE},count=${GPU_COUNT}"
EXTRA=()
if [[ "$SPOT" == "1" ]]; then
  EXTRA=(--provisioning-model=SPOT --instance-termination-action=STOP)
fi

echo "Creating VM '$VM_NAME' in $ZONE ($MACHINE, $GPU_COUNT x $GPU_TYPE, spot=$SPOT)"
gcloud compute instances create "$VM_NAME" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --accelerator="$ACCEL" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="${BOOT_DISK_GB}GB" \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True" \
  ${EXTRA[@]+"${EXTRA[@]}"}

echo
echo "VM created. SSH in with:"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE"
echo
echo "REMEMBER to stop/delete it when done so you stop paying:"
echo "  gcloud compute instances stop   $VM_NAME --zone=$ZONE   # keeps disk, cheap"
echo "  gcloud compute instances delete $VM_NAME --zone=$ZONE   # removes everything"
