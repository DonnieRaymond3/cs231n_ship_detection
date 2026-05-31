"""Evaluate a trained DINO + DETR model with COCO mAP.

By default this evaluates on the full SSDD dataset (cross-domain test) using
SSDD's category_id 0. Override --ann / --images / --label-to-cat for HRSID.

Examples:
    python evaluate.py --model-dir outputs                 # SSDD (default)
    python evaluate.py --model-dir outputs \\
        --ann ../HRSID/HRSID_JPG/annotations/test2017.json \\
        --images ../HRSID/HRSID_JPG/JPEGImages --label-to-cat 1

``run_eval`` is importable so the Modal app can call it on a GPU.
"""

import argparse
import json
import os


def pick_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


COCO_STAT_NAMES = [
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
]


def load_trained_model(model_dir, backbone, pretrained_detr=None, device=None):
    """Load a checkpoint saved from our custom DINO+DETR adapter model."""
    import torch

    from model import build_model

    device = device or pick_device()
    model = build_model(
        backbone,
        num_labels=1,
        freeze_backbone=True,
        pretrained_detr_name=pretrained_detr,
    )

    safetensors_path = os.path.join(model_dir, "model.safetensors")
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.isfile(safetensors_path):
        from safetensors.torch import load_file

        state = load_file(safetensors_path)
    elif os.path.isfile(bin_path):
        state = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {model_dir}")

    target_state = model.state_dict()
    remapped = {}
    prefix_map = {
        # Some Transformers save_pretrained paths serialize custom backbone
        # adapters under the original DETR conv-encoder namespace.
        "model.backbone.conv_encoder.model.": "model.backbone.backbone.",
        "model.backbone.conv_encoder.backbone.": "model.backbone.backbone.model.",
    }
    for key, value in state.items():
        new_key = key
        for old_prefix, target_prefix in prefix_map.items():
            if key.startswith(old_prefix):
                candidate = target_prefix + key[len(old_prefix):]
                if candidate in target_state and target_state[candidate].shape == value.shape:
                    new_key = candidate
                break
        if new_key == key:
            candidate = key
            candidate = candidate.replace(".self_attn.out_proj.", ".self_attn.o_proj.")
            candidate = candidate.replace(".encoder_attn.out_proj.", ".encoder_attn.o_proj.")
            candidate = candidate.replace(".fc1.", ".mlp.fc1.")
            candidate = candidate.replace(".fc2.", ".mlp.fc2.")
            if candidate in target_state and target_state[candidate].shape == value.shape:
                new_key = candidate
        remapped[new_key] = value

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        print(
            f"Loaded checkpoint with {len(missing)} missing and {len(unexpected)} unexpected keys.",
            flush=True,
        )
    return model.to(device).eval()


def evaluate_model_on_coco(
    model,
    processor,
    images,
    ann,
    output_dir,
    threshold=0.0,
    label_to_cat=0,
    limit=0,
    batch_size=8,
    device=None,
    metric_prefix="val",
    wandb_project=None,
    wandb_run_name=None,
    wandb_step=None,
):
    """Run COCO bbox evaluation for an in-memory DETR model."""
    import torch
    from PIL import Image
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    device = device or pick_device()
    was_training = model.training
    model.eval()

    with open(ann) as f:
        coco = json.load(f)
    img_list = coco["images"]
    if limit:
        img_list = img_list[:limit]

    results = []
    with torch.no_grad():
        for start in range(0, len(img_list), batch_size):
            batch_infos = img_list[start: start + batch_size]
            batch_images = [
                Image.open(os.path.join(images, info["file_name"])).convert("RGB")
                for info in batch_infos
            ]
            inputs = processor(images=batch_images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            target_sizes = torch.tensor(
                [[info["height"], info["width"]] for info in batch_infos],
                device=device,
            )
            processed_batch = processor.post_process_object_detection(
                outputs, threshold=threshold, target_sizes=target_sizes
            )

            for info, processed in zip(batch_infos, processed_batch):
                for score, box in zip(processed["scores"], processed["boxes"]):
                    x1, y1, x2, y2 = box.tolist()
                    results.append(
                        {
                            "image_id": int(info["id"]),
                            "category_id": int(label_to_cat),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score),
                        }
                    )

    if was_training:
        model.train()

    if not results:
        print(f"No detections produced for {metric_prefix} evaluation.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    pred_file = os.path.join(output_dir, f"{metric_prefix}_predictions.json")
    with open(pred_file, "w") as f:
        json.dump(results, f)

    coco_gt = COCO(ann)
    coco_dt = coco_gt.loadRes(pred_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if limit:
        coco_eval.params.imgIds = [int(i["id"]) for i in img_list]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats.tolist()

    metrics = {f"{metric_prefix}/{n}": v for n, v in zip(COCO_STAT_NAMES, stats)}
    if wandb_project:
        import wandb

        if wandb.run is None:
            wandb.init(project=wandb_project, id=wandb_run_name, name=wandb_run_name, resume="allow")
        wandb.log(metrics, step=wandb_step)

    return stats


def run_eval(model_dir, images, ann, threshold=0.0, label_to_cat=0, limit=0,
             wandb_project=None, wandb_run_name=None, metric_prefix="ssdd",
             backbone=None, pretrained_detr=None):
    from transformers import AutoImageProcessor

    from config import DEFAULTS

    device = pick_device()
    processor = AutoImageProcessor.from_pretrained(model_dir)
    # Checkpoints from this project use a custom backbone adapter, so load the
    # model through build_model() instead of vanilla from_pretrained().
    model = load_trained_model(
        model_dir,
        backbone or DEFAULTS["backbone"],
        pretrained_detr=pretrained_detr,
        device=device,
    )
    stats = evaluate_model_on_coco(
        model=model,
        processor=processor,
        images=images,
        ann=ann,
        output_dir=model_dir,
        threshold=threshold,
        label_to_cat=label_to_cat,
        limit=limit,
        batch_size=8,
        device=device,
        metric_prefix=metric_prefix,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )
    return stats


def parse_args():
    from config import DEFAULTS

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default=DEFAULTS["output_dir"])
    p.add_argument("--images", default=DEFAULTS["test_images"])
    p.add_argument("--ann", default=DEFAULTS["test_ann"])
    p.add_argument("--threshold", type=float, default=0.0,
                   help="keep detections above this score (0.0 for full mAP)")
    p.add_argument("--label-to-cat", type=int, default=DEFAULTS["test_category_id"],
                   help="COCO category_id assigned to DETR label 0 (SSDD=0, HRSID=1)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--wandb-project", default=DEFAULTS["wandb_project"])
    p.add_argument("--wandb-run-name", default=None,
                   help="attach metrics to this W&B run id (same as training run)")
    p.add_argument("--no-wandb", action="store_true", help="disable W&B logging")
    p.add_argument("--metric-prefix", default="ssdd")
    p.add_argument("--backbone", default=DEFAULTS["backbone"])
    p.add_argument("--pretrained-detr", default=DEFAULTS["pretrained_detr"])
    return p.parse_args()


def main():
    args = parse_args()
    run_eval(
        model_dir=args.model_dir,
        images=args.images,
        ann=args.ann,
        threshold=args.threshold,
        label_to_cat=args.label_to_cat,
        limit=args.limit,
        wandb_project=None if args.no_wandb else args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        metric_prefix=args.metric_prefix,
        backbone=args.backbone,
        pretrained_detr=args.pretrained_detr or None,
    )


if __name__ == "__main__":
    main()
