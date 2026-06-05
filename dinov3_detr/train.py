"""Train DINOv2/DINOv3 + DETR on HRSID (COCO format).

Examples:
    python train.py --epochs 12 --batch-size 4
    python train.py --limit 50 --epochs 1   # quick smoke test

``run_training`` is importable so the Modal app can call it on a GPU.
"""

import argparse
import os


def run_training(
    backbone,
    pretrained_detr,
    image_processor,
    train_images,
    train_ann,
    val_images,
    val_ann,
    output_dir,
    epochs=12,
    batch_size=4,
    lr=1e-4,
    weight_decay=1e-4,
    warmup_steps=300,
    freeze_backbone=True,
    limit=0,
    wandb_project=None,
    wandb_run_name=None,
    compute_epoch_map=True,
    map_limit=0,
):
    # Heavy deps are imported lazily so this module imports cleanly without a
    # full ML stack (e.g. for py_compile or on the Modal client side).
    import torch
    from transformers import AutoImageProcessor, Trainer, TrainerCallback, TrainingArguments

    from dataset import HRSIDDetectionDataset, make_collate_fn
    from evaluate import evaluate_model_on_coco
    from model import build_model

    # Weights & Biases: the HF Trainer auto-logs train/val loss + lr when
    # report_to includes "wandb". A shared run id lets the separate evaluation
    # step attach SSDD metrics to the same run.
    report_to = "none"
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
        if wandb_run_name:
            os.environ.setdefault("WANDB_RUN_ID", wandb_run_name)
            os.environ.setdefault("WANDB_RESUME", "allow")
        report_to = ["wandb"]

    print(f"Backbone: {backbone}", flush=True)
    print(f"Pretrained DETR init: {pretrained_detr or 'none'}", flush=True)
    print(f"Image processor: {image_processor}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    processor = AutoImageProcessor.from_pretrained(image_processor)

    train_ds = HRSIDDetectionDataset(train_images, train_ann)
    val_ds = HRSIDDetectionDataset(val_images, val_ann)
    if limit:
        train_ds.image_ids = train_ds.image_ids[:limit]
        val_ds.image_ids = val_ds.image_ids[: max(1, limit // 5)]

    model = build_model(
        backbone,
        num_labels=1,
        freeze_backbone=freeze_backbone,
        pretrained_detr_name=pretrained_detr,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: total={total_params:,} trainable={trainable_params:,}", flush=True)
    if wandb_project:
        try:
            import wandb

            if wandb.run is None:
                wandb.init(project=wandb_project, id=wandb_run_name, name=wandb_run_name, resume="allow")
            wandb.config.update(
                {
                    "backbone": backbone,
                    "pretrained_detr": pretrained_detr,
                    "image_processor": image_processor,
                    "freeze_backbone": freeze_backbone,
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                },
                allow_val_change=True,
            )
        except Exception as exc:
            print(f"W&B config logging failed: {exc}", flush=True)

    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        dataloader_pin_memory=False,
        fp16=use_cuda,  # fp16 only on CUDA; CPU/MPS run fp32
        remove_unused_columns=False,
        # Custom DINO backbone adapter checkpoints reload cleanly at the end via
        # trainer.save_model(); avoid a noisy intermediate reload with adapter
        # key-name warnings.
        load_best_model_at_end=False,
        report_to=report_to,
        run_name=wandb_run_name,
    )

    callbacks = []
    if compute_epoch_map:
        class EpochCOCOMapCallback(TrainerCallback):
            def on_epoch_end(self, args, state, control, **kwargs):
                print(f"Running COCO mAP on validation split at epoch {state.epoch:.3f}...")
                evaluate_model_on_coco(
                    model=kwargs["model"],
                    processor=processor,
                    images=val_images,
                    ann=val_ann,
                    output_dir=output_dir,
                    threshold=0.0,
                    label_to_cat=1,  # HRSID category_id for ship.
                    limit=map_limit,
                    batch_size=batch_size,
                    device=next(kwargs["model"].parameters()).device,
                    metric_prefix="val",
                    wandb_project=wandb_project,
                    wandb_run_name=wandb_run_name,
                    wandb_step=state.global_step,
                )
                return control

        callbacks.append(EpochCOCOMapCallback())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=make_collate_fn(processor),
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved model + image processor to {output_dir}")
    return output_dir


def parse_args():
    from config import DEFAULTS

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default=DEFAULTS["backbone"])
    p.add_argument("--pretrained-detr", default=DEFAULTS["pretrained_detr"],
                   help="checkpoint used to initialize DETR transformer/query/bbox pieces; empty disables")
    p.add_argument("--image-processor", default=DEFAULTS["image_processor"])
    p.add_argument("--train-images", default=DEFAULTS["train_images"])
    p.add_argument("--train-ann", default=DEFAULTS["train_ann"])
    p.add_argument("--val-images", default=DEFAULTS["val_images"])
    p.add_argument("--val-ann", default=DEFAULTS["val_ann"])
    p.add_argument("--output-dir", default=DEFAULTS["output_dir"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--warmup-steps", type=int, default=DEFAULTS["warmup_steps"])
    p.add_argument("--no-freeze-backbone", action="store_true",
                   help="fine-tune the DINO backbone too (not recommended on SAR)")
    p.add_argument("--limit", type=int, default=0, help="cap dataset size for debugging")
    p.add_argument("--wandb-project", default=DEFAULTS["wandb_project"])
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-wandb", action="store_true", help="disable W&B logging")
    p.add_argument("--no-epoch-map", action="store_true", help="disable per-epoch COCO mAP on HRSID val")
    p.add_argument("--map-limit", type=int, default=0, help="cap val images for per-epoch mAP debugging")
    return p.parse_args()


def main():
    args = parse_args()
    run_training(
        backbone=args.backbone,
        pretrained_detr=args.pretrained_detr or None,
        image_processor=args.image_processor,
        train_images=args.train_images,
        train_ann=args.train_ann,
        val_images=args.val_images,
        val_ann=args.val_ann,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        freeze_backbone=not args.no_freeze_backbone,
        limit=args.limit,
        wandb_project=None if args.no_wandb else args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        compute_epoch_map=not args.no_epoch_map,
        map_limit=args.map_limit,
    )


if __name__ == "__main__":
    main()
