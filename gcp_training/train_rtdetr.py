import argparse
import os
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(repo_root / "datasets" / "HRSID_YOLO" / "hrsid.yaml"))
    ap.add_argument("--model", default="rtdetr-l.pt", help="Pretrained RT-DETR weights.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default=str(repo_root / "runs"), help="Local output dir for runs.")
    ap.add_argument("--name", default="rtdetr_hrsid_50ep", help="Run name (also the W&B run name).")
    ap.add_argument("--wandb-project", default="hrsid-rtdetr", help="W&B project name.")
    ap.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    ap.add_argument("--resume", action="store_true", help="Resume from last.pt of this run.")
    args = ap.parse_args()

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            if wandb.api.api_key is None:
                print("WARNING: not logged into W&B -> logging disabled. "
                      "Run `wandb login` (or set WANDB_API_KEY) then rerun.")
                use_wandb = False
        except ImportError:
            print("WARNING: wandb not installed -> logging disabled.")
            use_wandb = False
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    from ultralytics import RTDETR, settings

    settings.update({"wandb": use_wandb})

    model = RTDETR(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,

        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.01,
        warmup_epochs=3,
        weight_decay=1e-4,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.3,
        fliplr=0.5,
        flipud=0.3,
        degrees=15.0,

        mosaic=0.0,
        copy_paste=0.0,
        mixup=0.0,

        project=args.wandb_project if use_wandb else args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,

        patience=20,
        save=True,
        plots=True,
        verbose=True,
    )

    run_dir = Path(args.wandb_project if use_wandb else args.project) / args.name
    print("\nTraining complete.")
    print(f"  best weights: {run_dir / 'weights' / 'best.pt'}")
    print(f"  last weights: {run_dir / 'weights' / 'last.pt'}")
    print(f"  results csv:  {run_dir / 'results.csv'}")

if __name__ == "__main__":
    main()
