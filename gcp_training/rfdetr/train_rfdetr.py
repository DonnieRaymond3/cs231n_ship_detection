import argparse
from pathlib import Path

MODELS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(repo_root / "datasets" / "HRSID_RFDETR"))
    ap.add_argument("--model", default="medium", choices=list(MODELS))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4, help="effective batch = batch * grad_accum")
    ap.add_argument("--resolution", type=int, default=None,
                    help="Input size (must be divisible by 64). Default: model's native.")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--wandb-project", default="rfdetr-hrsid")
    ap.add_argument("--run", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--no-early-stopping", action="store_true")
    ap.add_argument("--progress-bar", default="rich", choices=["rich", "tqdm", "none"],
                    help="Live progress bar + ETA (default rich).")
    args = ap.parse_args()

    run_name = args.run or f"rfdetr_{args.model}_hrsid_{args.epochs}ep"
    out_dir = args.output_dir or str(repo_root / "runs_rfdetr" / run_name)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            if wandb.api.api_key is None:
                print("WARNING: not logged into W&B -> disabling. Run `wandb login`.")
                use_wandb = False
        except ImportError:
            use_wandb = False

    import rfdetr
    model = getattr(rfdetr, MODELS[args.model])()

    train_kwargs = dict(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        output_dir=out_dir,
        early_stopping=not args.no_early_stopping,
        tensorboard=True,
        wandb=use_wandb,
        project=args.wandb_project,
        run=run_name,
        progress_bar=None if args.progress_bar == "none" else args.progress_bar,
    )
    if args.resolution is not None:
        if args.resolution % 64 != 0:
            raise SystemExit(f"--resolution must be divisible by 64 (got {args.resolution}); "
                             f"try 576, 640, 704, or 768.")
        train_kwargs["resolution"] = args.resolution

    print(f"Training {MODELS[args.model]} on {args.dataset_dir}")
    print(f"  effective batch = {args.batch} x {args.grad_accum} = {args.batch * args.grad_accum}")
    print(f"  output -> {out_dir}  | wandb={use_wandb}")
    model.train(**train_kwargs)

    print("\nDone. Checkpoints + metrics in:", out_dir)

if __name__ == "__main__":
    main()
