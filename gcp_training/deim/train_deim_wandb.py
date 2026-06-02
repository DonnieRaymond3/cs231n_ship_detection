#!/usr/bin/env python3
"""
Launch DEIM's train.py in-process with live Weights & Biases logging.

DEIM logs to TensorBoard natively. By calling wandb.init(sync_tensorboard=True)
in this same process *before* DEIM creates its SummaryWriter, every scalar DEIM
writes is mirrored to W&B live. Runs single-process (no torchrun) so the patch
applies to the one process that does the training.
"""
import argparse
import os
import runpy
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deim-root", required=True)
    ap.add_argument("--config", required=True, help="config path relative to deim-root")
    ap.add_argument("--tuning", required=True, help="COCO-pretrained .pth to fine-tune from")
    ap.add_argument("--summary-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default="deim-hrsid")
    ap.add_argument("--name", default="deim_hrsid")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    os.chdir(args.deim_root)  # train.py uses cwd-relative config + output paths

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            if wandb.api.api_key is None:
                print("WARNING: not logged into W&B -> disabling. Run `wandb login`.")
                use_wandb = False
        except ImportError:
            print("WARNING: wandb not installed -> disabling.")
            use_wandb = False

    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.name, sync_tensorboard=True)
        print(f"W&B live logging -> project '{args.wandb_project}', run '{args.name}'")

    sys.argv = [
        "train.py",
        "-c", args.config,
        "-t", args.tuning,
        "--use-amp",
        "--seed", str(args.seed),
        "--summary-dir", args.summary_dir,
        "--output-dir", args.output_dir,
    ]
    try:
        runpy.run_path(os.path.join(args.deim_root, "train.py"), run_name="__main__")
    finally:
        if use_wandb:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
