"""
train.py — AttGAN training entry point.

python train.py                                   # default config
python train.py --exp exp1_baseline               # experiment 1
python train.py --exp exp2_high_rec               # experiment 2
python train.py --exp exp3_strong_attr            # experiment 3
python train.py --exp exp1_baseline --epochs 10   # override epochs
python train.py --exp exp1_baseline --resume checkpoints/exp1_baseline/ckpt_epoch010.pt
python train.py --exp exp1_baseline --eval-only  --resume checkpoints/exp1_baseline/ckpt_epoch030.pt
python train.py --exp exp1_baseline --no-metrics  # skip FID/DACID
"""

import argparse
import importlib
import torch

from config      import Config
from src.dataset import get_loaders
from src.models  import build_models
from src.trainer import Trainer
from src.utils   import (plot_losses, attribute_demo,
                          evaluate_attribute_accuracy,
                          evaluate_reconstruction)

_EXP_MAP = {
    "exp1_baseline":    ("experiments.exp1_baseline", "Exp1Config"),
    "exp2_high_rec":    ("experiments.exp2_high_rec", "Exp2Config"),
    "exp3_strong_attr": ("experiments.exp3_low_rec",  "Exp3Config"),
}


def _load_cfg(exp_name):
    if exp_name is None:
        return Config()
    mod_path, cls_name = _EXP_MAP[exp_name]
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp",        type=str, default=None,
                   choices=list(_EXP_MAP.keys()))
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--batch",      type=int, default=None)
    p.add_argument("--resume",     type=str, default=None)
    p.add_argument("--eval-only",  action="store_true")
    p.add_argument("--no-metrics", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = _load_cfg(args.exp)
    if args.epochs:     cfg.N_EPOCHS       = args.epochs
    if args.batch:      cfg.BATCH_SIZE     = args.batch
    if args.no_metrics: cfg.COMPUTE_METRICS = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Experiment : {cfg.EXPERIMENT_NAME}")
    print(f"Device     : {device}")
    print(f"lambda_rec : {cfg.LAMBDA_REC}  "
          f"lambda_cls_G : {cfg.LAMBDA_CLS_G}")

    train_loader, test_loader = get_loaders(cfg)
    enc, gen, dis = build_models(cfg, device)

    trainer = Trainer(enc, gen, dis, train_loader, test_loader, cfg, device)

    if args.eval_only:
        if not args.resume:
            raise ValueError("--eval-only requires --resume <path>")
        from src.utils import load_checkpoint
        load_checkpoint(args.resume, enc, gen, dis)
    else:
        g_losses, d_losses = trainer.train(resume_path=args.resume)
        plot_losses(g_losses, d_losses, cfg)

    print("\nRunning evaluation...")
    evaluate_attribute_accuracy(enc, gen, dis, test_loader, cfg)
    evaluate_reconstruction(enc, gen, test_loader, cfg)
    attribute_demo(enc, gen, test_loader, cfg)
    print(f"\nDone. Results -> {cfg.RESULTS_DIR}")


if __name__ == "__main__":
    main()
