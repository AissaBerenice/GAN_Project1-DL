"""
train_simple_gan.py — Simple GAN (DCGAN) training entry point.

python train_simple_gan.py
python train_simple_gan.py --epochs 20
python train_simple_gan.py --no-metrics
"""

import argparse
import json
import torch

from config          import Config
from src.dataset     import get_loaders
from src.simple_gan  import build_simple_models, train_simple_gan
from src.utils       import plot_losses

LATENT_DIM = 100


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,  default=None)
    p.add_argument("--batch",      type=int,  default=None)
    p.add_argument("--no-metrics", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()
    cfg.EXPERIMENT_NAME = "simple_gan"
    cfg.__init__()

    if args.epochs:     cfg.N_EPOCHS       = args.epochs
    if args.batch:      cfg.BATCH_SIZE     = args.batch
    if args.no_metrics: cfg.COMPUTE_METRICS = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Experiment : {cfg.EXPERIMENT_NAME}")
    print(f"Device     : {device}")

    train_loader, test_loader = get_loaders(cfg)
    gen, dis = build_simple_models(latent_dim=LATENT_DIM, device=device)

    g_losses, d_losses = train_simple_gan(
        gen, dis, train_loader, cfg, device, latent_dim=LATENT_DIM)
    plot_losses(g_losses, d_losses, cfg)

    scores = {}
    if cfg.COMPUTE_METRICS:
        from src.metrics import compute_metrics_simple_gan
        scores = compute_metrics_simple_gan(
            gen, test_loader, LATENT_DIM, cfg, device)

    # Save checkpoint
    ckpt = cfg.CHECKPOINT_DIR / "simple_gan_final.pt"
    torch.save({"gen": gen.state_dict(), "dis": dis.state_dict()}, ckpt)
    print(f"Checkpoint -> {ckpt}")

    # Save metrics.json
    payload = {
        "experiment": cfg.EXPERIMENT_NAME,
        "model":      "SimpleGAN",
        "fid":        scores.get("fid"),
        "dacid":      scores.get("dacid"),
        "g_losses":   g_losses,
        "d_losses":   d_losses,
    }
    out = cfg.RESULTS_DIR / "metrics.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Metrics    -> {out}")
    print(f"Done. Results -> {cfg.RESULTS_DIR}")


if __name__ == "__main__":
    main()
