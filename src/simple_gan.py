"""
src/simple_gan.py
Unconditional DCGAN on CelebA (Radford et al., 2015).

Baseline GAN before AttGAN. Key differences:
    - No conditioning (pure random noise)
    - BCE loss instead of LSGAN
    - 64x64 output resolution
    - No attribute control

Classes:   SimpleGenerator, SimpleDiscriminator
Functions: build_simple_models, train_simple_gan
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm

from src.utils import denorm


# ── Architecture ──────────────────────────────────────────────────────

class SimpleGenerator(nn.Module):
    """noise z (latent_dim x 1 x 1) -> image (3 x 64 x 64)"""
    def __init__(self, latent_dim=100, dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, dim*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(dim*8), nn.ReLU(True),
            nn.ConvTranspose2d(dim*8, dim*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim*4), nn.ReLU(True),
            nn.ConvTranspose2d(dim*4, dim*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim*2), nn.ReLU(True),
            nn.ConvTranspose2d(dim*2, dim,   4, 2, 1, bias=False),
            nn.BatchNorm2d(dim),   nn.ReLU(True),
            nn.ConvTranspose2d(dim, 3,       4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)


class SimpleDiscriminator(nn.Module):
    """image (3 x 64 x 64) -> real/fake scalar"""
    def __init__(self, dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,     dim,   4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(dim,   dim*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(dim*2, dim*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(dim*4, dim*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(dim*8, 1,    4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).view(-1, 1)


# ── Factory ───────────────────────────────────────────────────────────

def build_simple_models(latent_dim=100, dim=64,
                         device=torch.device("cpu")):
    gen = SimpleGenerator(latent_dim=latent_dim, dim=dim).to(device)
    dis = SimpleDiscriminator(dim=dim).to(device)

    def _p(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"[simple_gan] Generator={_p(gen)/1e6:.2f}M  "
          f"Discriminator={_p(dis)/1e6:.2f}M")
    return gen, dis


# ── Training ──────────────────────────────────────────────────────────

def train_simple_gan(gen, dis, train_loader, cfg, device, latent_dim=100):
    """
    Train a DCGAN with binary cross-entropy loss.
    Returns (g_losses, d_losses) per-epoch averages.
    """
    criterion = nn.BCELoss()
    optim_G   = optim.Adam(gen.parameters(),
                            lr=cfg.LR, betas=(cfg.BETA1, cfg.BETA2))
    optim_D   = optim.Adam(dis.parameters(),
                            lr=cfg.LR, betas=(cfg.BETA1, cfg.BETA2))

    fixed_z   = torch.randn(64, latent_dim, 1, 1, device=device)
    g_losses, d_losses = [], []

    print(f"\n[simple_gan] Training {cfg.N_EPOCHS} epochs  "
          f"(experiment: {cfg.EXPERIMENT_NAME})")

    for epoch in range(1, cfg.N_EPOCHS + 1):
        gen.train(); dis.train()
        g_sum = d_sum = n = 0.0

        for imgs, _ in tqdm(train_loader, desc=f"  Epoch {epoch}", leave=False):
            # Downsample to 64x64 for DCGAN (loaded at 128x128)
            imgs = torch.nn.functional.interpolate(imgs, size=64,
                                                    mode="bilinear",
                                                    align_corners=False)
            imgs = imgs.to(device)
            B    = imgs.size(0)
            real_lbl = torch.ones(B,  1, device=device)
            fake_lbl = torch.zeros(B, 1, device=device)

            # D step
            optim_D.zero_grad()
            loss_D = (criterion(dis(imgs), real_lbl)
                      + criterion(dis(
                            gen(torch.randn(B, latent_dim, 1, 1,
                                             device=device)).detach()
                         ), fake_lbl))
            loss_D.backward()
            optim_D.step()

            # G step
            optim_G.zero_grad()
            z      = torch.randn(B, latent_dim, 1, 1, device=device)
            loss_G = criterion(dis(gen(z)), real_lbl)
            loss_G.backward()
            optim_G.step()

            g_sum += loss_G.item()
            d_sum += loss_D.item()
            n     += 1

        g_avg, d_avg = g_sum / n, d_sum / n
        g_losses.append(g_avg)
        d_losses.append(d_avg)
        print(f"Epoch [{epoch:>3}/{cfg.N_EPOCHS}]  G={g_avg:.4f}  D={d_avg:.4f}")

        if epoch % cfg.SAVE_EVERY == 0 or epoch == 1:
            _save_samples(gen, fixed_z, epoch, cfg)

    print("[simple_gan] Training complete.")
    return g_losses, d_losses


def _save_samples(gen, fixed_z, epoch, cfg):
    gen.eval()
    with torch.no_grad():
        imgs = gen(fixed_z).cpu()
    grid = torchvision.utils.make_grid(denorm(imgs), nrow=8, padding=2)
    path = cfg.RESULTS_DIR / f"simple_gan_epoch{epoch:03d}.png"
    torchvision.utils.save_image(grid, path)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")
    plt.title(f"Simple GAN — Epoch {epoch}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.show()
    plt.close()
    print(f"[simple_gan] Samples -> {path}")
    gen.train()
