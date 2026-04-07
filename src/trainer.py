"""
src/trainer.py
AttGAN training loop with checkpoint and metrics support.
"""

import json
import torch
from tqdm import tqdm

from src.losses import AttGANLoss, build_optimizers
from src.utils  import save_checkpoint, load_checkpoint, visualise_samples


class Trainer:
    """
    Manages the full AttGAN training loop.

    Args:
        enc, gen, dis : model instances already on device
        train_loader  : training DataLoader
        test_loader   : test DataLoader (used for visualisation)
        cfg           : Config instance
        device        : torch.device
    """

    def __init__(self, enc, gen, dis, train_loader, test_loader, cfg, device):
        self.enc = enc
        self.gen = gen
        self.dis = dis
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.cfg    = cfg
        self.device = device

        self.criterion = AttGANLoss(device)
        self.optim_G, self.optim_D = build_optimizers(enc, gen, dis, cfg)

        # Fixed test batch for consistent per-epoch visualisation
        imgs, attrs = next(iter(test_loader))
        self.test_imgs  = imgs[:8].to(device)
        self.test_attrs = attrs[:8].to(device)

        self.g_losses = []
        self.d_losses = []

    # ── One epoch ─────────────────────────────────────────────────────

    def _train_epoch(self):
        self.enc.train(); self.gen.train(); self.dis.train()
        g_sum = d_sum = n = 0.0
        cfg  = self.cfg
        crit = self.criterion

        for step, (imgs, attrs) in enumerate(
                tqdm(self.train_loader, desc="  batches", leave=False)):
            imgs  = imgs.to(self.device)
            attrs = attrs.to(self.device)
            B     = imgs.size(0)
            perm  = torch.randperm(B)
            tgt   = attrs[perm]

            # ── Train D ───────────────────────────────────────────────
            self.optim_D.zero_grad()
            adv_real, cls_real = self.dis(imgs)
            loss_D = (crit.d_adv_real(adv_real)
                      + crit.d_cls(cls_real, attrs) * cfg.LAMBDA_CLS_D)

            with torch.no_grad():
                fakes = self.gen(self.enc(imgs), tgt)
            adv_fake, _ = self.dis(fakes.detach())
            loss_D += crit.d_adv_fake(adv_fake)
            loss_D.backward()
            self.optim_D.step()

            # ── Train G ───────────────────────────────────────────────
            self.optim_G.zero_grad()
            z    = self.enc(imgs)
            rec  = self.gen(z, attrs)
            loss_rec = crit.g_rec(rec, imgs)

            fakes = self.gen(z, tgt)
            adv_fake, cls_fake = self.dis(fakes)
            loss_G = (crit.g_adv(adv_fake)
                      + crit.g_cls(cls_fake, tgt) * cfg.LAMBDA_CLS_G
                      + loss_rec * cfg.LAMBDA_REC)
            loss_G.backward()
            self.optim_G.step()

            g_sum += loss_G.item()
            d_sum += loss_D.item()
            n     += 1

            if (step + 1) % cfg.LOG_EVERY_STEPS == 0:
                tqdm.write(f"    step {step+1:>4}  "
                           f"G={loss_G.item():.4f}  D={loss_D.item():.4f}")

        return g_sum / n, d_sum / n

    # ── Full run ──────────────────────────────────────────────────────

    def train(self, resume_path=None):
        """
        Run training. Pass resume_path to continue from a checkpoint.
        Returns (g_losses, d_losses) lists.
        """
        start = 0
        if resume_path is not None:
            start = load_checkpoint(resume_path, self.enc, self.gen, self.dis)

        cfg = self.cfg
        print(f"\n[trainer] {cfg.EXPERIMENT_NAME}  "
              f"epochs={cfg.N_EPOCHS}  "
              f"lambda_rec={cfg.LAMBDA_REC}  "
              f"lambda_cls_G={cfg.LAMBDA_CLS_G}")

        for epoch in range(start + 1, cfg.N_EPOCHS + 1):
            g, d = self._train_epoch()
            self.g_losses.append(g)
            self.d_losses.append(d)
            print(f"Epoch [{epoch:>3}/{cfg.N_EPOCHS}]  "
                  f"G={g:.4f}  D={d:.4f}")

            if epoch % cfg.SAVE_EVERY == 0 or epoch == 1:
                visualise_samples(self.enc, self.gen,
                                  self.test_imgs, self.test_attrs,
                                  epoch, cfg)
                save_checkpoint(self.enc, self.gen, self.dis, epoch, cfg)

        print("\n[trainer] Training complete.")

        if cfg.COMPUTE_METRICS:
            from src.metrics import compute_metrics
            scores = compute_metrics(
                self.enc, self.gen, self.test_loader, cfg, self.device)
            self._save_metrics(scores)

        return self.g_losses, self.d_losses

    def _save_metrics(self, scores):
        payload = {
            "experiment":   self.cfg.EXPERIMENT_NAME,
            "model":        "AttGAN",
            "lambda_rec":   self.cfg.LAMBDA_REC,
            "lambda_cls_d": self.cfg.LAMBDA_CLS_D,
            "lambda_cls_g": self.cfg.LAMBDA_CLS_G,
            "fid":          scores.get("fid"),
            "dacid":        scores.get("dacid"),
            "g_losses":     self.g_losses,
            "d_losses":     self.d_losses,
        }
        path = self.cfg.RESULTS_DIR / "metrics.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[trainer] Metrics -> {path}")
