"""
src/losses.py
Loss functions and optimizers for AttGAN.

Three losses:
    L_adv  MSELoss (LSGAN)         - makes fakes look real
    L_cls  BCEWithLogitsLoss        - ensures correct attributes appear
    L_rec  L1Loss                   - preserves identity / non-edited regions

Total generator loss:
    L_G = L_adv + lambda_cls_G * L_cls + lambda_rec * L_rec

Total discriminator loss:
    L_D = L_adv_real + L_adv_fake + lambda_cls_D * L_cls_real
"""

import torch
import torch.nn as nn
import torch.optim as optim


class AttGANLoss:
    def __init__(self, device):
        self.device  = device
        self.adv = nn.MSELoss()
        self.cls = nn.BCEWithLogitsLoss()
        self.rec = nn.L1Loss()

    # ── label helpers ─────────────────────────────────────────────────
    def ones(self, n):
        return torch.ones(n, 1, device=self.device)

    def zeros(self, n):
        return torch.zeros(n, 1, device=self.device)

    @staticmethod
    def to_binary(a):
        """Bipolar {-1,+1} -> binary {0,1} for BCE."""
        return (a + 1) / 2

    # ── Discriminator losses ──────────────────────────────────────────
    def d_adv_real(self, adv_real):
        return self.adv(adv_real, self.ones(adv_real.size(0)))

    def d_adv_fake(self, adv_fake):
        return self.adv(adv_fake, self.zeros(adv_fake.size(0)))

    def d_cls(self, cls_logits, attrs):
        return self.cls(cls_logits, self.to_binary(attrs))

    # ── Generator losses ──────────────────────────────────────────────
    def g_adv(self, adv_fake):
        return self.adv(adv_fake, self.ones(adv_fake.size(0)))

    def g_cls(self, cls_logits, target_attrs):
        return self.cls(cls_logits, self.to_binary(target_attrs))

    def g_rec(self, rec_img, real_img):
        return self.rec(rec_img, real_img)


def build_optimizers(enc, gen, dis, cfg):
    """Returns (optim_G, optim_D)."""
    optim_G = optim.Adam(list(enc.parameters()) + list(gen.parameters()),
                         lr=cfg.LR, betas=(cfg.BETA1, cfg.BETA2))
    optim_D = optim.Adam(dis.parameters(),
                         lr=cfg.LR, betas=(cfg.BETA1, cfg.BETA2))
    return optim_G, optim_D
