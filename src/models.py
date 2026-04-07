"""
src/models.py
AttGAN architecture: Encoder, Generator, Discriminator.

Architecture overview:
    image -> Encoder -> z (latent)
                           |
             target_attrs -+- (tiled spatially, concat)
                           |
                       Generator -> fake_image
                                        |
                       Discriminator <--+
                           |- adv_head  (LSGAN real/fake)
                           |- cls_head  (13 attribute logits)
"""

import torch
import torch.nn as nn


# ── Building blocks ───────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d -> InstanceNorm -> LeakyReLU  (Encoder / Discriminator)"""
    def __init__(self, in_ch, out_ch, kernel=4, stride=2, pad=1,
                 norm=True, act="leaky"):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=not norm)]
        if norm:              layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        if act == "leaky":    layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif act == "relu":   layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    """ConvTranspose2d -> InstanceNorm -> ReLU  (Generator)"""
    def __init__(self, in_ch, out_ch, kernel=4, stride=2, pad=1,
                 norm=True, act="relu"):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, pad,
                                     bias=not norm)]
        if norm:              layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        if act == "relu":     layers.append(nn.ReLU(inplace=True))
        elif act == "tanh":   layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Encoder ───────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    image (3 x 128 x 128) -> z (dim*8 x 4 x 4)
    Five strided convolutions halve spatial dims each step.
    """
    def __init__(self, dim=64):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3,      dim,   norm=False),   # 128->64
            ConvBlock(dim,    dim*2),                # 64->32
            ConvBlock(dim*2,  dim*4),                # 32->16
            ConvBlock(dim*4,  dim*8),                # 16->8
            ConvBlock(dim*8,  dim*8),                # 8->4
        )

    def forward(self, x):
        return self.net(x)


# ── Generator / Decoder ───────────────────────────────────────────────

class Generator(nn.Module):
    """
    (z, target_attrs) -> image (3 x 128 x 128)
    Attributes are tiled spatially and concatenated to z before decoding.
    """
    def __init__(self, n_attrs, dim=64):
        super().__init__()
        self.net = nn.Sequential(
            UpBlock(dim*8 + n_attrs, dim*8),   # 4->8
            UpBlock(dim*8,  dim*4),             # 8->16
            UpBlock(dim*4,  dim*2),             # 16->32
            UpBlock(dim*2,  dim),               # 32->64
            UpBlock(dim,    3, norm=False, act="tanh"),  # 64->128
        )

    def forward(self, z, a):
        h, w   = z.shape[2], z.shape[3]
        a_tile = a.view(a.size(0), -1, 1, 1).expand(-1, -1, h, w)
        return self.net(torch.cat([z, a_tile], dim=1))


# ── Discriminator ────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    image -> (adv_logit, cls_logits)
    adv_head: scalar  (LSGAN real/fake)
    cls_head: n_attrs (attribute classification)
    """
    def __init__(self, n_attrs, dim=64):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,     dim,   norm=False),   # 128->64
            ConvBlock(dim,   dim*2),                # 64->32
            ConvBlock(dim*2, dim*4),                # 32->16
            ConvBlock(dim*4, dim*8),                # 16->8
            ConvBlock(dim*8, dim*8),                # 8->4
        )
        self.adv_head = nn.Conv2d(dim*8, 1,       kernel_size=4, stride=1, padding=0)
        self.cls_head = nn.Conv2d(dim*8, n_attrs, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        feat = self.features(x)
        adv  = self.adv_head(feat).view(x.size(0), -1)
        cls  = self.cls_head(feat).view(x.size(0), -1)
        return adv, cls


# ── Factory ───────────────────────────────────────────────────────────

def build_models(cfg, device):
    enc = Encoder(dim=cfg.ENC_DIM).to(device)
    gen = Generator(n_attrs=cfg.N_ATTRS, dim=cfg.DEC_DIM).to(device)
    dis = Discriminator(n_attrs=cfg.N_ATTRS, dim=cfg.DIS_DIM).to(device)

    def _p(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"[models] Encoder={_p(enc)/1e6:.2f}M  "
          f"Generator={_p(gen)/1e6:.2f}M  "
          f"Discriminator={_p(dis)/1e6:.2f}M")
    return enc, gen, dis
