"""
src/metrics.py
FID and DACID image quality metrics.

FID  (Frechet Inception Distance) — standard GAN evaluation metric.
     Frechet distance between multivariate Gaussians fitted to Inception-v3
     pool-3 features of real vs generated images. Lower = better.

DACID (Dany Aissa & Clara's Image Distance) — custom lightweight metric.
     L2 distance between the mean Inception feature vectors of real and fake
     distributions. Faster than FID, good for quick iteration comparisons.
     Lower = better.

Both metrics share the same Inception forward pass (computed once).
"""

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from scipy.linalg import sqrtm


# ── Inception extractor ───────────────────────────────────────────────

def _build_inception(device):
    inc = models.inception_v3(
        weights=models.Inception_V3_Weights.DEFAULT,
        transform_input=False,
    ).to(device)
    inc.eval()
    inc.fc = torch.nn.Identity()   # return pool-3 features (2048-d)
    return inc


_preprocess = T.Compose([
    T.Resize((299, 299), antialias=True),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _prep_batch(imgs):
    """[-1,1] tensor -> Inception-ready tensor in ImageNet normalisation."""
    imgs = (imgs.clamp(-1, 1) + 1) / 2          # -> [0,1]
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1, 3, 1, 1)           # grayscale -> RGB
    return torch.stack([_preprocess(img) for img in imgs])


@torch.no_grad()
def _extract_features(inception, images, batch_size=64):
    """Run images through Inception in batches -> (N, 2048) numpy array."""
    device = next(inception.parameters()).device
    feats  = []
    for start in range(0, len(images), batch_size):
        batch = _prep_batch(images[start:start+batch_size]).to(device)
        feats.append(inception(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


# ── DACID ─────────────────────────────────────────────────────────────

def dacid_score(real_feats, fake_feats):
    """
    DACID — Dany Aissa & Clara's Image Distance.
    L2 distance between mean Inception feature vectors.
    Lower = better.
    """
    return float(np.linalg.norm(
        np.mean(real_feats, axis=0) - np.mean(fake_feats, axis=0)
    ))


# ── FID ───────────────────────────────────────────────────────────────

def fid_score(real_feats, fake_feats):
    """
    Frechet Inception Distance (Heusel et al., 2017).
    FID = ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2*sqrt(Sigma_r*Sigma_f))
    Lower = better.
    """
    mu_r  = np.mean(real_feats, axis=0)
    mu_f  = np.mean(fake_feats, axis=0)
    cov_r = np.cov(real_feats, rowvar=False)
    cov_f = np.cov(fake_feats, rowvar=False)
    diff  = mu_r - mu_f
    covmean, _ = sqrtm(cov_r @ cov_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov_r + cov_f - 2 * covmean))


# ── AttGAN entry point ────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(enc, gen, test_loader, cfg, device):
    """
    Compute FID and DACID for an AttGAN model.
    Generates cfg.METRICS_N_SAMPLES fake images by translating real images
    with shuffled target attributes.
    Returns dict {"fid": float, "dacid": float}.
    """
    print("\n[metrics] Building Inception extractor...")
    inception = _build_inception(device)
    enc.eval(); gen.eval()

    real_list, fake_list, collected = [], [], 0
    for imgs, attrs in test_loader:
        if collected >= cfg.METRICS_N_SAMPLES:
            break
        imgs  = imgs.to(device)
        attrs = attrs.to(device)
        perm  = torch.randperm(imgs.size(0))
        fakes = gen(enc(imgs), attrs[perm])
        real_list.append(imgs.cpu())
        fake_list.append(fakes.cpu())
        collected += imgs.size(0)

    real_imgs = torch.cat(real_list)[:cfg.METRICS_N_SAMPLES]
    fake_imgs = torch.cat(fake_list)[:cfg.METRICS_N_SAMPLES]

    print(f"[metrics] Extracting features ({len(real_imgs)} images each)...")
    real_feats = _extract_features(inception, real_imgs)
    fake_feats = _extract_features(inception, fake_imgs)

    fid   = fid_score(real_feats, fake_feats)
    dacid = dacid_score(real_feats, fake_feats)
    print(f"[metrics] FID   = {fid:.4f}")
    print(f"[metrics] DACID = {dacid:.4f}")

    enc.train(); gen.train()
    return {"fid": round(fid, 4), "dacid": round(dacid, 4)}


# ── SimpleGAN entry point ─────────────────────────────────────────────

@torch.no_grad()
def compute_metrics_simple_gan(gen, test_loader, latent_dim, cfg, device):
    """
    Compute FID and DACID for an unconditional GAN.
    Returns dict {"fid": float, "dacid": float}.
    """
    print("\n[metrics] Building Inception extractor...")
    inception = _build_inception(device)
    gen.eval()

    real_list, fake_list, collected = [], [], 0
    for imgs, _ in test_loader:
        if collected >= cfg.METRICS_N_SAMPLES:
            break
        B     = imgs.size(0)
        z     = torch.randn(B, latent_dim, 1, 1, device=device)
        fakes = gen(z)
        real_list.append(imgs.cpu())
        fake_list.append(fakes.cpu())
        collected += B

    real_imgs = torch.cat(real_list)[:cfg.METRICS_N_SAMPLES]
    fake_imgs = torch.cat(fake_list)[:cfg.METRICS_N_SAMPLES]

    print(f"[metrics] Extracting features ({len(real_imgs)} images each)...")
    real_feats = _extract_features(inception, real_imgs)
    fake_feats = _extract_features(inception, fake_imgs)

    fid   = fid_score(real_feats, fake_feats)
    dacid = dacid_score(real_feats, fake_feats)
    print(f"[metrics] FID   = {fid:.4f}")
    print(f"[metrics] DACID = {dacid:.4f}")

    gen.train()
    return {"fid": round(fid, 4), "dacid": round(dacid, 4)}
