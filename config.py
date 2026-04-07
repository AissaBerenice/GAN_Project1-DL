"""
config.py
Central configuration for the whole project.

Usage:
    from config import Config
    cfg = Config()

For experiments, import the subclass directly:
    from experiments.exp1_baseline import Exp1Config
    cfg = Exp1Config()
"""

from pathlib import Path


class Config:
    # ── Experiment name ───────────────────────────────────────────────
    # Results and checkpoints are saved to subfolders named after this.
    EXPERIMENT_NAME = "default"

    # ── Paths ─────────────────────────────────────────────────────────
    ROOT     = Path(__file__).parent.resolve()
    DATA_DIR = ROOT / "data"          # torchvision downloads CelebA here

    @property
    def RESULTS_DIR(self):
        return self.ROOT / "results" / self.EXPERIMENT_NAME

    @property
    def CHECKPOINT_DIR(self):
        return self.ROOT / "checkpoints" / self.EXPERIMENT_NAME

    # ── Dataset ───────────────────────────────────────────────────────
    IMG_SIZE = 128

    # 13 attributes chosen from CelebA's 40
    ATTRS = [
        "Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair",
        "Bushy_Eyebrows", "Eyeglasses", "Male",
        "Mouth_Slightly_Open", "Mustache", "No_Beard",
        "Pale_Skin", "Young",
    ]
    N_ATTRS = len(ATTRS)   # 13

    # ── Training ──────────────────────────────────────────────────────
    BATCH_SIZE  = 32
    N_EPOCHS    = 30
    LR          = 0.0002
    BETA1       = 0.5
    BETA2       = 0.999
    NUM_WORKERS = 2

    # ── AttGAN loss weights (paper defaults) ──────────────────────────
    LAMBDA_REC   = 100.0   # reconstruction loss weight
    LAMBDA_CLS_D =  10.0   # discriminator classification weight
    LAMBDA_CLS_G =   1.0   # generator classification weight

    # ── Architecture ──────────────────────────────────────────────────
    ENC_DIM = 64
    DEC_DIM = 64
    DIS_DIM = 64

    # ── Logging ───────────────────────────────────────────────────────
    SAVE_EVERY      = 5    # save samples + checkpoint every N epochs
    LOG_EVERY_STEPS = 100  # print batch loss every N steps

    # ── Metrics ───────────────────────────────────────────────────────
    COMPUTE_METRICS   = True   # set False to skip FID/DACID (~5 min on T4)
    METRICS_N_SAMPLES = 2048

    def __init__(self):
        for d in [self.DATA_DIR, self.RESULTS_DIR, self.CHECKPOINT_DIR]:
            d.mkdir(parents=True, exist_ok=True)
