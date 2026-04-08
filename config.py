"""
config.py
Central configuration for the whole project.

Usage:
    from config import Config
    cfg = Config()

For experiments, import the subclass directly:
    from experiments.exp1_baseline import Exp1Config
    cfg = Exp1Config()

Colab credit guide (T4 GPU):
    SimpleGAN  10 epochs  ~25 min
    AttGAN     10 epochs  ~90 min per experiment
"""

from pathlib import Path


class Config:
    # ── Experiment name ───────────────────────────────────────────────
    EXPERIMENT_NAME = "default"

    # ── Paths ─────────────────────────────────────────────────────────
    ROOT     = Path(__file__).parent.resolve()
    DATA_DIR = ROOT / "data"

    @property
    def RESULTS_DIR(self):
        return self.ROOT / "results" / self.EXPERIMENT_NAME

    @property
    def CHECKPOINT_DIR(self):
        return self.ROOT / "checkpoints" / self.EXPERIMENT_NAME

    # ── Dataset ───────────────────────────────────────────────────────
    IMG_SIZE = 128

    ATTRS = [
        "Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair",
        "Bushy_Eyebrows", "Eyeglasses", "Male",
        "Mouth_Slightly_Open", "Mustache", "No_Beard",
        "Pale_Skin", "Young",
    ]
    N_ATTRS = len(ATTRS)   # 13

    # ── Training ──────────────────────────────────────────────────────
    BATCH_SIZE  = 32    # safe for T4 16 GB at 128x128
    N_EPOCHS    = 10    # enough to see clear results; raise to 30 for full run
    LR          = 0.0002
    BETA1       = 0.5
    BETA2       = 0.999
    NUM_WORKERS = 2

    # ── AttGAN loss weights (paper defaults) ──────────────────────────
    LAMBDA_REC   = 100.0
    LAMBDA_CLS_D =  10.0
    LAMBDA_CLS_G =   1.0

    # ── Architecture ──────────────────────────────────────────────────
    ENC_DIM = 64
    DEC_DIM = 64
    DIS_DIM = 64

    # ── Logging ───────────────────────────────────────────────────────
    SAVE_EVERY      = 2    # save samples + checkpoint every 2 epochs
    LOG_EVERY_STEPS = 200  # print batch loss every 200 steps

    # ── Metrics ───────────────────────────────────────────────────────
    COMPUTE_METRICS   = True
    METRICS_N_SAMPLES = 512   # 512 is plenty for a project; 2048 takes 4x longer

    def __init__(self):
        for d in [self.DATA_DIR, self.RESULTS_DIR, self.CHECKPOINT_DIR]:
            d.mkdir(parents=True, exist_ok=True)