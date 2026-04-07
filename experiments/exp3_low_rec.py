"""
experiments/exp3_low_rec.py
Experiment 3 — Stronger attribute signal.
lambda_rec=50  lambda_cls_D=10  lambda_cls_G=5
Hypothesis: sharper attribute edits, reduced identity preservation.
"""
from config import Config


class Exp3Config(Config):
    EXPERIMENT_NAME = "exp3_strong_attr"
    LAMBDA_REC   =  50.0
    LAMBDA_CLS_D =  10.0
    LAMBDA_CLS_G =   5.0
    DESCRIPTION  = ("Low lambda_rec=50, high lambda_cls_G=5. "
                    "Expected: sharper edits, some identity loss.")
