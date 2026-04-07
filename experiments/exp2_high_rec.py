"""
experiments/exp2_high_rec.py
Experiment 2 — Higher reconstruction weight.
lambda_rec=200  lambda_cls_D=10  lambda_cls_G=1
Hypothesis: stronger identity preservation, softer attribute edits.
"""
from config import Config


class Exp2Config(Config):
    EXPERIMENT_NAME = "exp2_high_rec"
    LAMBDA_REC   = 200.0
    LAMBDA_CLS_D =  10.0
    LAMBDA_CLS_G =   1.0
    DESCRIPTION  = ("High lambda_rec=200. "
                    "Expected: better identity preservation, softer edits.")
