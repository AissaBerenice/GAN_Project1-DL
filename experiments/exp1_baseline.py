"""
experiments/exp1_baseline.py
Experiment 1 — AttGAN paper defaults.
lambda_rec=100  lambda_cls_D=10  lambda_cls_G=1
"""
from config import Config


class Exp1Config(Config):
    EXPERIMENT_NAME = "exp1_baseline"
    LAMBDA_REC   = 100.0
    LAMBDA_CLS_D =  10.0
    LAMBDA_CLS_G =   1.0
    DESCRIPTION  = ("Baseline — paper default lambda values. "
                    "Balanced reconstruction vs attribute editing.")
