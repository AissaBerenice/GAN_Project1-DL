# GAN_Project1-DL

GAN_Project_DL/
├── config.py               ← DATA_DIR, experiment names, all hyperparams
├── train.py                ← AttGAN CLI: --exp / --resume / --eval-only
├── train_simple_gan.py     ← SimpleGAN CLI
├── export_results.py       ← Comparison charts after all experiments
├── requirements.txt        ← torch, torchvision, scipy, matplotlib, etc.
├── README.md
├── .gitignore
├── src/
│   ├── dataset.py          ← torchvision CelebA, {0,1}→{-1,+1}
│   ├── models.py           ← Encoder, Generator, Discriminator
│   ├── losses.py           ← L_adv (LSGAN), L_cls (BCE), L_rec (L1)
│   ├── trainer.py          ← alternating D/G training + metrics save
│   ├── simple_gan.py       ← DCGAN 64×64 + training loop
│   ├── metrics.py          ← FID + DACID (shared Inception pass)
│   └── utils.py            ← visualise, checkpoint, eval, attribute demo
├── experiments/
│   ├── exp1_baseline.py    ← λ_rec=100
│   ├── exp2_high_rec.py    ← λ_rec=200
│   └── exp3_low_rec.py     ← λ_rec=50, λ_cls_G=5
└── notebooks/
    ├── AttGAN_Colab.ipynb  ← 13 cells, experiment selector, export
    └── SimpleGAN_Colab.ipynb ← 9 cells, full DCGAN flow
