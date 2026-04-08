
# GAN_Project_DL

PyTorch implementation of two GAN models on CelebA, progressing from an
unconditional baseline to a conditional attribute-editing network.

| Model | Paper | Task |
|---|---|---|
| **Simple GAN** | Radford et al., 2015 (DCGAN) | Unconditional 64×64 face generation |
| **AttGAN** | He et al., 2019 | Conditional facial attribute editing 128×128 |


<img src="https://iteso.mx/documents/27014/202031/Logo-ITESO-MinimoH.png"
     align="right"
     width="300"/>

- Aissa Berenice González Fosado
- Daniela de la Torre Gallo
- Clara Paola Aguilar Casillas


---

## Repository structure

```
GAN_Project_DL/
├── config.py                  ← All hyperparameters and paths
├── train.py                   ← AttGAN training entry point
├── train_simple_gan.py        ← Simple GAN training entry point
├── export_results.py          ← Compare all experiments after training
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── dataset.py             ← CelebA loader (torchvision)
│   ├── models.py              ← Encoder · Generator · Discriminator
│   ├── simple_gan.py          ← DCGAN generator and discriminator
│   ├── losses.py              ← L_adv · L_cls · L_rec + optimizers
│   ├── trainer.py             ← AttGAN training loop
│   ├── metrics.py             ← FID + DACID (shared Inception pass)
│   └── utils.py               ← Visualisation · eval · checkpointing
│
├── experiments/
│   ├── exp1_baseline.py       ← λ_rec=100  λ_cls_G=1   (paper defaults)
│   ├── exp2_high_rec.py       ← λ_rec=200  λ_cls_G=1   (stronger identity)
│   └── exp3_low_rec.py        ← λ_rec=50   λ_cls_G=5   (sharper edits)
│
├── notebooks/
│   ├── AttGAN_Colab.ipynb     ← Full AttGAN walkthrough for Google Colab
│   └── SimpleGAN_Colab.ipynb  ← Simple GAN walkthrough for Google Colab
│
├── data/                      ← CelebA downloaded here by torchvision
├── results/                   ← One subfolder per experiment
└── checkpoints/               ← One subfolder per experiment
```


## References

- Goodfellow, I. et al. (2014). *Generative Adversarial Nets.* NeurIPS.
- Radford, A. et al. (2015). *Unsupervised Representation Learning with DCGANs.* ICLR 2016.
- He, Z. et al. (2019). *AttGAN: Facial Attribute Editing by Only Changing What You Want.* IEEE TIP.
- Heusel, M. et al. (2017). *GANs Trained by a Two Time-Scale Update Rule.* NeurIPS.
