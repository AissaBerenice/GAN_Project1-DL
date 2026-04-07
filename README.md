
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

---

## Run on Google Colab (recommended)

**Replace `YOUR_USERNAME` with your GitHub username in Cell 1 of each notebook.**

| Notebook | Open |
|---|---|
| Simple GAN | `notebooks/SimpleGAN_Colab.ipynb` |
| AttGAN | `notebooks/AttGAN_Colab.ipynb` |

**Steps:**
1. Open a notebook in Colab
2. `Runtime → Change runtime type → T4 GPU → Save`
3. Run Cell 1 — clones the repo and installs requirements
4. For AttGAN: change `EXPERIMENT` in Cell 3 and run the full notebook once per experiment
5. After all three AttGAN experiments: run Cell 13 to generate comparison charts

---

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/GAN_Project_DL.git
cd GAN_Project_DL
pip install -r requirements.txt

# Simple GAN
python train_simple_gan.py
python train_simple_gan.py --epochs 20
python train_simple_gan.py --no-metrics     # skip FID/DACID

# AttGAN — one experiment at a time
python train.py --exp exp1_baseline
python train.py --exp exp2_high_rec
python train.py --exp exp3_strong_attr

# Resume from checkpoint
python train.py --exp exp1_baseline --resume checkpoints/exp1_baseline/ckpt_epoch010.pt

# Evaluation only (no training)
python train.py --exp exp1_baseline --eval-only --resume checkpoints/exp1_baseline/ckpt_epoch030.pt

# Skip FID/DACID computation
python train.py --exp exp1_baseline --no-metrics

# Compare all experiments
python export_results.py
```

---

## Experiments

Three AttGAN configurations that vary the loss weights:

| ID | EXPERIMENT_NAME | λ_rec | λ_cls_D | λ_cls_G | Hypothesis |
|---|---|---|---|---|---|
| 1 | `exp1_baseline` | 100 | 10 | 1 | Paper defaults — balanced |
| 2 | `exp2_high_rec` | 200 | 10 | 1 | Stronger identity, softer edits |
| 3 | `exp3_strong_attr` | 50 | 10 | 5 | Sharper edits, some identity loss |

Each config is a subclass of `Config` in `experiments/` that only overrides
the changed values. Results land in isolated subfolders so runs never overwrite each other.

After running all three, `python export_results.py` produces:
- `results/comparison_table.csv`
- `results/comparison_metrics.png` — FID & DACID bar charts
- `results/comparison_losses.png` — overlaid G/D loss curves

---

## Architecture

### Simple GAN (DCGAN)
```
z (100×1×1) → Generator (5× ConvTranspose2d) → image (3×64×64)
                                                       ↓
                              Discriminator (5× Conv2d) → scalar
Loss G: BCE(D(G(z)), 1)
Loss D: BCE(D(x), 1) + BCE(D(G(z)), 0)
```

### AttGAN
```
image → Encoder → z
                  ↓
    target_attrs──┤ (tiled spatially, concat)
                  ↓
             Generator → fake_image
                              ↓
             Discriminator ←──┘
               ├─ adv_head → real/fake    MSELoss (LSGAN)
               └─ cls_head → 13 attrs    BCEWithLogitsLoss
```

**Total Generator loss:**  `L_adv + λ_cls_G · L_cls + λ_rec · L_rec`  
**Total Discriminator loss:** `L_adv_real + L_adv_fake + λ_cls_D · L_cls_real`

---

## Metrics

| Metric | Description | Better |
|---|---|---|
| **FID** | Fréchet Inception Distance — Fréchet distance between Inception-v3 feature distributions of real vs generated images | Lower |
| **DACID** | Dany Aissa & Clara's Image Distance — L2 distance between mean Inception feature vectors | Lower |

Both are computed using the same Inception forward pass (no double cost).
Set `COMPUTE_METRICS = False` in config or pass `--no-metrics` to skip (~5 min on T4).

---

## Output files per experiment

| File | Description |
|---|---|
| `results/<exp>/samples_epoch***.png` | Visual progress grid every 5 epochs |
| `results/<exp>/attribute_demo.png` | Each attribute toggled independently |
| `results/<exp>/attr_accuracy.png` | Per-attribute classification accuracy |
| `results/<exp>/loss_curves.png` | G and D loss over all epochs |
| `results/<exp>/metrics.json` | FID, DACID, full loss history |

---

## CelebA download issue

If torchvision raises `FileURLRetrievalError: Too many users have viewed or downloaded
this file recently`, the Google Drive quota is temporarily exceeded.
Use the fallback cells at the bottom of either Colab notebook
(Kaggle API or Google Drive symlink).

---

## References

- Goodfellow, I. et al. (2014). *Generative Adversarial Nets.* NeurIPS.
- Radford, A. et al. (2015). *Unsupervised Representation Learning with DCGANs.* ICLR 2016.
- He, Z. et al. (2019). *AttGAN: Facial Attribute Editing by Only Changing What You Want.* IEEE TIP.
- Heusel, M. et al. (2017). *GANs Trained by a Two Time-Scale Update Rule.* NeurIPS.
