"""
src/utils.py
Visualisation, checkpointing, and evaluation utilities.
"""

import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision


# ── Image helpers ─────────────────────────────────────────────────────

def denorm(x):
    """[-1,1] tensor -> [0,1]"""
    return (x.clamp(-1, 1) + 1) / 2


def _to_np(t):
    return denorm(t).cpu().permute(1, 2, 0).numpy()


# ── Sample visualisation ─────────────────────────────────────────────

def visualise_samples(enc, gen, test_imgs, test_attrs, epoch, cfg):
    """
    Save a grid showing: original | reconstruction | 4 attribute flips.
    Called automatically during training every SAVE_EVERY epochs.
    """
    enc.eval(); gen.eval()
    n = test_imgs.size(0)

    with torch.no_grad():
        z   = enc(test_imgs)
        rec = gen(z, test_attrs)
        edits = []
        for i in range(min(4, cfg.N_ATTRS)):
            a = test_attrs.clone()
            a[:, i] = -a[:, i]
            edits.append(gen(z, a))

    rows   = [test_imgs.cpu(), rec.cpu()] + [e.cpu() for e in edits]
    labels = ["Original", "Reconstruct"] + \
             [f"Flip: {cfg.ATTRS[i]}" for i in range(len(edits))]

    fig, axes = plt.subplots(len(labels), n, figsize=(n * 2, len(labels) * 2.2))
    for r, (row_imgs, lbl) in enumerate(zip(rows, labels)):
        for c in range(n):
            ax = axes[r][c]
            ax.imshow(_to_np(row_imgs[c]))
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(lbl, fontsize=7, rotation=0,
                              labelpad=72, va="center")

    plt.suptitle(f"Epoch {epoch}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = cfg.RESULTS_DIR / f"samples_epoch{epoch:03d}.png"
    plt.savefig(path, dpi=90, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"[utils] Saved -> {path}")
    enc.train(); gen.train()


# ── Loss curves ───────────────────────────────────────────────────────

def plot_losses(g_losses, d_losses, cfg):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(g_losses, color="#e74c3c", label="G loss")
    ax1.set_title("Generator loss"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(d_losses, color="#2980b9", label="D loss")
    ax2.set_title("Discriminator loss"); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.suptitle(f"Training — {cfg.EXPERIMENT_NAME}", fontweight="bold")
    plt.tight_layout()
    path = cfg.RESULTS_DIR / "loss_curves.png"
    plt.savefig(path, dpi=100)
    plt.show()
    plt.close()
    print(f"[utils] Saved -> {path}")


# ── Checkpointing ────────────────────────────────────────────────────

def save_checkpoint(enc, gen, dis, epoch, cfg):
    path = cfg.CHECKPOINT_DIR / f"ckpt_epoch{epoch:03d}.pt"
    torch.save({"epoch": epoch,
                "enc": enc.state_dict(),
                "gen": gen.state_dict(),
                "dis": dis.state_dict()}, path)
    print(f"[utils] Checkpoint -> {path}")


def load_checkpoint(path, enc, gen, dis):
    ckpt = torch.load(path, map_location="cpu")
    enc.load_state_dict(ckpt["enc"])
    gen.load_state_dict(ckpt["gen"])
    dis.load_state_dict(ckpt["dis"])
    epoch = ckpt.get("epoch", 0)
    print(f"[utils] Resumed from epoch {epoch}")
    return epoch


# ── Attribute demo ───────────────────────────────────────────────────

def attribute_demo(enc, gen, test_loader, cfg, n_imgs=4):
    """
    For each test image: original + every attribute toggled independently.
    One row per image, one column per attribute.
    """
    enc.eval(); gen.eval()
    device = next(enc.parameters()).device
    imgs, attrs = next(iter(test_loader))
    imgs  = imgs[:n_imgs].to(device)
    attrs = attrs[:n_imgs].to(device)

    with torch.no_grad():
        z = enc(imgs)
        n_cols = cfg.N_ATTRS + 1
        fig, axes = plt.subplots(n_imgs, n_cols,
                                  figsize=(n_cols * 1.9, n_imgs * 2.1))
        for img_i in range(n_imgs):
            axes[img_i][0].imshow(_to_np(imgs[img_i]))
            axes[img_i][0].axis("off")
            if img_i == 0:
                axes[img_i][0].set_title("Original", fontsize=6, fontweight="bold")
            for attr_i, name in enumerate(cfg.ATTRS):
                a = attrs[img_i:img_i+1].clone()
                cur = a[0, attr_i].item()
                a[0, attr_i] = -cur
                edited = gen(z[img_i:img_i+1], a)
                ax = axes[img_i][attr_i + 1]
                ax.imshow(_to_np(edited[0]))
                ax.axis("off")
                if img_i == 0:
                    sign = "+" if cur < 0 else "-"
                    ax.set_title(f"{sign}{name}", fontsize=5.5)

    plt.suptitle(f"Attribute demo — {cfg.EXPERIMENT_NAME}\n"
                 "(each column = one attribute flipped)",
                 fontsize=9, fontweight="bold")
    plt.tight_layout()
    path = cfg.RESULTS_DIR / "attribute_demo.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"[utils] Saved -> {path}")


# ── Quantitative evaluation ──────────────────────────────────────────

def evaluate_attribute_accuracy(enc, gen, dis, test_loader, cfg,
                                  n_batches=20):
    """
    Uses D's cls_head to measure how accurately the generator produces
    the requested attributes on translated images.
    Returns overall accuracy (float, 0-100).
    """
    enc.eval(); gen.eval(); dis.eval()
    device  = next(enc.parameters()).device
    correct = torch.zeros(cfg.N_ATTRS, device=device)
    total   = 0

    with torch.no_grad():
        for i, (imgs, attrs) in enumerate(test_loader):
            if i >= n_batches:
                break
            imgs  = imgs.to(device)
            attrs = attrs.to(device)
            B     = imgs.size(0)
            perm  = torch.randperm(B)
            tgt   = attrs[perm]
            fakes = gen(enc(imgs), tgt)
            _, cls = dis(fakes)
            pred  = (cls > 0).float() * 2 - 1
            correct += (pred == tgt).sum(dim=0)
            total   += B

    acc_per = (correct / total * 100).cpu().numpy()
    overall  = float(acc_per.mean())

    print("\n[eval] Attribute accuracy on generated images:")
    for name, acc in zip(cfg.ATTRS, acc_per):
        bar = "█" * int(acc / 5)
        print(f"  {name:<22} {acc:5.1f}%  {bar}")
    print(f"\n  Overall: {overall:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(cfg.ATTRS, acc_per, color="#3498db", edgecolor="white")
    ax.axhline(overall, color="#e74c3c", ls="--",
               label=f"Mean {overall:.1f}%")
    ax.set_ylim(0, 105)
    ax.set_xticklabels(cfg.ATTRS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Attribute classification accuracy on generated images",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = cfg.RESULTS_DIR / "attr_accuracy.png"
    plt.savefig(path, dpi=100)
    plt.show()
    plt.close()
    print(f"[eval] Saved -> {path}")

    enc.train(); gen.train(); dis.train()
    return overall


def evaluate_reconstruction(enc, gen, test_loader, cfg, n_batches=20):
    """L1 reconstruction error on test set. Lower = better."""
    enc.eval(); gen.eval()
    device = next(enc.parameters()).device
    crit   = torch.nn.L1Loss()
    total, n = 0.0, 0
    with torch.no_grad():
        for i, (imgs, attrs) in enumerate(test_loader):
            if i >= n_batches:
                break
            imgs  = imgs.to(device)
            attrs = attrs.to(device)
            rec   = gen(enc(imgs), attrs)
            total += crit(rec, imgs).item()
            n     += 1
    avg = total / n
    print(f"\n[eval] Reconstruction L1: {avg:.4f}  (lower = better)")
    enc.train(); gen.train()
    return avg
