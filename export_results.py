"""
export_results.py — Compare all experiments after training.

Run after completing all experiments:
    python export_results.py

Reads results/<exp>/metrics.json for each experiment and produces:
    results/comparison_table.csv
    results/comparison_metrics.png   (FID & DACID bar charts)
    results/comparison_losses.png    (overlaid loss curves)

Missing experiments are skipped automatically.
"""

import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

ROOT    = Path(__file__).parent.resolve()
RES_DIR = ROOT / "results"

EXPERIMENTS = [
    "simple_gan",
    "exp1_baseline",
    "exp2_high_rec",
    "exp3_strong_attr",
]

LABELS = {
    "simple_gan":       "Simple GAN\n(DCGAN)",
    "exp1_baseline":    "AttGAN Exp 1\n(λ_rec=100)",
    "exp2_high_rec":    "AttGAN Exp 2\n(λ_rec=200)",
    "exp3_strong_attr": "AttGAN Exp 3\n(λ_rec=50)",
}

COLORS = {
    "simple_gan":       "#7F77DD",
    "exp1_baseline":    "#1D9E75",
    "exp2_high_rec":    "#BA7517",
    "exp3_strong_attr": "#D85A30",
}


def load_results():
    rows = []
    for exp in EXPERIMENTS:
        path = RES_DIR / exp / "metrics.json"
        if not path.exists():
            print(f"[export] Skipping {exp} (no metrics.json)")
            continue
        with open(path) as f:
            data = json.load(f)
        data["_key"] = exp
        rows.append(data)
        print(f"[export] Loaded {exp}  FID={data.get('fid')}  "
              f"DACID={data.get('dacid')}")
    return rows


def export_csv(rows):
    out = RES_DIR / "comparison_table.csv"
    fields = ["experiment", "model", "fid", "dacid",
              "final_g_loss", "final_d_loss", "n_epochs"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            g = r.get("g_losses", [])
            d = r.get("d_losses", [])
            w.writerow({
                "experiment":   r.get("experiment", r["_key"]),
                "model":        r.get("model", "AttGAN"),
                "fid":          r.get("fid"),
                "dacid":        r.get("dacid"),
                "final_g_loss": round(g[-1], 4) if g else None,
                "final_d_loss": round(d[-1], 4) if d else None,
                "n_epochs":     len(g),
            })
    print(f"[export] CSV -> {out}")


def export_metrics_chart(rows):
    valid = [r for r in rows if r.get("fid") is not None]
    if not valid:
        print("[export] No FID/DACID values — skipping metrics chart")
        return

    keys   = [r["_key"] for r in valid]
    fids   = [r["fid"]   for r in valid]
    dacids = [r["dacid"] for r in valid]
    colors = [COLORS.get(k, "#888") for k in keys]
    labels = [LABELS.get(k, k) for k in keys]
    x      = np.arange(len(keys))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    bars1 = ax1.bar(x, fids, 0.6, color=colors, edgecolor="white")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("FID (lower = better)")
    ax1.set_title("Frechet Inception Distance", fontweight="bold")
    ax1.bar_label(bars1, fmt="%.1f", fontsize=8, padding=3)
    ax1.set_ylim(0, max(fids) * 1.25); ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(x, dacids, 0.6, color=colors, edgecolor="white")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("DACID (lower = better)")
    ax2.set_title("DACID Score", fontweight="bold")
    ax2.bar_label(bars2, fmt="%.1f", fontsize=8, padding=3)
    ax2.set_ylim(0, max(dacids) * 1.25); ax2.grid(axis="y", alpha=0.3)

    patches = [mpatches.Patch(color=COLORS.get(k, "#888"),
                               label=LABELS.get(k, k).split("\n")[0])
               for k in keys]
    fig.legend(handles=patches, loc="lower center", ncol=len(keys),
               fontsize=8, bbox_to_anchor=(0.5, -0.04))
    plt.suptitle("Experiment Comparison — FID & DACID",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = RES_DIR / "comparison_metrics.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"[export] Metrics chart -> {out}")


def export_loss_curves(rows):
    attgan = [r for r in rows if r["_key"] != "simple_gan"
              and r.get("g_losses")]
    if not attgan:
        print("[export] No AttGAN loss data — skipping loss chart")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    for r in attgan:
        k   = r["_key"]
        lbl = LABELS.get(k, k).split("\n")[0]
        col = COLORS.get(k, "#888")
        ax1.plot(r["g_losses"], label=lbl, color=col)
        ax2.plot(r["d_losses"], label=lbl, color=col)

    ax1.set_title("Generator Loss");     ax1.set_xlabel("Epoch")
    ax1.legend(fontsize=8);              ax1.grid(alpha=0.3)
    ax2.set_title("Discriminator Loss"); ax2.set_xlabel("Epoch")
    ax2.legend(fontsize=8);              ax2.grid(alpha=0.3)
    plt.suptitle("AttGAN Training — All Experiments",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = RES_DIR / "comparison_losses.png"
    plt.savefig(out, dpi=120)
    plt.show()
    plt.close()
    print(f"[export] Loss chart -> {out}")


def print_summary(rows):
    print("\n" + "="*68)
    print(f"  {'Experiment':<24} {'Model':<10} {'FID':>8} "
          f"{'DACID':>8} {'G_loss':>8} {'D_loss':>8}")
    print("="*68)
    for r in rows:
        g = r.get("g_losses", [])
        d = r.get("d_losses", [])
        print(f"  {r['_key']:<24} "
              f"{r.get('model','AttGAN'):<10} "
              f"{str(r.get('fid',  'n/a')):>8} "
              f"{str(r.get('dacid','n/a')):>8} "
              f"{(round(g[-1],4) if g else 'n/a'):>8} "
              f"{(round(d[-1],4) if d else 'n/a'):>8}")
    print("="*68 + "\n")


def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_results()
    if not rows:
        print("No results found. Run training first.")
        return
    print_summary(rows)
    export_csv(rows)
    export_metrics_chart(rows)
    export_loss_curves(rows)
    print(f"\nAll exports -> {RES_DIR}/")


if __name__ == "__main__":
    main()
