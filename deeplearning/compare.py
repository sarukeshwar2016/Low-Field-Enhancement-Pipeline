# ============================================================
# COMPARISON VISUALIZATION — ALL METHODS
# ============================================================
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

sys.path.insert(0, os.path.dirname(__file__))

from dataset_loader import get_dataloaders
from unet import UNet
from dncnn import DnCNN

DEVICE = torch.device("cpu")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")


def load_results():
    """Load patient-level results CSV."""
    path = os.path.join(RESULTS_DIR, "patient_level_results.csv")
    if not os.path.exists(path):
        print("ERROR: Run evaluate.py first!")
        sys.exit(1)
    return pd.read_csv(path)


def plot_boxplots(df):
    """Grouped boxplots for PSNR, SSIM, SNR across methods."""
    methods = []
    for col in df.columns:
        if col.endswith("_PSNR") and col != "Patient":
            methods.append(col.replace("_PSNR", ""))

    # Reshape for seaborn
    rows = []
    for _, row in df.iterrows():
        for m in methods:
            label = m if m != "Classical" else "Ours (Physics)"
            psnr = row.get(f"{m}_PSNR", np.nan)
            ssim_v = row.get(f"{m}_SSIM", np.nan)
            snr = row.get(f"{m}_SNR", np.nan)
            if not np.isnan(psnr):
                rows.append({"Method": label, "Metric": "PSNR", "Value": psnr})
            if not np.isnan(ssim_v):
                rows.append({"Method": label, "Metric": "SSIM", "Value": ssim_v})
            if not np.isnan(snr):
                rows.append({"Method": label, "Metric": "SNR", "Value": snr})

    plot_df = pd.DataFrame(rows)

    for metric in ["PSNR", "SSIM", "SNR"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        sub = plot_df[plot_df["Metric"] == metric]
        sns.boxplot(x="Method", y="Value", data=sub, palette="Set2", ax=ax)
        sns.stripplot(x="Method", y="Value", data=sub, color="black", alpha=0.4, size=4, ax=ax)
        ax.set_title(f"{metric} Comparison Across Methods", fontsize=14, fontweight="bold")
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xlabel("")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"boxplot_{metric.lower()}.png"), dpi=300)
        plt.close()
        print(f"  ✔ boxplot_{metric.lower()}.png")


def plot_bar_comparison(df):
    """Grouped bar chart: mean PSNR/SSIM per method."""
    methods = []
    for col in df.columns:
        if col.endswith("_PSNR") and col != "Patient":
            methods.append(col.replace("_PSNR", ""))

    labels = [m if m != "Classical" else "Ours (Physics)" for m in methods]

    psnr_means = [df[f"{m}_PSNR"].mean() for m in methods]
    ssim_means = [df[f"{m}_SSIM"].mean() for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

    axes[0].bar(labels, psnr_means, color=colors[:len(methods)], edgecolor="black")
    axes[0].set_title("Mean PSNR by Method", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("PSNR (dB)")
    for i, v in enumerate(psnr_means):
        axes[0].text(i, v + 0.2, f"{v:.2f}", ha="center", fontweight="bold")

    axes[1].bar(labels, ssim_means, color=colors[:len(methods)], edgecolor="black")
    axes[1].set_title("Mean SSIM by Method", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("SSIM")
    for i, v in enumerate(ssim_means):
        axes[1].text(i, v + 0.005, f"{v:.4f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "bar_comparison.png"), dpi=300)
    plt.close()
    print("  ✔ bar_comparison.png")


def plot_visual_slices():
    """Side-by-side visual comparison of a single slice across all methods."""
    _, test_loader = get_dataloaders(batch_size=1)

    unet = UNet()
    unet.load_state_dict(torch.load(os.path.join(MODEL_DIR, "unet_model.pth"), map_location=DEVICE))
    unet.eval()

    dncnn = DnCNN()
    dncnn.load_state_dict(torch.load(os.path.join(MODEL_DIR, "dncnn_model.pth"), map_location=DEVICE))
    dncnn.eval()

    # Pick 3 representative slices
    samples = []
    for lf_t, hf_t, pid, si in test_loader:
        if len(samples) >= 3:
            break
        with torch.no_grad():
            unet_out = unet(lf_t)[0, 0].numpy().clip(0, 1)
            dncnn_out = dncnn(lf_t)[0, 0].numpy().clip(0, 1)

        samples.append({
            "pid": pid[0],
            "lf": lf_t[0, 0].numpy(),
            "hf": hf_t[0, 0].numpy(),
            "unet": unet_out,
            "dncnn": dncnn_out,
        })

    fig, axes = plt.subplots(len(samples), 4, figsize=(16, 4 * len(samples)))
    titles = ["LF Input", "DnCNN Output", "U-Net Output", "HF Ground Truth"]

    for r, s in enumerate(samples):
        images = [s["lf"], s["dncnn"], s["unet"], s["hf"]]
        for c, (img, title) in enumerate(zip(images, titles)):
            ax = axes[r, c] if len(samples) > 1 else axes[c]
            ax.imshow(img.T, cmap="gray", origin="lower")
            if r == 0:
                ax.set_title(title, fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"Patient {s['pid']}", fontsize=11, fontweight="bold")
            ax.axis("off")

    plt.suptitle("Visual Slice Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "visual_comparison.png"), dpi=300)
    plt.close()
    print("  ✔ visual_comparison.png")


def print_summary_table(df):
    """Print a clean comparison table to console and save to CSV."""
    methods = []
    for col in df.columns:
        if col.endswith("_PSNR") and col != "Patient":
            methods.append(col.replace("_PSNR", ""))

    print("\n" + "=" * 65)
    print("METHOD COMPARISON SUMMARY")
    print("=" * 65)
    print(f"{'Method':16s} | {'PSNR':>12s} | {'SSIM':>14s} | {'SNR':>12s}")
    print("-" * 65)

    table_rows = []
    for m in methods:
        label = m if m != "Classical" else "Ours (Physics)"
        psnr = df[f"{m}_PSNR"].dropna()
        ssim_v = df[f"{m}_SSIM"].dropna()
        snr = df[f"{m}_SNR"].dropna()
        print(f"{label:16s} | {psnr.mean():5.2f}±{psnr.std():4.2f} "
              f"| {ssim_v.mean():.4f}±{ssim_v.std():.4f} "
              f"| {snr.mean():5.2f}±{snr.std():4.2f}")
        table_rows.append({
            "Method": label,
            "PSNR_mean": psnr.mean(), "PSNR_std": psnr.std(),
            "SSIM_mean": ssim_v.mean(), "SSIM_std": ssim_v.std(),
            "SNR_mean": snr.mean(), "SNR_std": snr.std(),
        })

    pd.DataFrame(table_rows).to_csv(
        os.path.join(RESULTS_DIR, "method_comparison.csv"), index=False
    )
    print(f"\n✔ Comparison table saved to {RESULTS_DIR}/method_comparison.csv")


def main():
    print("Loading patient-level results...")
    df = load_results()

    print("\nGenerating comparison visualizations...")
    plot_boxplots(df)
    plot_bar_comparison(df)
    plot_visual_slices()
    print_summary_table(df)

    print("\n✔ All comparison outputs generated in", RESULTS_DIR)


if __name__ == "__main__":
    main()
