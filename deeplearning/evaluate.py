# ============================================================
# EVALUATION — COMPARE ALL 4 METHODS ON TEST SET
# ============================================================
import os
import sys
import glob
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch

sys.path.insert(0, os.path.dirname(__file__))

from dataset_loader import get_dataloaders, LF_DIR, HF_DIR, EXCLUDE
from unet import UNet
from dncnn import DnCNN
from skimage.metrics import structural_similarity as ssim

DEVICE = torch.device("cpu")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
CLASSICAL_DIR = r"D:\lowfieldPipeline\outputs"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# METRICS
# ============================================================
def compute_psnr(a, b, data_range=1.0):
    mse = np.mean((a - b) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * np.log10(data_range / np.sqrt(mse))


def compute_snr(img):
    vals = img[img > 0]
    if len(vals) < 100:
        return np.nan
    signal = np.mean(vals[vals > np.percentile(vals, 70)])
    noise = np.std(vals[vals < np.percentile(vals, 30)])
    return signal / (noise + 1e-8)


def compute_ssim_2d(a, b):
    return ssim(a, b, data_range=1.0)


# ============================================================
# MAIN EVALUATION
# ============================================================
def main():
    print("Loading test set...")
    _, test_loader = get_dataloaders(batch_size=1)
    print(f"Test samples: {len(test_loader)}")

    # Load trained models
    unet = UNet()
    unet.load_state_dict(torch.load(os.path.join(MODEL_DIR, "unet_model.pth"), map_location=DEVICE))
    unet.eval()

    dncnn = DnCNN()
    dncnn.load_state_dict(torch.load(os.path.join(MODEL_DIR, "dncnn_model.pth"), map_location=DEVICE))
    dncnn.eval()

    results = []

    with torch.no_grad():
        for lf_t, hf_t, pid_list, slice_idx_list in test_loader:
            pid = pid_list[0]
            si = slice_idx_list[0].item()

            lf_np = lf_t[0, 0].numpy()
            hf_np = hf_t[0, 0].numpy()

            # DL model predictions
            unet_out = unet(lf_t)[0, 0].numpy().clip(0, 1)
            dncnn_out = dncnn(lf_t)[0, 0].numpy().clip(0, 1)

            # Metrics for LF (baseline)
            lf_psnr = compute_psnr(lf_np, hf_np)
            lf_ssim = compute_ssim_2d(lf_np, hf_np)
            lf_snr = compute_snr(lf_np)

            # Metrics for U-Net
            unet_psnr = compute_psnr(unet_out, hf_np)
            unet_ssim = compute_ssim_2d(unet_out, hf_np)
            unet_snr = compute_snr(unet_out)

            # Metrics for DnCNN
            dncnn_psnr = compute_psnr(dncnn_out, hf_np)
            dncnn_ssim = compute_ssim_2d(dncnn_out, hf_np)
            dncnn_snr = compute_snr(dncnn_out)

            results.append({
                "Patient": pid, "Slice": si,
                "LF_PSNR": lf_psnr, "LF_SSIM": lf_ssim, "LF_SNR": lf_snr,
                "UNet_PSNR": unet_psnr, "UNet_SSIM": unet_ssim, "UNet_SNR": unet_snr,
                "DnCNN_PSNR": dncnn_psnr, "DnCNN_SSIM": dncnn_ssim, "DnCNN_SNR": dncnn_snr,
            })

    # Add classical pipeline results (per-patient average from report)
    # We'll load from the filtered report
    classical_metrics = load_classical_metrics()

    # Save raw slice-level results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "slice_level_results.csv"), index=False)

    # Aggregate per-patient
    patient_avg = df.groupby("Patient").agg({
        "LF_PSNR": "mean", "LF_SSIM": "mean", "LF_SNR": "mean",
        "UNet_PSNR": "mean", "UNet_SSIM": "mean", "UNet_SNR": "mean",
        "DnCNN_PSNR": "mean", "DnCNN_SSIM": "mean", "DnCNN_SNR": "mean",
    }).reset_index()

    # Merge classical results
    if classical_metrics:
        cl_df = pd.DataFrame(classical_metrics)
        patient_avg = patient_avg.merge(cl_df, on="Patient", how="left")

    patient_avg.to_csv(os.path.join(RESULTS_DIR, "patient_level_results.csv"), index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS (Mean ± Std)")
    print("=" * 60)
    for method in ["LF", "UNet", "DnCNN"]:
        psnr_col = f"{method}_PSNR"
        ssim_col = f"{method}_SSIM"
        snr_col = f"{method}_SNR"
        print(f"{method:8s} | PSNR={df[psnr_col].mean():.2f}±{df[psnr_col].std():.2f} "
              f"| SSIM={df[ssim_col].mean():.4f}±{df[ssim_col].std():.4f} "
              f"| SNR={df[snr_col].mean():.2f}±{df[snr_col].std():.2f}")

    if "Classical_PSNR" in patient_avg.columns:
        cl_psnr = patient_avg["Classical_PSNR"].dropna()
        cl_ssim = patient_avg["Classical_SSIM"].dropna()
        cl_snr = patient_avg["Classical_SNR"].dropna()
        print(f"{'Ours':8s} | PSNR={cl_psnr.mean():.2f}±{cl_psnr.std():.2f} "
              f"| SSIM={cl_ssim.mean():.4f}±{cl_ssim.std():.4f} "
              f"| SNR={cl_snr.mean():.2f}±{cl_snr.std():.2f}")

    print(f"\n✔ Results saved to {RESULTS_DIR}")


def load_classical_metrics():
    """Parse classical pipeline metrics from the filtered report."""
    import re
    report_path = r"D:\lowfieldPipeline\report.txt"
    if not os.path.exists(report_path):
        return []

    results = []
    with open(report_path, "r") as f:
        content = f.read()

    # Find all PATIENT blocks with their PSNR/SSIM
    blocks = re.findall(
        r"PATIENT (\d{4}).*?Final Enhanced.*?SNR=\s*([\d.]+).*?PSNR=([\d.]+),\s*SSIM=([\d.]+)",
        content, re.DOTALL
    )
    for pid, snr, psnr, ssim_v in blocks:
        results.append({
            "Patient": pid,
            "Classical_PSNR": float(psnr),
            "Classical_SSIM": float(ssim_v),
            "Classical_SNR": float(snr),
        })
    return results


if __name__ == "__main__":
    main()
