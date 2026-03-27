# ============================================================
# FINAL: RESEARCH-GRADE MRI ENHANCEMENT PIPELINE (BATCH 9-100)
# ============================================================

import os
import glob
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from skimage.restoration import wiener
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_filter
from scipy.stats import skew

# ============================================================
# PATHS
# ============================================================
LF_DIR  = r"D:\01_MRI_Data\nifti_output\low_field_simulated"
HF_DIR  = r"D:\01_MRI_Data\nifti_output\high_field_nifti"
OUT_DIR = r"D:\lowfieldPipeline\outputs"
REPORT_PATH = r"D:\lowfieldPipeline\final_batch_report.txt"

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# UTILS
# ============================================================
def compute_snr(img):
    vals = img[img > 0]
    if len(vals) < 100: return np.nan
    signal = np.mean(vals[vals > np.percentile(vals, 70)])
    noise  = np.std(vals[vals < np.percentile(vals, 30)])
    return signal / (noise + 1e-8)

def match_mean(img, target):
    vals = img[img > 0]
    return img * (target / (np.mean(vals) + 1e-8))

def stats_line(name, img):
    vals = img[img > 0]
    return f"{name:25s} | Mean={np.mean(vals):8.2f} | Std={np.std(vals):8.2f} | Skew={skew(vals):6.2f} | SNR={compute_snr(img):6.2f}\n"

# ============================================================
# METRICS
# ============================================================
def compute_metrics(hf_ref, img):
    mse = np.mean((hf_ref - img) ** 2)
    psnr = 20 * np.log10(np.max(hf_ref) / np.sqrt(mse)) if mse > 1e-10 else 100

    scores = []
    for i in range(hf_ref.shape[2]):
        r, t = hf_ref[:,:,i], img[:,:,i]
        if np.std(r) < 1e-6 or np.std(t) < 1e-6: continue
        scores.append(ssim(r, t, data_range=r.max()-r.min()))
    ssim_val = np.mean(scores) if scores else 0.0

    # FIXED histogram overlap
    r_v = hf_ref[hf_ref > 0]
    i_v = img[img > 0]
    min_v, max_v = min(r_v.min(), i_v.min()), max(r_v.max(), i_v.max())
    h1, _ = np.histogram(r_v, bins=100, range=(min_v, max_v), density=True)
    h2, _ = np.histogram(i_v, bins=100, range=(min_v, max_v), density=True)
    hist_o = np.sum(np.sqrt(h1 * h2))

    return psnr, ssim_val, hist_o

# ============================================================
# PROCESS PATIENT
# ============================================================
def process_patient(lf_path, hf_path):

    pid = os.path.basename(lf_path)[:4]

    lf_nii = nib.load(lf_path)
    lf = lf_nii.get_fdata().astype(np.float32) # (X, Y, Z)
    affine, header = lf_nii.affine, lf_nii.header

    hf = nib.load(hf_path).get_fdata().astype(np.float32) # (X, Y, Z)
    hf_mean = np.mean(hf[hf > 0])
    hf_snr  = compute_snr(hf)

    report = f"\nPATIENT {pid}\n" + "="*40 + "\n"
    report += stats_line("HF Original", hf)
    report += stats_line("LF Simulated", lf)

    # ============================================================
    # STAGE 1: INTENSITY INHOMOGENEITY CORRECTION (N4)
    # ============================================================
    # ✅ FIX: Transpose to (Z, Y, X) for SITK
    sitk_img = sitk.GetImageFromArray(np.transpose(lf, (2, 1, 0)))
    mask = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
    mask = sitk.BinaryMorphologicalClosing(mask, [3]*3)
    mask = sitk.BinaryFillhole(mask)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([30,30,20,10])

    # ✅ FIX: Get back to (X, Y, Z)
    n4 = np.transpose(sitk.GetArrayFromImage(corrector.Execute(sitk_img, mask)), (2, 1, 0)).astype(np.float32)
    n4 = match_mean(n4, hf_mean)

    report += stats_line("Stage 1: N4", n4)

    # ============================================================
    # STAGE 2: INTENSITY STANDARDIZATION
    # ============================================================
    p2, p98 = np.percentile(n4, [2, 98])
    n4 = np.clip((n4 - p2)/(p98 - p2 + 1e-8), 0, 1)
    n4 = match_mean(n4 * hf_mean, hf_mean)

    report += stats_line("Stage 2: Standardized", n4)

    # ============================================================
    # STAGE 3: WIENER RECONSTRUCTION
    # ============================================================
    psf_size = 9
    ax = np.arange(psf_size) - psf_size//2
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2)/(2*0.8**2))
    psf /= psf.sum()

    p1, p99 = np.percentile(n4, [1, 99])
    norm = np.clip((n4 - p1)/(p99 - p1 + 1e-8), 0, 1)

    w_out = np.zeros_like(norm)
    for i in range(norm.shape[2]):
        w_out[:,:,i] = wiener(norm[:,:,i], psf, balance=0.4)

    w_out = w_out*(p99-p1)+p1
    w_out = match_mean(w_out, hf_mean)

    # SNR safety
    snr_curr = compute_snr(w_out)
    if snr_curr > hf_snr:
        w_out *= hf_snr / (snr_curr + 1e-8)

    report += stats_line("Stage 3: Wiener", w_out)

    # ============================================================
    # STAGE 4: RESOLUTION MODELING
    # ============================================================
    degraded = gaussian_filter(w_out, sigma=[0.2,0.2,0.6])
    w_out = 0.85 * w_out + 0.15 * degraded

    report += stats_line("Stage 4: Resolution Model", w_out)

    # ============================================================
    # STAGE 5: STRUCTURAL REFINEMENT
    # ============================================================
    blur = gaussian_filter(w_out, sigma=0.5)
    sharpened = w_out + 0.1 * (w_out - blur)
    w_out = np.clip(sharpened, 0, None)

    report += stats_line("Stage 5: Refinement", w_out)

    # ============================================================
    # STAGE 6: FINAL SCALING
    # ============================================================
    final = w_out.copy()
    snr = compute_snr(final)

    if snr < 0.7 * hf_snr:
        final *= (0.7 * hf_snr)/(snr + 1e-8)
    elif snr > 0.9 * hf_snr:
        final *= (0.9 * hf_snr)/(snr + 1e-8)

    final = match_mean(final, hf_mean)
    final = np.clip(final, 0, None)

    report += stats_line("Final Enhanced", final)

    # SAVE
    out_path = os.path.join(OUT_DIR, f"{pid}_enhanced.nii.gz")
    nib.save(nib.Nifti1Image(final.astype(np.float32), affine, header), out_path)

    # ALIGN HF
    hf_sitk = sitk.ReadImage(hf_path, sitk.sitkFloat32)
    lf_sitk = sitk.ReadImage(lf_path, sitk.sitkFloat32)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(lf_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)

    hf_aligned = np.transpose(
        sitk.GetArrayFromImage(resampler.Execute(hf_sitk)),
        (2,1,0)
    ).astype(np.float32)

    psnr, ssim_v, hist_o = compute_metrics(hf_aligned, final)

    report += f"\nMETRICS:\nPSNR={psnr:.2f}, SSIM={ssim_v:.4f}, Hist={hist_o:.4f}\n"

    return report, {"id": pid, "snr": compute_snr(final), "psnr": psnr, "ssim": ssim_v}

# ============================================================
# MAIN
# ============================================================
def main():

    # 1. Load existing report to preserve results for patients 1-8
    existing_report = ""
    summary = []
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r") as f:
            lines = f.readlines()
            # Preserve details for 1-8 but strip the old final summary
            for idx, line in enumerate(lines):
                if "FINAL SUMMARY" in line:
                    break
                existing_report += line
                # Extract summary stats for 1-8 if possible
                if "|" in line and "SNR=" in line and len(line.split("|")) == 4:
                    parts = [p.strip() for p in line.split("|")]
                    if parts[0].isdigit() and int(parts[0]) <= 8:
                        pid = parts[0].zfill(4)
                        snr = float(parts[1].split("=")[1])
                        psnr = float(parts[2].split("=")[1])
                        ssim_val = float(parts[3].split("=")[1])
                        summary.append({"id": pid, "snr": snr, "psnr": psnr, "ssim": ssim_val})

    # 2. Identify files to process (starting from 9)
    lf_files = sorted(glob.glob(os.path.join(LF_DIR, "*_LF.nii.gz")))
    
    # Filter for patients 0009 and above
    target_files = [f for f in lf_files if int(os.path.basename(f)[:4]) >= 9]

    print(f"Enhancing {len(target_files)} patients (from {os.path.basename(target_files[0])[:4]})")

    report_content = existing_report
    
    for lf_path in target_files:
        pid = os.path.basename(lf_path)[:4]
        hf_matches = glob.glob(os.path.join(HF_DIR, f"{pid}_HF.nii.gz"))
        if not hf_matches: continue

        chunk, stats = process_patient(lf_path, hf_matches[0])
        report_content += chunk
        summary.append(stats)
        print(f"✔ {pid}")

        # Intermediate save
        with open(REPORT_PATH, "w") as f:
            f.write(report_content)

    # 3. Final Summary
    final_report = report_content + "\nFINAL SUMMARY\n" + "="*40 + "\n"
    # Sort summary by ID
    summary.sort(key=lambda x: x["id"])
    for s in summary:
        final_report += f"{s['id']} | SNR={s['snr']:.2f} | PSNR={s['psnr']:.2f} | SSIM={s['ssim']:.4f}\n"

    with open(REPORT_PATH, "w") as f:
        f.write(final_report)

    print("\nDONE. All details updated in " + REPORT_PATH)

if __name__ == "__main__":
    main()
