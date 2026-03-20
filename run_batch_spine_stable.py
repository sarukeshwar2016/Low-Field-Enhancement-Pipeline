"""
============================================================
FINAL: STABLE SPINE MRI ENHANCEMENT - BATCH (RESEARCH)
============================================================
Pipeline:
LF → N4 → Intensity Std → Wiener → Smooth → Micro Contrast → Final
Metrics: PSNR, SSIM, Hist Overlap (HF resampled to LF)
============================================================
"""

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
# DIRECTORIES
# ============================================================
LF_DIR  = r"D:\01_MRI_Data\nifti_output\low_field_simulated"
HF_DIR  = r"D:\01_MRI_Data\nifti_output\high_field_nifti"
OUT_DIR = r"D:\lowfieldPipeline\batch_enhanced_stable"

if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)

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
    if len(vals) == 0: return img
    return img * (target / (np.mean(vals) + 1e-8))

def compute_psnr(ref, img):
    mse = np.mean((ref - img)**2)
    if mse < 1e-10: return 100
    return 20 * np.log10(np.max(ref) / np.sqrt(mse))

def compute_ssim_3d(ref, img):
    scores = []
    for i in range(ref.shape[2]):
        r, t = ref[:,:,i], img[:,:,i]
        dr = r.max() - r.min()
        if dr < 1e-6 or np.std(r) < 1e-6 or np.std(t) < 1e-6: continue
        scores.append(ssim(r, t, data_range=dr))
    return np.mean(scores) if scores else 0.0

# ============================================================
# PROCESS PATIENT
# ============================================================
def process_patient(lf_path, hf_path):
    p_id = os.path.basename(lf_path).replace("_LF.nii.gz", "")
    print(f"\nProcessing {p_id}...")

    # Load LF
    lf_nii = nib.load(lf_path)
    lf_data = lf_nii.get_fdata().astype(np.float32)
    affine, header = lf_nii.affine, lf_nii.header

    # Load HF for stats
    hf_nii = nib.load(hf_path)
    hf_orig = hf_nii.get_fdata().astype(np.float32)
    hf_mean = np.mean(hf_orig[hf_orig > 0])
    hf_std  = np.std(hf_orig[hf_orig > 0])
    hf_snr  = compute_snr(hf_orig)

    # 🧪 RESAMPLE HF TO LF GRID (FOR METRICS)
    hf_sitk = sitk.ReadImage(hf_path, sitk.sitkFloat32)
    lf_sitk = sitk.ReadImage(lf_path, sitk.sitkFloat32)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(lf_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform())
    hf_aligned = sitk.GetArrayFromImage(resampler.Execute(hf_sitk)).astype(np.float32)
    hf_ref = np.transpose(hf_aligned, (2, 1, 0)) # Aligned HF for metrics

    # 1. N4
    print("  - N4 Bias Correction...")
    mask = sitk.OtsuThreshold(lf_sitk, 0, 1, 200)
    mask = sitk.BinaryMorphologicalClosing(mask, [3]*3)
    mask = sitk.BinaryFillhole(mask)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([30,30,20,10])
    n4 = sitk.GetArrayFromImage(corrector.Execute(lf_sitk, mask)).astype(np.float32)
    n4 = match_mean(n4, hf_mean)

    # 2. Intensity Standardization
    print("  - Intensity Standardization...")
    p2, p98 = np.percentile(n4, [2, 98])
    n4_std = np.clip((n4 - p2)/(p98 - p2 + 1e-8), 0, 1)
    data = match_mean(n4_std * hf_mean, hf_mean)

    # 3. Wiener
    print("  - Wiener Reconstruction...")
    psf_size = 9; ax = np.arange(psf_size) - psf_size//2; xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2)/(2*0.8**2)); psf /= psf.sum()
    p1, p99 = np.percentile(data, [1, 99])
    norm = np.clip((data - p1)/(p99 - p1 + 1e-8), 0, 1)
    wi_out = np.zeros_like(norm)
    for i in range(norm.shape[2]):
        wi_out[:,:,i] = np.clip(wiener(norm[:,:,i], psf, balance=0.3), 0, 1)
    data = wi_out*(p99-p1)+p1
    data *= np.sqrt(np.mean(n4**2)/(np.mean(data**2)+1e-8))
    data = match_mean(data, hf_mean)
    snr_wiener = compute_snr(data)
    if snr_wiener > 0.9 * hf_snr: data *= (0.9 * hf_snr)/(snr_wiener + 1e-8)

    # 4. Smoothing
    print("  - Mild Smoothing...")
    smooth = gaussian_filter(data, sigma=[0.3, 0.3, 0.8])
    data = 0.9 * data + 0.1 * smooth
    data = match_mean(data, hf_mean)

    # 5. Micro Contrast
    print("  - Micro Contrast Adj...")
    mean_val = np.mean(data[data > 0])
    data = (data - mean_val) * 1.02 + mean_val
    data = match_mean(data, hf_mean)
    data = np.clip(data, 0, None)

    # 6. Final Scaling
    print("  - Final Scaling...")
    final = data.copy()
    snr_final = compute_snr(final)
    t_lo, t_hi = 0.75 * hf_snr, 0.90 * hf_snr
    if snr_final < t_lo: final *= t_lo / (snr_final + 1e-8)
    elif snr_final > t_hi: final *= t_hi / (snr_final + 1e-8)
    if np.std(final[final > 0]) < 0.7 * hf_std: final *= 1.1
    final = match_mean(final, hf_mean)
    final = np.clip(final, 0, None)

    # Save
    out_name = os.path.join(OUT_DIR, f"{p_id}_final_enhanced.nii.gz")
    nib.save(nib.Nifti1Image(final.astype(np.float32), affine, header), out_name)
    
    # Results
    psnr_val = compute_psnr(hf_ref, final)
    ssim_val = compute_ssim_3d(hf_ref, final)
    print(f"  SUCCESS: {out_name} (SNR: {compute_snr(final):.2f}, PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f})")
    return {"id": p_id, "snr": compute_snr(final), "psnr": psnr_val, "ssim": ssim_val}

def main():
    lf_files = sorted(glob.glob(os.path.join(LF_DIR, "*_LF.nii.gz")))
    print(f"Found {len(lf_files)} simulated volumes. Starting batch...")
    
    results = []
    for lf_path in lf_files:
        p_id = os.path.basename(lf_path).replace("_LF.nii.gz", "")
        hf_matches = glob.glob(os.path.join(HF_DIR, f"{p_id}_HF.nii.gz"))
        if not hf_matches: continue
        results.append(process_patient(lf_path, hf_matches[0]))

    print("\n" + "="*55)
    print("BATCH SUMMARY REPORT")
    print("="*55)
    print(f"{'Patient':<10s} | {'SNR':>8s} | {'PSNR':>8s} | {'SSIM':>8s}")
    for r in results:
        print(f"{r['id']:<10s} | {r['snr']:>8.2f} | {r['psnr']:>8.2f} | {r['ssim']:>8.4f}")

if __name__ == "__main__": main()
