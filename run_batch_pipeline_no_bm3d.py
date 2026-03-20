"""
============================================================
LINEAR PHYSICS-LOCKED MRI ENHANCEMENT (NO BM3D) - BATCH
============================================================
LF → N4 → Gaussian → Wiener → CLAHE → Final
Processes all volumes in the simulated directory.
============================================================
"""

import os
import glob
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from skimage.restoration import wiener
from skimage import exposure
from scipy.stats import skew

# ============================================================
# DIRECTORIES
# ============================================================
LF_DIR  = r"D:\01_MRI_Data\nifti_output\low_field_simulated"
HF_DIR  = r"D:\01_MRI_Data\nifti_output\high_field_nifti"
OUT_DIR = r"D:\lowfieldPipeline\batch_enhanced_no_bm3d"

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

def match_std(img, target_std):
    vals = img[img > 0]
    if len(vals) == 0: return img
    curr_std = np.std(vals)
    return img * (target_std / (curr_std + 1e-8))

# ============================================================
# PROCESS PATIENT
# ============================================================
def process_patient(lf_path, hf_path):
    p_id = os.path.basename(lf_path).replace("_LF.nii.gz", "")
    print(f"\nProcessing {p_id}...")

    # Load Data
    nii_hf = nib.load(hf_path)
    hf = nii_hf.get_fdata().astype(np.float32)
    
    nii_lf = nib.load(lf_path)
    lf = nii_lf.get_fdata().astype(np.float32)
    affine, header = nii_lf.affine, nii_lf.header

    hf_mean = np.mean(hf[hf > 0])
    hf_std  = np.std(hf[hf > 0])
    hf_snr  = compute_snr(hf)

    # 1. N4
    print("  - N4 Bias Correction...")
    sitk_img = sitk.GetImageFromArray(lf)
    mask = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
    mask = sitk.BinaryMorphologicalClosing(mask, [3,3,3])
    mask = sitk.BinaryFillhole(mask)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([30,30,20,10])
    n4 = sitk.GetArrayFromImage(corrector.Execute(sitk_img, mask)).astype(np.float32)
    n4 = match_mean(n4, hf_mean)

    # 2. Gaussian
    print("  - Gaussian Denoising...")
    gauss = gaussian_filter(n4, sigma=[0.5, 0.5, 1.0])
    gauss = match_mean(gauss, hf_mean)
    gauss = match_std(gauss, np.std(n4[n4 > 0]))
    snr_g = compute_snr(gauss)
    if snr_g > 0.75 * hf_snr: gauss *= (0.75 * hf_snr) / (snr_g + 1e-8)

    # 3. Wiener
    print("  - Wiener Deconvolution...")
    psf_size = 9; ax = np.arange(psf_size) - psf_size//2; xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2)/(2*0.8**2)); psf /= psf.sum()
    p1, p99 = np.percentile(gauss, [1,99])
    norm = np.clip((gauss - p1)/(p99-p1+1e-8), 0, 1)
    wi_out = np.zeros_like(norm)
    for i in range(norm.shape[2]):
        wi = wiener(norm[:,:,i], psf, balance=0.25)
        wi_out[:,:,i] = np.clip(wi, 0, 1)
    wi_data = wi_out*(p99-p1)+p1
    wi_data *= np.sqrt(np.mean(gauss**2)/(np.mean(wi_data**2)+1e-8))
    wi_data = match_mean(wi_data, hf_mean)

    # 4. CLAHE
    print("  - CLAHE Enhancement...")
    p1, p99 = np.percentile(wi_data, [1,99])
    norm = np.clip((wi_data - p1)/(p99-p1+1e-8), 0, 1)
    cl_out = np.zeros_like(norm)
    for i in range(norm.shape[2]): cl_out[:,:,i] = exposure.equalize_adapthist(norm[:,:,i], clip_limit=0.0025)
    clahe_data = cl_out*(p99-p1)+p1
    clahe_data = match_mean(clahe_data, hf_mean)
    clahe_data = match_std(clahe_data, np.std(wi_data[wi_data > 0]))

    # 5. Final
    print("  - Final Scaling & Cleanup...")
    final = clahe_data.copy()
    snr = compute_snr(final)
    t_lo, t_hi = 0.75 * hf_snr, 0.90 * hf_snr
    if snr < t_lo: final *= t_lo / (snr + 1e-8)
    elif snr > t_hi: final *= t_hi / (snr + 1e-8)
    if np.std(final[final > 0]) < 0.7 * hf_std: final *= 1.1
    final = match_mean(final, hf_mean)

    # Save
    out_name = os.path.join(OUT_DIR, f"{p_id}_final_enhanced.nii.gz")
    nib.save(nib.Nifti1Image(final.astype(np.float32), affine, header), out_name)
    print(f"  SUCCESS: {out_name} (SNR: {compute_snr(final):.2f})")

def main():
    lf_files = sorted(glob.glob(os.path.join(LF_DIR, "*_LF.nii.gz")))
    print(f"Found {len(lf_files)} simulated volumes.")
    for lf_path in lf_files:
        p_id = os.path.basename(lf_path).replace("_LF.nii.gz", "")
        hf_matches = glob.glob(os.path.join(HF_DIR, f"{p_id}_HF.nii.gz"))
        if not hf_matches: continue
        process_patient(lf_path, hf_matches[0])

if __name__ == "__main__": main()
