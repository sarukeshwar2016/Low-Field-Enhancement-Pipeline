"""
============================================================
FINAL: FULLY CORRECTED MRI ENHANCEMENT - BATCH (0001-0008)
============================================================
Pipeline:
LF → N4 → Intensity Std → Wiener → Histogram Match → Smooth → Contrast → Final
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
from skimage.exposure import match_histograms
from scipy.ndimage import gaussian_filter
from scipy.stats import skew

# ============================================================
# DIRECTORIES
# ============================================================
LF_DIR  = r"D:\01_MRI_Data\nifti_output\low_field_simulated"
HF_DIR  = r"D:\01_MRI_Data\nifti_output\high_field_nifti"
OUT_DIR = r"D:\lowfieldPipeline\batch_enhanced_final"

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
    return 20 * np.log10(np.max(ref) / (np.sqrt(mse) + 1e-8))

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

    # Load Data
    lf_nii = nib.load(lf_path)
    lf_data = lf_nii.get_fdata().astype(np.float32)
    affine, header = lf_nii.affine, lf_nii.header

    hf_nii = nib.load(hf_path)
    hf_orig = hf_nii.get_fdata().astype(np.float32)
    hf_mean = np.mean(hf_orig[hf_orig > 0])
    hf_snr  = compute_snr(hf_orig)

    # 🧪 RESAMPLE HF TO LF GRID
    print("  - Resampling HF Reference...")
    hf_sitk = sitk.ReadImage(hf_path, sitk.sitkFloat32)
    lf_sitk = sitk.ReadImage(lf_path, sitk.sitkFloat32)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(lf_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform())
    hf_aligned = sitk.GetArrayFromImage(resampler.Execute(hf_sitk)).astype(np.float32)
    hf_ref = np.transpose(hf_aligned, (2, 1, 0)) # Aligned HF (X,Y,Z)

    # 1. N4
    print("  - N4 Bias Correction...")
    mask = sitk.OtsuThreshold(lf_sitk, 0, 1, 200)
    mask = sitk.BinaryMorphologicalClosing(mask, [3]*3)
    mask = sitk.BinaryFillhole(mask)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([30,30,20,10])
    data_sitk = corrector.Execute(lf_sitk, mask)
    data = np.transpose(sitk.GetArrayFromImage(data_sitk), (2, 1, 0)).astype(np.float32)
    data = match_mean(data, hf_mean)

    # 2. Intensity Std
    print("  - Intensity Std...")
    p2, p98 = np.percentile(data, [2, 98])
    data = np.clip((data - p2)/(p98 - p2 + 1e-8), 0, 1)
    data = match_mean(data * hf_mean, hf_mean)

    # 3. Wiener
    print("  - Wiener Reconstruction...")
    psf_size = 9; ax = np.arange(psf_size) - psf_size//2; xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2)/(2*0.8**2)); psf /= psf.sum()
    p1, p99 = np.percentile(data, [1, 99])
    norm = np.clip((data - p1)/(p99 - p1 + 1e-8), 0, 1)
    wi_out = np.zeros_like(norm)
    for i in range(norm.shape[2]):
        wi = wiener(norm[:,:,i], psf, balance=0.5)
        wi_out[:,:,i] = np.clip(wi, 0, 1)
    data = wi_out*(p99-p1)+p1
    data *= np.sqrt(np.mean(hf_ref**2)/(np.mean(data**2)+1e-8))
    data = match_mean(data, hf_mean)
    snr_w = compute_snr(data)
    if snr_w > hf_snr: data *= (hf_snr/(snr_w+1e-8))

    # 4. Histogram Matching
    print("  - Histogram Matching...")
    data = match_histograms(data, hf_ref)
    data = match_mean(data, hf_mean)

    # 5. Mild Smoothing
    print("  - Mild Smoothing...")
    smooth = gaussian_filter(data, sigma=[0.3, 0.3, 0.8])
    data = 0.9*data + 0.1*smooth
    data = match_mean(data, hf_mean)

    # 6. Micro Contrast
    print("  - Micro Contrast Adj...")
    mean_val = np.mean(data[data > 0])
    data = (data - mean_val)*1.02 + mean_val
    data = match_mean(data, hf_mean)

    # Final
    final = np.clip(data, 0, None)

    # Save
    out_name = os.path.join(OUT_DIR, f"{p_id}_final_enhanced.nii.gz")
    nib.save(nib.Nifti1Image(final.astype(np.float32), affine, header), out_name)
    
    # Metrics
    psnr_val = compute_psnr(hf_ref, final)
    ssim_val = compute_ssim_3d(hf_ref, final)
    print(f"  SUCCESS: {out_name} (SNR: {compute_snr(final):.2f}, SSIM: {ssim_val:.4f})")
    
    return {"id": p_id, "snr": compute_snr(final), "psnr": psnr_val, "ssim": ssim_val}

def main():
    lf_files = sorted(glob.glob(os.path.join(LF_DIR, "*_LF.nii.gz")))
    print(f"Found {len(lf_files)} volumes. Starting final batch...")
    
    results = []
    for lf_path in lf_files:
        p_id = os.path.basename(lf_path).replace("_LF.nii.gz", "")
        hf_matches = glob.glob(os.path.join(HF_DIR, f"{p_id}_HF.nii.gz"))
        if not hf_matches: continue
        try:
            results.append(process_patient(lf_path, hf_matches[0]))
        except Exception as e:
            print(f"  FAILED {p_id}: {str(e)}")

    print("\n" + "="*55)
    print("FINAL BATCH SUMMARY")
    print("="*55)
    print(f"{'Patient':<10s} | {'SNR':>8s} | {'PSNR':>8s} | {'SSIM':>8s}")
    for r in results:
        print(f"{r['id']:<10s} | {r['snr']:>8.2f} | {r['psnr']:>8.2f} | {r['ssim']:>8.4f}")

if __name__ == "__main__": main()
