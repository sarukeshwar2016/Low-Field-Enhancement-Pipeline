import os
import glob
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import bm3d
from skimage.restoration import wiener
from skimage import exposure

# ============================================================
# DIRECTORIES
# ============================================================
LF_SIM_DIR   = r"D:\01_MRI_Data\nifti_output\low_field_simulated"
HF_REF_DIR   = r"D:\01_MRI_Data\nifti_output\high_field_nifti"
OUTPUT_DIR   = r"D:\lowfieldPipeline\batch_enhanced"

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# ============================================================
# PARAMETERS (SCALING-ONLY PHYSICS)
# ============================================================
N4_ITERATIONS = [30, 30, 20, 10]
BM3D_SIGMA    = 0.045
WIENER_SIGMA  = 0.8
WIENER_BAL    = 0.25
WIENER_CAP    = 0.85 
CLAHE_CLIP    = 0.0035
GAMMA         = 1.0
SNR_WINDOW    = (0.50, 0.75) 

def compute_snr(img):
    img = img.astype(np.float32)
    vals = img[img > 0]
    if len(vals) < 100: return np.nan
    signal = np.mean(vals[vals > np.percentile(vals, 70)])
    noise  = np.std(vals[vals < np.percentile(vals, 30)])
    return signal / (noise + 1e-8)

def process_volume(lf_path, hf_path):
    p_id = os.path.basename(lf_path).replace("_LF.nii.gz", "")
    print(f"\nProcessing {p_id}...")

    # Load HF Stats
    nii_hf = nib.load(hf_path)
    hf_data = nii_hf.get_fdata().astype(np.float32)
    hf_snr  = compute_snr(hf_data)
    hf_mean = np.mean(hf_data[hf_data > 0])
    hf_std  = np.std(hf_data[hf_data > 0])

    # Load LF
    nii_lf = nib.load(lf_path)
    image_itk = sitk.ReadImage(lf_path, sitk.sitkFloat32)
    orig_mean = np.mean(sitk.GetArrayFromImage(image_itk))
    affine, header = nii_lf.affine, nii_lf.header

    # 1. N4 Correction (Scaling-Only Preservation)
    print("  - N4 Bias Correction...")
    mask_itk = sitk.OtsuThreshold(image_itk, 0, 1)
    mask_itk = sitk.BinaryMorphologicalClosing(mask_itk, [3,3,3])
    mask_itk = sitk.BinaryFillhole(mask_itk)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(N4_ITERATIONS)
    corrected_itk = corrector.Execute(image_itk, mask_itk)
    n4_arr = sitk.GetArrayFromImage(corrected_itk)
    # Intensity match to original
    n4_arr *= (orig_mean / (n4_arr.mean() + 1e-8))
    # Normalize [0,1]
    p1, p99 = np.percentile(n4_arr, [1, 99])
    data = np.clip((n4_arr - p1) / (p99 - p1 + 1e-8), 0, 1)

    # 2. BM3D (Hard Thresholding Only)
    print(f"  - BM3D Denoising (sigma={BM3D_SIGMA})...")
    bm_out = np.zeros_like(data)
    for i in range(data.shape[2]):
        bm_out[:,:,i] = np.clip(bm3d.bm3d(data[:,:,i], sigma_psd=BM3D_SIGMA, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING), 0, 1)
    data = bm_out

    # 3. Wiener (Energy preservation + Scaling Cap)
    print(f"  - Wiener Deconvolution (bal={WIENER_BAL})...")
    psf_size = 11; ax = np.arange(psf_size)-psf_size//2; xx,yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2+yy**2)/(2.*WIENER_SIGMA**2)); psf /= psf.sum()
    wi_out = np.zeros_like(data)
    for i in range(data.shape[2]):
        wi_out[:,:,i] = np.clip(wiener(data[:,:,i], psf, balance=WIENER_BAL), 0, 1)
    # Energy Preserve
    orig_energy = np.mean(data**2)
    new_energy  = np.mean(wi_out**2)
    data = wi_out * np.sqrt(orig_energy / (new_energy + 1e-8))
    # SNR Cap (85% HF)
    w_snr = compute_snr(data)
    if w_snr > WIENER_CAP * hf_snr:
        data *= (WIENER_CAP * hf_snr / (w_snr + 1e-8))

    # 4. CLAHE
    print(f"  - CLAHE enhancement (clip={CLAHE_CLIP})...")
    cl_out = np.zeros_like(data)
    for i in range(data.shape[2]):
         cl_out[:,:,i] = exposure.equalize_adapthist(data[:,:,i], clip_limit=CLAHE_CLIP)
    data = np.clip(cl_out, 0, None)
    # Mean Correction (Mandatory HF Match)
    data *= (hf_mean / (np.mean(data[data>0]) + 1e-8))

    # 5. Final (Gamma + Scaling-Only Guards)
    print(f"  - Final Normalization...")
    data = np.power(data, GAMMA)
    # SNR Window (50-75% HF)
    f_snr = compute_snr(data)
    target_max = SNR_WINDOW[1] * hf_snr
    target_min = SNR_WINDOW[0] * hf_snr
    if f_snr > target_max: data *= (target_max / (f_snr + 1e-8))
    elif f_snr < target_min: data *= (target_min / (f_snr + 1e-8))
    
    # STD Safeguard
    if np.std(data[data>0]) < 0.6 * hf_std:
        data *= 1.1

    # Save Result
    final_out = os.path.join(OUTPUT_DIR, f"{p_id}_final_enhanced.nii.gz")
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine, header), final_out)
    print(f"  SUCCESS: {final_out} (Final SNR: {compute_snr(data):.2f})")

def main():
    lf_files = sorted(glob.glob(os.path.join(LF_SIM_DIR, "*_LF.nii.gz")))
    print(f"Found {len(lf_files)} simulated low-field volumes.")
    
    for lf_path in lf_files:
        p_id = os.path.basename(lf_path).replace("_LF.nii.gz", "")
        hf_matches = glob.glob(os.path.join(HF_REF_DIR, f"{p_id}_HF.nii.gz"))
        if not hf_matches: continue
        process_volume(lf_path, hf_matches[0])

if __name__ == "__main__": main()
