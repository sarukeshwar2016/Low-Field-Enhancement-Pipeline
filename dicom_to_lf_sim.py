import os
import glob
import time
import shutil
import random
import numpy as np
import nibabel as nib
import dicom2nifti
import logging
logging.getLogger("dicom2nifti").setLevel(logging.CRITICAL)

from nibabel.processing import resample_to_output
from scipy.ndimage import gaussian_filter

# ==================================================
# SETTINGS (PHYSICALLY REALISTIC)
# ==================================================
DICOM_ROOT_DIR = r"D:\01_MRI_Data"
HF_NIFTI_DIR   = r"D:\01_MRI_Data\nifti_output\high_field_nifti"
LF_SIM_DIR     = r"D:\01_MRI_Data\nifti_output\low_field_simulated"

# ✅ STRICT: realistic spacing
LF_SPACING = (1.5, 1.5, 3.0)

MAX_SUCCESS = 20

# ==================================================
# ROBUST SNR (BACKGROUND-STD METHOD)
# ==================================================
def compute_snr(img):
    img = img.astype(np.float32)
    vals = img[img > 0]

    if len(vals) < 100:
        return np.nan

    # Signal = mean of top 30% of non-zero pixels
    signal = np.mean(vals[vals > np.percentile(vals, 70)])
    # Noise = std of bottom 30% of non-zero pixels (quasi-background)
    noise  = np.std(vals[vals < np.percentile(vals, 30)])

    if noise < 1e-6:
        return np.nan

    return signal / (noise + 1e-8)

# ==================================================
# VALID SCAN FILTER (STRICT)
# ==================================================
def is_valid_scan(img, spacing):
    slices = img.shape[2]
    # Filter for realistic clinical scans (no localizers or bad data)
    return (
        slices >= 10 and            # Reject few-slice localizers
        spacing[2] <= 6.0 and        # Reject ultra-thick slices
        np.std(img) >= 20.0          # Reject noise-only or flat scans
    )

# ==================================================
# DICOM → NIFTI
# ==================================================
def convert_dicom_to_nifti(dicom_dir, out_dir, patient_id):
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(out_dir, "temp_" + patient_id)
    os.makedirs(temp_dir, exist_ok=True)

    try:
        dicom2nifti.convert_directory(dicom_dir, temp_dir, compression=True, reorient=True)
        files = glob.glob(os.path.join(temp_dir, "*.nii.gz"))

        if not files:
            raise RuntimeError("No NIfTI produced")

        final_path = os.path.join(out_dir, f"{patient_id}_HF.nii.gz")
        shutil.move(files[0], final_path)
        shutil.rmtree(temp_dir)
        return final_path

    except:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None

# ==================================================
# LOW-FIELD SIMULATION (DETERMINISTIC & PHYSICS-DRIVEN)
# ==================================================
def simulate_low_field(hf_path, out_dir, patient_id):
    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, f"{patient_id}_LF.nii.gz")

    try:
        nii_hf = nib.load(hf_path)
        img_hf = nii_hf.get_fdata().astype(np.float32)
        spacing = nii_hf.header.get_zooms()[:3]

        if not is_valid_scan(img_hf, spacing):
            print(f"    Skipping: {patient_id} (Scan Filter failed: slices={img_hf.shape[2]}, std={np.std(img_hf):.1f})")
            return None

        hf_snr = compute_snr(img_hf)
        print(f"    HF: shape={img_hf.shape}, SNR={hf_snr:.2f}")

        # ==================================================
        # 1. RESAMPLE TO LOW-FIELD SPACING
        # ==================================================
        nii_res = resample_to_output(nii_hf, voxel_sizes=LF_SPACING)
        img_res = nii_res.get_fdata().astype(np.float32)
        affine  = nii_res.affine
        header  = nii_res.header.copy()

        # ==================================================
        # 2. BLUR (PSF SIMULATION) - BEFORE NOISE
        # ==================================================
        img_blur = gaussian_filter(img_res, sigma=[0.6, 0.6, 1.2])

        # ==================================================
        # 3. RICIAN NOISE INJECTION
        # ==================================================
        # Strict target: 25% - 40% of High Field SNR
        target_ratio = random.uniform(0.25, 0.40)
        target_snr   = hf_snr * target_ratio
        
        # Estimate signal level for noise scaling
        signal_level = np.mean(img_blur[img_blur > np.percentile(img_blur, 50)])
        
        # Corrected formula for noise scale
        # noise_std = signal_level / (target_snr * 0.45)
        # Higher denominator = lower noise_std = higher raw SNR
        # Lower denominator (0.45) = stronger noise injection
        noise_std = signal_level / (target_snr * 0.45)

        # Apply Rician Noise (magnitude of complex Gaussian)
        n1 = np.random.normal(0, noise_std, img_blur.shape)
        n2 = np.random.normal(0, noise_std, img_blur.shape)
        img_noisy = np.sqrt((img_blur + n1)**2 + n2**2)

        # ==================================================
        # 4. SNR CORRECTION RULE (GUARANTEE LF < HF)
        # ==================================================
        lf_snr = compute_snr(img_noisy)
        
        # Safety Loop: If SNR is still too high, inject more noise until it drops properly
        attempts = 0
        while not np.isnan(lf_snr) and lf_snr >= hf_snr and attempts < 10:
            noise_std *= 1.2
            n1 = np.random.normal(0, noise_std, img_blur.shape)
            n2 = np.random.normal(0, noise_std, img_blur.shape)
            img_noisy = np.sqrt((img_blur + n1)**2 + n2**2)
            lf_snr = compute_snr(img_noisy)
            attempts += 1
            
        # ==================================================
        # 5. INTENSITY & MEAN PRESERVATION
        # ==================================================
        hf_mean = np.mean(img_hf[img_hf > 0])
        lf_mean = np.mean(img_noisy[img_noisy > 0])
        
        # Scale to match original HF mean within ±5%
        img_final = img_noisy * (hf_mean / (lf_mean + 1e-8))
        img_final = np.clip(img_final, 0, None)

        final_lf_snr = compute_snr(img_final)
        print(f"    LF: SNR={final_lf_snr:.2f} (TargetRatio: {target_ratio:.2f})")

        # Reject if SNR violation still exists (safety)
        if final_lf_snr >= hf_snr:
            print(f"    ❌ Rejecting {patient_id}: LF SNR ({final_lf_snr:.2f}) >= HF SNR ({hf_snr:.2f})")
            return None

        # ==================================================
        # SAVE
        # ==================================================
        nib.save(nib.Nifti1Image(img_final.astype(np.float32), affine, header), final_path)
        print(f"    ✅ Saved: {os.path.basename(final_path)}")

        return final_path

    except Exception as e:
        print(f"    ⚠️ ERROR in {patient_id}: {e}")
        return None

# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PHYSICS-DRIVEN LOW-FIELD MRI SIMULATION")
    print("=" * 60)

    folders = []
    for root, _, files in os.walk(DICOM_ROOT_DIR):
        if any(f.endswith((".dcm", ".ima")) for f in files):
            folders.append(root)

    print(f"Found {len(folders)} DICOM folders\n")

    success = 0
    start = time.time()

    for i, d in enumerate(folders):
        if success >= MAX_SUCCESS:
            break

        # Extract patient ID (Assuming D:\01_MRI_Data\PID\...)
        parts = d.split(os.sep)
        pid = parts[-3] if len(parts) >= 3 else f"P{i:03d}"

        print(f"\n[{i+1}] Processing {pid}")

        hf = convert_dicom_to_nifti(d, HF_NIFTI_DIR, pid)
        if not hf:
            continue

        if simulate_low_field(hf, LF_SIM_DIR, pid):
            success += 1

        print("-" * 50)

    print("\nDONE")
    print(f"Successful simulations: {success}")
    print(f"Total time: {(time.time()-start)/60:.1f} min")