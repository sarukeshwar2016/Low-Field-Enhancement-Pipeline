import os
import glob
import time
import random
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
from scipy.ndimage import gaussian_filter

# ==================================================
# SETTINGS (PHYSICALLY REALISTIC)
# ==================================================
HF_DIR = r"D:\01_MRI_Data\nifti_output\high_field_nifti"
LF_DIR = r"D:\01_MRI_Data\nifti_output\low_field_simulated"

LF_SPACING = (1.5, 1.5, 3.0)

# ==================================================
# UTILS
# ==================================================
def compute_snr(img):
    img = img.astype(np.float32)
    vals = img[img > 0]
    if len(vals) < 100: return np.nan
    signal = np.mean(vals[vals > np.percentile(vals, 70)])
    noise  = np.std(vals[vals < np.percentile(vals, 30)])
    return signal / (noise + 1e-8) if noise > 1e-6 else np.nan

def is_valid_scan(img, spacing):
    slices = img.shape[2]
    return (slices >= 10 and spacing[2] <= 6.0 and np.std(img) >= 20.0)

# ==================================================
# LOW-FIELD SIMULATION
# ==================================================
def simulate_low_field(hf_path, out_dir, patient_id):
    os.makedirs(out_dir, exist_ok=True)
    # Correct filename: PID_LF.nii.gz to match enhancement pipelines
    final_path = os.path.join(out_dir, f"{patient_id}_LF.nii.gz")

    try:
        nii_hf = nib.load(hf_path)
        img_hf = nii_hf.get_fdata().astype(np.float32)
        spacing = nii_hf.header.get_zooms()[:3]

        if not is_valid_scan(img_hf, spacing):
            print(f"    Skipping {patient_id}: Scan filter failed.")
            return None

        hf_snr = compute_snr(img_hf)
        
        # 1. RESAMPLE
        nii_res = resample_to_output(nii_hf, voxel_sizes=LF_SPACING)
        img_res = nii_res.get_fdata().astype(np.float32)
        affine  = nii_res.affine
        header  = nii_res.header.copy()

        # 2. BLUR
        img_blur = gaussian_filter(img_res, sigma=[0.6, 0.6, 1.2])

        # 3. NOISE (Target SNR: 25% - 40% of HF)
        target_snr = hf_snr * random.uniform(0.25, 0.40)
        signal_level = np.mean(img_blur[img_blur > np.percentile(img_blur, 50)])
        noise_std = signal_level / (target_snr * 0.45)

        n1 = np.random.normal(0, noise_std, img_blur.shape)
        n2 = np.random.normal(0, noise_std, img_blur.shape)
        img_noisy = np.sqrt((img_blur + n1)**2 + n2**2)

        # 4. SAFETY SNR DROP
        lf_snr = compute_snr(img_noisy)
        attempts = 0
        while not np.isnan(lf_snr) and lf_snr >= hf_snr and attempts < 5:
            noise_std *= 1.2
            n1 = np.random.normal(0, noise_std, img_blur.shape)
            n2 = np.random.normal(0, noise_std, img_blur.shape)
            img_noisy = np.sqrt((img_blur + n1)**2 + n2**2)
            lf_snr = compute_snr(img_noisy)
            attempts += 1

        # 5. MEAN PRESERVATION
        hf_mean = np.mean(img_hf[img_hf > 0])
        lf_mean = np.mean(img_noisy[img_noisy > 0])
        img_final = np.clip(img_noisy * (hf_mean / (lf_mean + 1e-8)), 0, None)

        # SAVE
        nib.save(nib.Nifti1Image(img_final.astype(np.float32), affine, header), final_path)
        print(f"    SUCCESS: {patient_id}_LF.nii.gz (Final SNR: {compute_snr(img_final):.2f})")
        return final_path

    except Exception as e:
        print(f"    ERROR in {patient_id}: {e}")
        return None

# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    hf_files = sorted(glob.glob(os.path.join(HF_DIR, "*_T2SAG.nii.gz")))
    print(f"Found {len(hf_files)} High-Field volumes. Starting Batch Simulation...")

    success = 0
    start = time.time()

    for hf_path in hf_files:
        filename = os.path.basename(hf_path)
        pid = filename[:4] # 0001
        
        # Check if already processed (though user said they deleted)
        out_name = os.path.join(LF_DIR, f"{pid}_LF.nii.gz")
        
        print(f"Processing {pid}...")
        if simulate_low_field(hf_path, LF_DIR, pid):
            success += 1

    print(f"\nDONE. Successful simulations: {success}")
    print(f"Total time: {(time.time()-start)/60:.1f} min")
