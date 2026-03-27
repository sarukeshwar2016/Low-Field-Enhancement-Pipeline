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
# SETTINGS
# ==================================================
DICOM_ROOT_DIR = r"D:\01_MRI_Data"
HF_NIFTI_DIR   = r"D:\01_MRI_Data\nifti_output\high_field_nifti"
LF_SIM_DIR     = r"D:\01_MRI_Data\nifti_output\low_field_simulated"

# Range: 9 to 100
START_INDEX = 9
END_INDEX   = 100

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

def convert_dicom_to_nifti(dicom_dir, out_dir, patient_id):
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(out_dir, "temp_" + patient_id)
    os.makedirs(temp_dir, exist_ok=True)
    try:
        dicom2nifti.convert_directory(dicom_dir, temp_dir, compression=True, reorient=True)
        files = glob.glob(os.path.join(temp_dir, "*.nii.gz"))
        if not files: return None
        final_path = os.path.join(out_dir, f"{patient_id}_HF.nii.gz")
        shutil.move(files[0], final_path)
        shutil.rmtree(temp_dir)
        return final_path
    except:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None

def simulate_low_field(hf_path, out_dir, patient_id):
    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, f"{patient_id}_LF.nii.gz")
    try:
        nii_hf = nib.load(hf_path)
        img_hf = nii_hf.get_fdata().astype(np.float32)
        spacing = nii_hf.header.get_zooms()[:3]
        if not is_valid_scan(img_hf, spacing): return None

        hf_snr = compute_snr(img_hf)
        nii_res = resample_to_output(nii_hf, voxel_sizes=LF_SPACING)
        img_res = nii_res.get_fdata().astype(np.float32)
        affine, header = nii_res.affine, nii_res.header.copy()

        img_blur = gaussian_filter(img_res, sigma=[0.6, 0.6, 1.2])
        target_snr = hf_snr * random.uniform(0.25, 0.40)
        signal_level = np.mean(img_blur[img_blur > np.percentile(img_blur, 50)])
        noise_std = signal_level / (target_snr * 0.45)

        n1 = np.random.normal(0, noise_std, img_blur.shape)
        n2 = np.random.normal(0, noise_std, img_blur.shape)
        img_noisy = np.sqrt((img_blur + n1)**2 + n2**2)

        hf_mean = np.mean(img_hf[img_hf > 0])
        lf_mean = np.mean(img_noisy[img_noisy > 0])
        img_final = np.clip(img_noisy * (hf_mean / (lf_mean + 1e-8)), 0, None)

        if compute_snr(img_final) >= hf_snr: return None

        nib.save(nib.Nifti1Image(img_final.astype(np.float32), affine, header), final_path)
        return final_path
    except: return None

# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    # Get all patient folders (0001, 0002, ...)
    patient_folders = sorted([f for f in os.listdir(DICOM_ROOT_DIR) if os.path.isdir(os.path.join(DICOM_ROOT_DIR, f))])
    
    # Filter for those that look like PIDs (0001 to 0100)
    # The user wants from 9th to 100th image.
    # We index from 0, so 9th is index 8. 100th is index 99.
    target_folders = patient_folders[START_INDEX-1:END_INDEX]

    print(f"Targeting {len(target_folders)} patients (from {target_folders[0]} to {target_folders[-1]})")

    success = 0
    start = time.time()

    for pid in target_folders:
        dicom_parent = os.path.join(DICOM_ROOT_DIR, pid)
        
        # Find a subfolder with DICOM files
        dicom_dir = None
        for root, _, files in os.walk(dicom_parent):
            if any(f.endswith((".dcm", ".ima")) for f in files):
                dicom_dir = root
                break 
        
        if not dicom_dir:
            print(f"Skipping {pid}: No DICOMs found")
            continue

        print(f"Processing {pid}...")
        hf = convert_dicom_to_nifti(dicom_dir, HF_NIFTI_DIR, pid)
        if hf and simulate_low_field(hf, LF_SIM_DIR, pid):
            print(f"  SUCCESS: {pid}")
            success += 1
        else:
            print(f"  FAILED: {pid}")

    print(f"\nDONE. Successful simulations: {success}")
    print(f"Total time: {(time.time()-start)/60:.1f} min")
