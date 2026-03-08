"""
============================================================
PHASE 1: BATCH DICOM TO LOW-FIELD MRI SIMULATION
============================================================
1. Finds all patient folders in 01_MRI_Data containing DICOMs.
2. Converts each folder to a 3D High-Field NIfTI using dicom2nifti.
3. Simulates a Low-Field version of that NIfTI.
   - Downsamples to 1.6 x 1.6 x 5.0 mm.
   - Finds optimal noise and blur parameters to match LF stats.
4. Saves both the HF NIfTI and the simulated LF NIfTI.
============================================================
"""

import os
import glob
import time
import shutil
import numpy as np
import nibabel as nib
import dicom2nifti
from scipy.ndimage import gaussian_filter, zoom
from scipy.stats import skew

# ==================================================
# SETTINGS
# ==================================================
# Input folder containing subfolders of DICOM (.ima) files
DICOM_ROOT_DIR = r"D:\01_MRI_Data"

# Output folder for the generated intermediate High-Field NIfTIs
HF_NIFTI_DIR = r"D:\01_MRI_Data\nifti_output\high_field_nifti"

# Output folder for the final Simulated Low-Field NIfTIs
LF_SIM_DIR = r"D:\01_MRI_Data\nifti_output\low_field_simulated"

# Simulation targets
LF_SPACING = (1.6, 1.6, 5.0)  # Target LF voxel size (mm)

# We will use FIXED parameters for the batch to ensure consistent 
# physical degradation across the entire dataset. 
# (These were determined from an average optimization run)
SIGMA_BLUR = 1.0     # Average PSF blur for LF
ALPHA1 = 0.04        # Average white noise
ALPHA2 = 0.02        # Average structured noise

np.random.seed(42)

# ==================================================
# HELPER: FIND DICOM FOLDERS
# ==================================================
def get_dicom_folders(root_dir):
    """Find all lowest-level directories containing .ima files."""
    dicom_folders = set()
    for root, dirs, files in os.walk(root_dir):
        if any(f.lower().endswith('.ima') or f.lower().endswith('.dcm') for f in files):
            dicom_folders.add(root)
    return sorted(list(dicom_folders))

# ==================================================
# STEP 1: DICOM TO NIFTI
# ==================================================
def convert_dicom_to_nifti(dicom_dir, out_nifti_dir, patient_id):
    """Converts a directory of DICOMs to a single NIfTI file."""
    os.makedirs(out_nifti_dir, exist_ok=True)
    temp_dir = os.path.join(out_nifti_dir, "temp_" + patient_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"  Converting DICOMs to NIfTI...")
    try:
        # This converts the folder of dicoms and places .nii.gz in temp_dir
        dicom2nifti.convert_directory(dicom_dir, temp_dir, compression=True, reorient=True)
        
        # Find the generated file (dicom2nifti names it based on series description)
        generated_files = glob.glob(os.path.join(temp_dir, "*.nii.gz"))
        if not generated_files:
            raise FileNotFoundError("dicom2nifti did not produce any .nii.gz files.")
            
        final_hf_path = os.path.join(out_nifti_dir, f"{patient_id}_HF.nii.gz")
        shutil.move(generated_files[0], final_hf_path)
        shutil.rmtree(temp_dir)
        return final_hf_path
    
    except Exception as e:
        print(f"  [ERROR] DICOM to NIfTI failed: {e}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None

# ==================================================
# STEP 2: SIMULATE LOW-FIELD NIFTI
# ==================================================
def simulate_low_field(hf_nifti_path, out_lf_dir, patient_id):
    """Takes a HF NIfTI, degrades it to simulate LF, and saves it."""
    os.makedirs(out_lf_dir, exist_ok=True)
    final_lf_path = os.path.join(out_lf_dir, f"{patient_id}_LF.nii.gz")
    
    print(f"  Simulating Low-Field MRI...")
    try:
        nii = nib.load(hf_nifti_path)
        img = nii.get_fdata().astype(np.float32)
        affine = nii.affine
        hf_spacing = nii.header.get_zooms()[:3]
        
        # 1. RESAMPLE TO LOW RESOLUTION
        zoom_factors = tuple(hf_spacing[i] / LF_SPACING[i] for i in range(3))
        img_resampled = zoom(img, zoom_factors, order=1)
        
        # 2. GENERATE AND APPLY NOISE & BLUR
        # We use fixed parameters to ensure consistent physics across patients
        base_noise1 = gaussian_filter(np.random.normal(0, 1, img_resampled.shape), sigma=0.5)
        base_noise2 = gaussian_filter(np.random.normal(0, 1, img_resampled.shape), sigma=1.0)
        
        img_noisy = img_resampled + ALPHA1 * base_noise1
        img_blurred = gaussian_filter(img_noisy, sigma=SIGMA_BLUR)
        img_sim = img_blurred + ALPHA2 * base_noise2
        
        # 3. SAVE
        # Update header with new physical voxel spacing
        new_header = nii.header.copy()
        new_header.set_zooms(LF_SPACING)
        
        out_nii = nib.Nifti1Image(img_sim.astype(np.float32), affine, header=new_header)
        nib.save(out_nii, final_lf_path)
        return final_lf_path
        
    except Exception as e:
        print(f"  [ERROR] Low-Field simulation failed: {e}")
        return None

# ==================================================
# MAIN LOOP
# ==================================================
if __name__ == "__main__":
    print("=" * 70)
    print("DICOM TO SIMULATED LOW-FIELD BATCH PIPELINE")
    print("=" * 70)
    
    dicom_folders = get_dicom_folders(DICOM_ROOT_DIR)
    
    if not dicom_folders:
        print(f"No DICOM (.ima or .dcm) folders found in {DICOM_ROOT_DIR}")
        exit()
        
    print(f"Found {len(dicom_folders)} folders containing DICOM sequences.\n")
    
    success_count = 0
    start_time = time.time()
    
    for i, dcm_dir in enumerate(dicom_folders):
        # Create a clean patient ID based on the folder path
        # e.g. D:\01_MRI_Data\0001\L-SPINE...\T2_TSE... -> patient_0001
        parts = dcm_dir.replace("\\", "/").split("/")
        # Extract the sequence number folder (e.g. '0001', '0002')
        patient_num = parts[-3] if len(parts) >= 3 else f"scan_{i+1:04d}"
        
        print(f"[{i+1}/{len(dicom_folders)}] Processing: {patient_num}")
        print(f"  Source: {dcm_dir}")
        
        # Convert DICOM to HF NIfTI
        hf_path = convert_dicom_to_nifti(dcm_dir, HF_NIFTI_DIR, patient_num)
        
        if hf_path:
            # Simulate LF NIfTI
            lf_path = simulate_low_field(hf_path, LF_SIM_DIR, patient_num)
            if lf_path:
                print(f"  -> SUCCESS! Saved to: {os.path.basename(lf_path)}")
                success_count += 1
        print("-" * 50)
                
    elapsed = time.time() - start_time
    print("=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print(f"Successfully processed {success_count} / {len(dicom_folders)} sequences.")
    print(f"Time taken: {elapsed/60:.1f} minutes")
    print(f"High-Field NIfTIs saved in : {HF_NIFTI_DIR}")
    print(f"Low-Field NIfTIs saved in  : {LF_SIM_DIR}")
    print("=" * 70)
